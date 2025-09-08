# Realtime audio session flow

This documents the realtime audio flow implemented by `solana_agent/adapters/openai_realtime_ws.py`, using GA-compliant events and sequencing. It reflects only the events and ordering used in the code.

## High-level
- Single WebSocket to OpenAI Realtime, session.type="realtime".
- Session-wide voice and audio formats are applied via `session.update` before any `response.create`.
- Input audio is streamed with `input_audio_buffer.append` and committed with `input_audio_buffer.commit`.
- When VAD is disabled, client triggers `response.create`; when enabled, the server may auto-create a response after commit.
- Audio out and transcripts stream over WS. Text and transcript deltas are optional but supported.
- Function calling follows GA flow: server emits a function_call item; client executes the tool; client posts a `function_call_output` item; client then sends `response.create` to continue.

## ASCII sequence (happy path)

```
Client                                              OpenAI Realtime Server
  |                                                               |
  |--------------------- Connect (WS) ---------------------------->|
  |                                                               |
  |<------------------------ session.created ---------------------|
  |                                                               |
  |-- session.update (session.type=realtime, audio.input/output,  |
  |                  output_modalities=["audio"], voice, tools) -->|
  |                                                               |
  |<------------------------- session.updated --------------------|
  |                                                               |
  |-- input_audio_buffer.append (base64 PCM16) ------------------->|  (repeatable)
  |-- input_audio_buffer.append (base64 PCM16) ------------------->|
  |-- input_audio_buffer.commit --------------------------------->|
  |                                                               |
  |<------------------- input_audio_buffer.committed -------------|  (item_id set)
  |                                                               |
  |-- response.create (gated; skipped if server auto-creates) --->|  (binds last input item)
  |                                                               |
  |<----------------- response.output_audio.delta -----------------|  (streaming PCM16)
  |<---------------- conversation.item.input_audio_transcription.. |  (input transcript deltas, optional)
  |<--------------------- response.output_text.delta -------------|  (assistant text deltas, optional)
  |                                                               |
  |<------------- conversation.item.created (function_call) ------|  (call_id)
  |                                                               |
  |<--------------------------- response.done --------------------|  (includes output items)
  |          (response.output[].type=function_call, status=completed)
  |                                                               |
  |== Execute tool locally (timeout-supported) ====================|
  |                                                               |
  |-- conversation.item.create (function_call_output, call_id, --->|
  |   output=JSON-string)                                         |
  |                                                               |
  |<----------- conversation.item.created (function_call_output) --|  (ack observed)
  |                                                               |
  |-- response.create (continue model after tool result) --------->|
  |                                                               |
  |<------------------- response.output_audio.delta --------------|  (more audio)
  |<----------------------- response.audio.done ------------------|  (or response.output_audio.done)
  |                                                               |
  |-- input_audio_buffer.append / commit (next turn) ------------->|  (repeat)
  |                                                               |
```

## Client -> Server events (as used)
- `session.update` with nested shape:
  - `session.type = "realtime"`
  - `session.output_modalities = ["audio"]`
  - `session.audio.input.format = { type, rate }`
  - `session.audio.input.turn_detection = { type: "server_vad", create_response: bool } | null`
  - `session.audio.output.format = { type, rate }`
  - `session.audio.output.voice = <Literal voice id>`
  - `session.audio.output.speed = <float>`
  - `session.instructions = <string>`
  - `session.tools = [...]` (GA schema; unsupported fields like `strict` are stripped)
  - `session.tool_choice` (optional)
- `input_audio_buffer.append` { audio: base64(PCM16) }
- `input_audio_buffer.commit`
- `input_audio_buffer.clear` (optional; resets state)
- `response.create` { response: { metadata: { type: "response" }, input: [ { type: "item_reference", id } ]? } }
- `conversation.item.create` with `item.type = "function_call_output"` including:
  - `call_id`
  - `output` (stringified JSON)

## Server -> Client events (as handled)
- Session lifecycle
  - `session.created`
  - `session.updated` (echoes applied config; used to gate first response)
- Input buffer
  - `input_audio_buffer.committed` (provides `item_id` for input reference)
- Output audio and transcripts
  - `response.output_audio.delta` or `response.audio.delta` (PCM16 chunk)
  - `response.output_audio.done` or `response.audio.done`
  - `response.output_audio_transcript.delta` or `response.audio_transcript.delta` (assistant transcript deltas)
- Input transcription (optional)
  - `conversation.item.input_audio_transcription.delta`
  - `conversation.item.input_audio_transcription.completed`
- Responses
  - `response.output_text.delta` (assistant text deltas)
  - `response.done` / `response.completed` / `response.complete`
    - May include `response.output[]` items; when a `function_call` item appears with `status=completed`, the client executes the tool.
- Function calling
  - `conversation.item.created` with `item.type = "function_call"` (readiness for the call_id)
  - `conversation.item.created` with `item.type = "function_call_output"` (ack for our posted output)
  - Legacy deltas like `response.function_call*.delta` are ignored.
- Errors (surfaced with correlation):
  - `error`, `*.error`, or events containing `failed` (correlated by `event_id` when available)

## Guardrails and sequencing (as implemented)
- `response.create` gating:
  - Wait for `session.updated` (or `session.created` fallback) so voice/instructions apply.
  - Skip if a response is already active (`conversation_already_has_active_response` prevention).
  - If server VAD auto-create is enabled and a commit happened very recently, skip manual `response.create`.
- Commit robustness:
  - Skip commit when response is active, during in-flight commit, or within a 1s debounce.
  - Require a minimum buffer (~100ms of audio) before commit.
- Function calls:
  - Execute only when a `function_call` item is present with status `completed` (from `response.done`).
  - Dedupe by `call_id` and bind each call to the active response generation.
  - Wait for `conversation.item.created` (function_call) readiness before posting `function_call_output`.
  - After posting `function_call_output`, wait for an ack `conversation.item.created` (function_call_output); retry once if no ack is seen. Only after ack do we send `response.create` to continue.
  - Tool execution supports long-running calls (timeout configured via options; default 300s). Optional freshness window can skip stale outputs.
- Event correlation:
  - Client sends carry an `event_id` and are stored for error correlation.

## Notes
- Audio formats are PCM16 24kHz by default, both in and out (configurable via options).
- Session-wide voice is hidden from user prompts and set only in the session.
- Input references: after `input_audio_buffer.committed`, the last `item_id` is attached to `response.create` so the model binds the response to the right audio.
- Transcription-only mode uses a different session (`transcription_session.update`); that flow is separate from this realtime audio flow.
