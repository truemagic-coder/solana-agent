# Bundled Tool Reference

This document is adapted from the Solana Agent Kit README so the full bundled tool surface lives inside this repo.

All of the tools below are bundled directly in `solana_agent.tools`. You do not need a separate `sakit` install when you are using the current Solana Agent runtime.

## General Notes

- Put secrets in environment variables rather than hard-coding them in config.
- For token operations, prefer mint addresses over tickers in user prompts.
- Helius RPC is recommended anywhere reliable transaction landing matters.
- Privy transaction tools generally require your Privy app credentials, a wallet authorization signing key, and a runtime wallet identifier when acting on a delegated wallet.

## Trading, Execution, And Lending

### Solana Transfer

Transfer SOL and SPL tokens from the agent wallet to a destination wallet.

```python
config = {
    "tools": {
        "solana_transfer": {
            "rpc_url": "https://mainnet.helius-rpc.com/?api-key=YOUR_KEY",
            "private_key": "your-base58-private-key",
        },
    },
}
```

### Solana Ultra

Swap tokens through Jupiter Ultra with automatic slippage handling, priority fees, and reliable transaction landing.

```python
config = {
    "tools": {
        "solana_ultra": {
            "private_key": "your-base58-private-key",
            "jupiter_api_key": "your-jupiter-api-key",
            "rpc_url": "https://mainnet.helius-rpc.com/?api-key=YOUR_KEY",
            "referral_account": "your-referral-account",
            "referral_fee": 50,
            "payer_private_key": "optional-gas-payer-key",
        },
    },
}
```

Notes:

- Jupiter API keys are available from `portal.jup.ag`.
- `payer_private_key` lets you sponsor gas for fully gasless flows.
- Referral fees can be collected when `referral_account` is configured.

### Solana Ultra Quote

Preview Jupiter Ultra swap details before execution.

```python
config = {
    "tools": {
        "solana_ultra_quote": {
            "private_key": "your-base58-private-key",
            "jupiter_api_key": "your-jupiter-api-key",
            "referral_account": "your-referral-account",
            "referral_fee": 50,
            "payer_private_key": "optional-gas-payer-key",
        },
    },
}
```

Typical flow:

1. Call `solana_ultra_quote`.
2. Show the user the quoted amounts, slippage, and price impact.
3. Call `solana_ultra` only after approval.

### Solana DFlow Swap

Fast swaps through DFlow with a direct Solana keypair.

```python
config = {
    "tools": {
        "solana_dflow_swap": {
            "private_key": "your-base58-private-key",
            "payer_private_key": "optional-gas-payer-key",
            "rpc_url": "https://api.mainnet-beta.solana.com",
        },
    },
}
```

Use Jupiter Ultra instead if you need referral or platform fee collection.

### Jupiter Trigger

Create, cancel, and manage Jupiter Trigger limit orders.

```python
config = {
    "tools": {
        "jupiter_trigger": {
            "private_key": "your-base58-private-key",
            "jupiter_api_key": "your-jupiter-api-key",
            "rpc_url": "https://mainnet.helius-rpc.com/?api-key=YOUR_KEY",
            "referral_account": "your-referral-account",
            "referral_fee": 50,
            "payer_private_key": "optional-gas-payer-key",
        },
    },
}
```

Actions:

- `create`
- `cancel`
- `cancel_all`
- `list`

### Jupiter Recurring

Create and manage DCA orders through Jupiter Recurring.

```python
config = {
    "tools": {
        "jupiter_recurring": {
            "private_key": "your-base58-private-key",
            "jupiter_api_key": "your-jupiter-api-key",
            "payer_private_key": "optional-gas-payer-key",
        },
    },
}
```

Actions:

- `create`
- `cancel`
- `list`

Important create parameters:

- `input_mint`
- `output_mint`
- `in_amount`
- `order_count`
- `frequency`
- optional `min_out_amount`, `max_out_amount`, and `start_at`

### Jupiter Earn

Deposit, withdraw, mint, and redeem through Jupiter Earn.

```python
config = {
    "tools": {
        "jupiter_earn": {
            "private_key": "your-base58-private-key",
            "jupiter_api_key": "your-jupiter-api-key",
            "rpc_url": "https://mainnet.helius-rpc.com/?api-key=YOUR_KEY",
        },
    },
}
```

Actions:

- `deposit`
- `withdraw`
- `mint`
- `redeem`
- `tokens`
- `positions`
- `earnings`

Current scope is SOL and USDC only.

### Jupiter Holdings

Fetch wallet token holdings and USD values through Jupiter Ultra.

```python
config = {
    "tools": {
        "jupiter_holdings": {
            "jupiter_api_key": "your-jupiter-api-key",
        },
    },
}
```

Returns balances, token metadata, and total portfolio value.

### Jupiter Shield

Fetch Jupiter token security warnings and risk flags.

```python
config = {
    "tools": {
        "jupiter_shield": {
            "jupiter_api_key": "your-jupiter-api-key",
        },
    },
}
```

### Jupiter Token Search

Search Solana tokens by symbol, name, or address.

```python
config = {
    "tools": {
        "jupiter_token_search": {
            "jupiter_api_key": "your-jupiter-api-key",
        },
    },
}
```

Returns mint addresses, symbols, names, and token metadata.

### Kamino

Use Kamino Earn and K-Lend, and query Kamino public API data.

```python
config = {
    "tools": {
        "kamino": {
            "private_key": "your-base58-private-key",
            "rpc_url": "https://mainnet.helius-rpc.com/?api-key=YOUR_KEY",
            "base_url": "https://api.kamino.finance",
            "referrer_private_key": "optional-referrer-key",
        },
    },
}
```

Actions:

- `earn_deposit`
- `earn_withdraw`
- `borrow_deposit`
- `borrow_borrow`
- `borrow_repay`
- `borrow_withdraw`
- `list_vaults`
- `list_markets`
- `oracle_prices`
- `vault_positions`
- `user_obligations`
- `api_get`
- `api_post`

## Privy Wallet Tools

### Privy Transfer

Transfer SOL and SPL tokens using Privy delegated wallets with a sponsored fee payer.

```python
config = {
    "tools": {
        "privy_transfer": {
            "app_id": "your-privy-app-id",
            "app_secret": "your-privy-app-secret",
            "signing_key": "wallet-auth:your-signing-key",
            "rpc_url": "https://mainnet.helius-rpc.com/?api-key=YOUR_KEY",
            "fee_payer": "your-fee-payer-private-key",
        },
    },
}
```

### Privy Kamino

Run Kamino Earn and K-Lend flows with Privy delegated wallets.

```python
config = {
    "tools": {
        "privy_kamino": {
            "app_id": "your-privy-app-id",
            "app_secret": "your-privy-app-secret",
            "signing_key": "wallet-auth:your-signing-key",
            "rpc_url": "https://mainnet.helius-rpc.com/?api-key=YOUR_KEY",
            "base_url": "https://api.kamino.finance",
            "referrer_private_key": "optional-referrer-key",
        },
    },
}
```

Actions are the same as `kamino`.

### Privy Privacy Cash

Deposit, withdraw, transfer, and check balances through PrivacyCash.

```python
config = {
    "tools": {
        "privy_privacy_cash": {
            "api_key": "your-privacycash-api-key",
            "base_url": "https://cash.solana-agent.com",
        },
    },
}
```

Actions:

- `transfer`
- `deposit`
- `withdraw`
- `balance`

### Privy Ultra

Run Jupiter Ultra swaps through Privy delegated wallets.

```python
config = {
    "tools": {
        "privy_ultra": {
            "app_id": "your-privy-app-id",
            "app_secret": "your-privy-app-secret",
            "signing_key": "wallet-auth:your-signing-key",
            "jupiter_api_key": "your-jupiter-api-key",
            "rpc_url": "https://mainnet.helius-rpc.com/?api-key=YOUR_KEY",
            "referral_account": "your-referral-account",
            "referral_fee": 50,
            "payer_private_key": "optional-gas-payer-key",
        },
    },
}
```

Use this when you want the same Jupiter Ultra behavior as `solana_ultra`, but with embedded Privy wallets.

### Privy Ultra Quote

Preview Jupiter Ultra swap details for Privy wallets before execution.

```python
config = {
    "tools": {
        "privy_ultra_quote": {
            "app_id": "your-privy-app-id",
            "app_secret": "your-privy-app-secret",
            "jupiter_api_key": "your-jupiter-api-key",
            "referral_account": "your-referral-account",
            "referral_fee": 50,
            "payer_private_key": "optional-gas-payer-key",
        },
    },
}
```

Typical flow is `privy_ultra_quote` first, then `privy_ultra` after approval.

### Privy Trigger

Create, cancel, and manage Jupiter Trigger limit orders with Privy delegated wallets.

```python
config = {
    "tools": {
        "privy_trigger": {
            "app_id": "your-privy-app-id",
            "app_secret": "your-privy-app-secret",
            "signing_key": "wallet-auth:your-signing-key",
            "jupiter_api_key": "your-jupiter-api-key",
            "rpc_url": "https://mainnet.helius-rpc.com/?api-key=YOUR_KEY",
            "referral_account": "your-referral-account",
            "referral_fee": 50,
            "payer_private_key": "optional-gas-payer-key",
        },
    },
}
```

Actions are the same as `jupiter_trigger`.

### Privy Recurring

Create and manage Jupiter DCA orders with Privy delegated wallets.

```python
config = {
    "tools": {
        "privy_recurring": {
            "app_id": "your-privy-app-id",
            "app_secret": "your-privy-app-secret",
            "signing_key": "wallet-auth:your-signing-key",
            "jupiter_api_key": "your-jupiter-api-key",
            "payer_private_key": "optional-gas-payer-key",
        },
    },
}
```

Actions are the same as `jupiter_recurring`.

### Privy Earn

Use Jupiter Earn with Privy delegated wallets.

```python
config = {
    "tools": {
        "privy_earn": {
            "app_id": "your-privy-app-id",
            "app_secret": "your-privy-app-secret",
            "signing_key": "wallet-auth:your-signing-key",
            "jupiter_api_key": "your-jupiter-api-key",
            "rpc_url": "https://mainnet.helius-rpc.com/?api-key=YOUR_KEY",
        },
    },
}
```

Actions are the same as `jupiter_earn`.

### Privy DFlow Swap

Run DFlow swaps through Privy delegated wallets.

```python
config = {
    "tools": {
        "privy_dflow_swap": {
            "app_id": "your-privy-app-id",
            "app_secret": "your-privy-app-secret",
            "signing_key": "wallet-auth:your-signing-key",
            "rpc_url": "https://mainnet.helius-rpc.com/?api-key=YOUR_KEY",
            "payer_private_key": "optional-gas-payer-key",
        },
    },
}
```

Use Jupiter Ultra instead if you need referral or platform fee collection.

### Privy Wallet Address

Fetch the wallet address for a Privy delegated wallet.

```python
config = {
    "tools": {
        "privy_wallet_address": {
            "app_id": "your-privy-app-id",
            "app_secret": "your-privy-app-secret",
        },
    },
}
```

### Privy Create User

Create a new Privy user linked to a Telegram account for bot-first flows.

```python
config = {
    "tools": {
        "privy_create_user": {
            "app_id": "your-privy-app-id",
            "app_secret": "your-privy-app-secret",
        },
    },
}
```

Parameter:

- `telegram_user_id`

Returns user id, creation time, and linked-account metadata.

### Privy Create Wallet

Create a wallet for an existing Privy user.

```python
config = {
    "tools": {
        "privy_create_wallet": {
            "app_id": "your-privy-app-id",
            "app_secret": "your-privy-app-secret",
            "owner_id": "optional-owner-id",
        },
    },
}
```

Parameters:

- `user_id`
- optional `chain_type`

### Privy Get User by Telegram

Look up an existing Privy user by Telegram id.

```python
config = {
    "tools": {
        "privy_get_user_by_telegram": {
            "app_id": "your-privy-app-id",
            "app_secret": "your-privy-app-secret",
        },
    },
}
```

Parameter:

- `telegram_user_id`

Returns `status`, `user_id`, discovered wallets, and `has_wallet`.

## Prediction, Analytics, And Research

### DFlow Prediction Market

Discover and trade prediction markets on Solana through DFlow with built-in safety scoring.

```python
config = {
    "tools": {
        "dflow_prediction": {
            "private_key": "your-base58-private-key",
            "rpc_url": "https://mainnet.helius-rpc.com/?api-key=YOUR_KEY",
            "platform_fee_bps": 50,
            "fee_account": "your-usdc-fee-account",
            "min_volume_usd": 1000,
            "min_liquidity_usd": 500,
            "include_risky": False,
        },
    },
}
```

Discovery actions:

- `search`
- `list_events`
- `get_event`
- `list_markets`
- `get_market`

Trading actions:

- `buy`
- `sell`
- `positions`

Safety levels are `HIGH`, `MEDIUM`, and `LOW`. Treat low-scoring markets with caution.

### Privy DFlow Prediction

Use the same DFlow prediction-market surface with Privy delegated wallets.

```python
config = {
    "tools": {
        "privy_dflow_prediction": {
            "app_id": "your-privy-app-id",
            "app_secret": "your-privy-app-secret",
            "signing_key": "wallet-auth:your-signing-key",
            "rpc_url": "https://mainnet.helius-rpc.com/?api-key=YOUR_KEY",
            "platform_fee_bps": 50,
            "fee_account": "your-usdc-fee-account",
            "min_volume_usd": 1000,
            "min_liquidity_usd": 500,
            "include_risky": False,
            "payer_private_key": "optional-gas-payer-key",
        },
    },
}
```

Actions are the same as `dflow_prediction`, with `privy_user_id` required for trading.

### Rugcheck

Check whether a token is likely to be a rug pull.

No config is required.

### Vybe

Label known Solana wallets such as exchanges, market makers, pools, treasuries, and other entities.

```python
config = {
    "tools": {
        "vybe": {
            "api_key": "your-vybe-api-key",
        },
    },
}
```

Parameters:

- `addresses`
- optional `refresh_cache`

### Birdeye

Comprehensive market, token, trader, and wallet analytics through the Birdeye API.

```python
config = {
    "tools": {
        "birdeye": {
            "api_key": "your-birdeye-api-key",
            "chain": "solana",
        },
    },
}
```

Major action groups:

- Price data: `price`, `multi_price`, `history_price`, `historical_price_unix`, `price_volume_single`, `price_volume_multi`
- OHLCV data: `ohlcv`, `ohlcv_pair`, `ohlcv_base_quote`, `ohlcv_v3`, `ohlcv_pair_v3`
- Trade data: `trades_token`, `trades_pair`, `trades_token_seek`, `trades_pair_seek`, `trades_v3`, `trades_token_v3`
- Token information: `token_list`, `token_overview`, `token_metadata_single`, `token_metadata_multiple`, `token_market_data`, `token_trade_data_single`, `token_holder`, `token_trending`, `token_new_listing`, `token_top_traders`, `token_markets`, `token_security`, `token_creation_info`, `token_mint_burn`, `token_all_time_trades_single`, `token_all_time_trades_multiple`
- Pair data: `pair_overview_single`, `pair_overview_multiple`
- Trader data: `trader_gainers_losers`, `trader_txs_seek`
- Wallet data: `wallet_token_list`, `wallet_token_balance`, `wallet_tx_list`, `wallet_balance_change`, `wallet_pnl_summary`, `wallet_pnl_details`, `wallet_pnl_multiple`, `wallet_current_net_worth`, `wallet_net_worth`, `wallet_net_worth_details`
- Exit liquidity: `token_exit_liquidity`, `token_exit_liquidity_multiple`
- Search and utilities: `search`, `latest_block`, `networks`, `supported_chains`

### Internet Search

Search the live web through OpenAI, Perplexity, or Grok.

```python
config = {
    "tools": {
        "search_internet": {
            "api_key": "your-search-api-key",
            "provider": "openai",
            "citations": True,
            "model": "gpt-4o-mini-search-preview",
            "grok_web_search": True,
            "grok_x_search": True,
            "grok_timeout": 90,
        },
    },
}
```

Available models:

- Perplexity: `sonar`, `sonar-pro`
- OpenAI: `gpt-4o-mini-search-preview`, `gpt-4o-search-preview`
- Grok: `grok-4-1-fast-non-reasoning`, `grok-4-fast`, `grok-4.1-fast`

### Technical Analysis

Compute technical indicators from Birdeye OHLCV data using `pandas-ta`.

```python
config = {
    "tools": {
        "technical_analysis": {
            "api_key": "your-birdeye-api-key",
        },
    },
}
```

Parameters:

- `address`
- optional `timeframe`

Indicator groups returned:

- Moving averages: EMA 9, 21, 50, 200 and SMA 20, 50, 200
- Momentum: MACD, RSI, stochastic, Williams %R, ROC, MFI, CCI, ADX
- Volatility: Bollinger Bands, ATR, Keltner Channels
- Volume: OBV and VWAP
- Support and resistance clusters
- Price-versus-indicator deltas for fast interpretation

## Infrastructure, Media, And Utilities

### MCP

Connect Solana Agent to one or more MCP servers and let OpenAI or Grok choose tools.

Simple single-server config:

```python
config = {
    "ai": {
        "api_key": "your-openai-api-key",
    },
    "tools": {
        "mcp": {
            "url": "https://your-mcp-server.example/api",
            "headers": {
                "Authorization": "Bearer your-token",
            },
            "llm_provider": "openai",
            "llm_model": "gpt-4.1-mini",
        },
    },
}
```

Multi-server config uses `servers` instead of a single `url`, with per-server headers.

### Image Generation

Generate images through OpenAI, Grok, or Gemini and upload them to S3-compatible storage.

```python
config = {
    "tools": {
        "image_gen": {
            "provider": "openai",
            "api_key": "your-image-api-key",
            "s3_endpoint_url": "https://your-s3-endpoint.example",
            "s3_access_key_id": "YOUR_S3_ACCESS_KEY",
            "s3_secret_access_key": "YOUR_S3_SECRET_KEY",
            "s3_bucket_name": "your-bucket-name",
            "s3_region_name": "optional-region",
            "s3_public_url_base": "https://cdn.example/",
        },
    },
}
```

Supported image models:

- OpenAI: `gpt-image-1`
- Grok: `grok-2-image`
- Gemini: `imagen-3.0-generate-002`

### Token Math

Reliable amount conversion and order-math helper for swaps, transfers, and limit orders.

No external config is required.

Actions:

- `swap`
- `transfer`
- `limit_order`
- `limit_order_info`
- `to_smallest_units`
- `to_human`

Use this before executing swaps, transfers, and limit orders. It exists because LLM arithmetic is not trustworthy for token amounts.

### x402 Request Helper

Issue generic GET and POST requests against x402-protected endpoints using either a direct Solana private key or Privy-backed signing.

```python
config = {
    "tools": {
        "x402_request": {
            "auth_mode": "x402_private_key",
            "private_key": "your-base58-private-key",
            "allowed_hosts": ["your-x402-service.example"],
            "x402_rpc_url": "https://mainnet.helius-rpc.com/?api-key=YOUR_KEY",
        },
    },
}
```

For Privy-backed signing, configure `auth_mode`, `privy_app_id`, and `privy_app_secret`, then provide `runtime_context["privy_wallet_id"]` at request time.

Supported request fields:

- `method`: `GET` or `POST`
- `url`
- optional `query_params`
- optional `headers`
- optional `json_body`
- optional `timeout_seconds`

Security note: `allowed_hosts` must be configured before this tool can be used.