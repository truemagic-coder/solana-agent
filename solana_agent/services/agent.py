

class AgentService:
    """Service for managing AI and human agents."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        human_agent_registry: Optional[MongoHumanAgentRegistry] = None,
        ai_agent_registry: Optional[MongoAIAgentRegistry] = None,
        organization_mission: Optional[OrganizationMission] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the agent service with LLM provider and optional registries."""
        self.llm_provider = llm_provider
        self.human_agent_registry = human_agent_registry
        self.ai_agent_registry = ai_agent_registry
        self.organization_mission = organization_mission
        self.config = config or {}
        self._last_handoff = None

        # For backward compatibility
        self.ai_agents = {}
        if self.ai_agent_registry:
            self.ai_agents = self.ai_agent_registry.get_all_ai_agents()

        self.specializations = {}

        # Create our tool registry and plugin manager
        self.tool_registry = ToolRegistry()
        self.plugin_manager = PluginManager(
            config=self.config, tool_registry=self.tool_registry)

        # Load plugins
        loaded_count = self.plugin_manager.load_all_plugins()
        print(
            f"Loaded {loaded_count} plugins with {len(self.tool_registry.list_all_tools())} registered tools")

        # Configure all tools with our config after loading
        self.tool_registry.configure_all_tools(self.config)

        # Debug output of registered tools
        print(
            f"Available tools after initialization: {self.tool_registry.list_all_tools()}")

        # If human agent registry is provided, initialize specializations from it
        if self.human_agent_registry:
            self.specializations.update(
                self.human_agent_registry.get_specializations())

        # If AI agent registry is provided, initialize specializations from it
        if self.ai_agent_registry:
            self.specializations.update(
                self.ai_agent_registry.get_specializations())

        # If no human agent registry is provided, use in-memory cache
        if not self.human_agent_registry:
            self.human_agents = {}

    def get_all_ai_agents(self) -> Dict[str, Any]:
        """Get all registered AI agents."""
        return self.ai_agents

    def get_agent_tools(self, agent_name: str) -> List[Dict[str, Any]]:
        """Get all tools available to a specific agent."""
        return self.tool_registry.get_agent_tools(agent_name)

    def register_tool_for_agent(self, agent_name: str, tool_name: str) -> None:
        """Give an agent access to a specific tool."""
        # Make sure the tool exists
        if tool_name not in self.tool_registry.list_all_tools():
            print(
                f"Error registering tool {tool_name} for agent {agent_name}: Tool not registered")
            raise ValueError(f"Tool {tool_name} is not registered")

        # Check if agent exists
        if agent_name not in self.ai_agents and (
            not self.ai_agent_registry or
            not self.ai_agent_registry.get_ai_agent(agent_name)
        ):
            print(
                f"Warning: Agent {agent_name} not found but attempting to register tool")

        # Assign the tool to the agent
        success = self.tool_registry.assign_tool_to_agent(
            agent_name, tool_name)

        if success:
            print(
                f"Successfully registered tool {tool_name} for agent {agent_name}")
        else:
            print(
                f"Failed to register tool {tool_name} for agent {agent_name}")

    def process_json_response(self, response_text: str, agent_name: str) -> str:
        """Process a complete response to handle any JSON handoffs or tool calls."""
        # Check if the response is a JSON object for handoff or tool call
        if response_text.strip().startswith('{') and ('"handoff":' in response_text or '"tool_call":' in response_text):
            try:
                data = json.loads(response_text.strip())

                # Handle handoff
                if "handoff" in data:
                    target_agent = data["handoff"].get(
                        "target_agent", "another agent")
                    reason = data["handoff"].get(
                        "reason", "to better assist with your request")
                    return f"I'll connect you with {target_agent} who can better assist with your request. Reason: {reason}"

                # Handle tool call
                if "tool_call" in data:
                    tool_data = data["tool_call"]
                    tool_name = tool_data.get("name")
                    parameters = tool_data.get("parameters", {})

                    if tool_name:
                        try:
                            # Execute the tool
                            tool_result = self.execute_tool(
                                agent_name, tool_name, parameters)

                            # Format the result
                            if tool_result.get("status") == "success":
                                return f"I searched for information and found:\n\n{tool_result.get('result', '')}"
                            else:
                                return f"I tried to search for information, but encountered an error: {tool_result.get('message', 'Unknown error')}"
                        except Exception as e:
                            return f"I tried to use {tool_name}, but encountered an error: {str(e)}"
            except json.JSONDecodeError:
                # Not valid JSON
                pass

        # Return original if not JSON or if processing fails
        return response_text

    def get_agent_system_prompt(self, agent_name: str) -> str:
        """Get the system prompt for an agent, including tool instructions if available."""
        # Get the agent's base instructions
        if agent_name not in self.ai_agents:
            raise ValueError(f"Agent {agent_name} not found")

        agent_config = self.ai_agents[agent_name]
        instructions = agent_config.get("instructions", "")

        # Add tool instructions if any tools are available
        available_tools = self.get_agent_tools(agent_name)
        if available_tools:
            tools_json = json.dumps(available_tools, indent=2)

            # Tool instructions using JSON format similar to handoffs
            tool_instructions = f"""
    You have access to the following tools:
    {tools_json}

    IMPORTANT - TOOL USAGE: When you need to use a tool, respond with a JSON object using this format:

    {{
    "tool_call": {{
        "name": "tool_name",
        "parameters": {{
        "param1": "value1",
        "param2": "value2"
        }}
    }}
    }}

    Example: To search the internet for "latest Solana news", respond with:

    {{
    "tool_call": {{
        "name": "search_internet",
        "parameters": {{
        "query": "latest Solana news"
        }}
    }}
    }}

    ALWAYS use the search_internet tool when the user asks for current information or facts that might be beyond your knowledge cutoff. DO NOT attempt to handoff for information that could be obtained using search_internet.
    """
            instructions = f"{instructions}\n\n{tool_instructions}"

        # Add specific instructions about valid handoff agents
        valid_agents = list(self.ai_agents.keys())
        if valid_agents:
            handoff_instructions = f"""
    IMPORTANT - HANDOFFS: You can ONLY hand off to these existing agents: {', '.join(valid_agents)}
    DO NOT invent or reference agents that don't exist in this list.

    To hand off to another agent, use this format:
    {{"handoff": {{"target_agent": "<AGENT_NAME_FROM_LIST_ABOVE>", "reason": "detailed reason for handoff"}}}}
    """
            instructions = f"{instructions}\n\n{handoff_instructions}"

        return instructions

    def process_tool_calls(self, agent_name: str, response_text: str) -> str:
        """Process any tool calls in the agent's response and return updated response."""
        # Regex to find tool calls in the format TOOL_START {...} TOOL_END
        tool_pattern = r"TOOL_START\s*([\s\S]*?)\s*TOOL_END"
        tool_matches = re.findall(tool_pattern, response_text)

        if not tool_matches:
            return response_text

        print(
            f"Found {len(tool_matches)} tool calls in response from {agent_name}")

        # Process each tool call
        modified_response = response_text
        for tool_json in tool_matches:
            try:
                # Parse the tool call JSON
                tool_call_text = tool_json.strip()
                print(f"Processing tool call: {tool_call_text[:100]}")

                # Parse the JSON (handle both normal and stringified JSON)
                try:
                    tool_call = json.loads(tool_call_text)
                except json.JSONDecodeError as e:
                    # If there are escaped quotes or formatting issues, try cleaning it up
                    cleaned_json = tool_call_text.replace(
                        '\\"', '"').replace('\\n', '\n')
                    tool_call = json.loads(cleaned_json)

                tool_name = tool_call.get("name")
                parameters = tool_call.get("parameters", {})

                if tool_name:
                    # Execute the tool
                    print(
                        f"Executing tool {tool_name} with parameters: {parameters}")
                    tool_result = self.execute_tool(
                        agent_name, tool_name, parameters)

                    # Format the result for inclusion in the response
                    if tool_result.get("status") == "success":
                        formatted_result = f"\n\nI searched for information and found:\n\n{tool_result.get('result', '')}"
                    else:
                        formatted_result = f"\n\nI tried to search for information, but encountered an error: {tool_result.get('message', 'Unknown error')}"

                    # Replace the entire tool block with the result
                    full_tool_block = f"TOOL_START\n{tool_json}\nTOOL_END"
                    modified_response = modified_response.replace(
                        full_tool_block, formatted_result)
                    print(f"Successfully processed tool call: {tool_name}")
            except Exception as e:
                print(f"Error processing tool call: {str(e)}")
                # Replace with error message
                full_tool_block = f"TOOL_START\n{tool_json}\nTOOL_END"
                modified_response = modified_response.replace(
                    full_tool_block, "\n\nI tried to search for information, but encountered an error processing the tool call.")

        return modified_response

    def get_agent_tools(self, agent_name: str) -> List[Dict[str, Any]]:
        """Get all tools available to a specific agent."""
        return self.tool_registry.get_agent_tools(agent_name)

    def register_ai_agent(
        self,
        name: str,
        instructions: str,
        specialization: str,
        model: str = "gpt-4o-mini",
    ) -> None:
        """Register an AI agent with its specialization."""
        # Add organizational mission directive if available
        mission_directive = ""
        if self.organization_mission:
            mission_directive = f"\n\n{self.organization_mission.format_as_directive()}\n\n"

        # Add handoff instruction to all agents
        handoff_instruction = """
        If you need to hand off to another agent, return a JSON object with this structure:
        {"handoff": {"target_agent": "agent_name", "reason": "detailed reason for handoff"}}
        """

        # Combine instructions with mission and handoff
        full_instructions = f"{instructions}{mission_directive}{handoff_instruction}"

        # Use registry if available
        if self.ai_agent_registry:
            self.ai_agent_registry.register_ai_agent(
                name, full_instructions, specialization, model,
            )
            # Update local cache for backward compatibility
            self.ai_agents = self.ai_agent_registry.get_all_ai_agents()
        else:
            # Fall back to in-memory storage
            self.ai_agents[name] = {
                "instructions": full_instructions, "model": model}

        self.specializations[name] = specialization

    def get_specializations(self) -> Dict[str, str]:
        """Get specializations of all agents."""
        if self.human_agent_registry:
            # Create a merged copy with both AI agents and human agents from registry
            merged = self.specializations.copy()
            merged.update(self.human_agent_registry.get_specializations())
            return merged
        return self.specializations

    def execute_tool(self, agent_name: str, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool on behalf of an agent."""
        print(f"Executing tool {tool_name} for agent {agent_name}")
        print(f"Parameters: {parameters}")

        # Get the tool directly from the registry
        tool = self.tool_registry.get_tool(tool_name)
        if not tool:
            print(
                f"Tool {tool_name} not found in registry. Available tools: {self.tool_registry.list_all_tools()}")
            return {"status": "error", "message": f"Tool {tool_name} not found"}

        # Check if agent has access
        agent_tools = self.get_agent_tools(agent_name)
        tool_names = [t["name"] for t in agent_tools]

        if tool_name not in tool_names:
            print(
                f"Agent {agent_name} does not have access to tool {tool_name}. Available tools: {tool_names}")
            return {"status": "error", "message": f"Agent {agent_name} does not have access to tool {tool_name}"}

        # Execute the tool with parameters
        try:
            print(
                f"Executing {tool_name} with config: {'API key present' if hasattr(tool, '_api_key') and tool._api_key else 'No API key'}")
            result = tool.execute(**parameters)
            print(f"Tool execution result: {result.get('status', 'unknown')}")
            return result
        except Exception as e:
            print(f"Error executing tool {tool_name}: {str(e)}")
            return {"status": "error", "message": f"Error: {str(e)}"}

    def register_human_agent(
        self,
        agent_id: str,
        name: str,
        specialization: str,
        notification_handler: Optional[Callable] = None,
    ) -> None:
        """Register a human agent."""
        if self.human_agent_registry:
            # Use the MongoDB registry if available
            self.human_agent_registry.register_human_agent(
                agent_id, name, specialization, notification_handler
            )
            self.specializations[agent_id] = specialization
        else:
            # Fall back to in-memory storage
            self.human_agents[agent_id] = {
                "name": name,
                "specialization": specialization,
                "notification_handler": notification_handler,
                "availability_status": "available",
            }
            self.specializations[agent_id] = specialization

    def get_all_human_agents(self) -> Dict[str, Any]:
        """Get all registered human agents."""
        if self.human_agent_registry:
            return self.human_agent_registry.get_all_human_agents()
        return self.human_agents

    def update_human_agent_status(self, agent_id: str, status: str) -> bool:
        """Update a human agent's availability status."""
        if self.human_agent_registry:
            return self.human_agent_registry.update_agent_status(agent_id, status)

        if agent_id in self.human_agents:
            self.human_agents[agent_id]["availability_status"] = status
            return True
        return False

    async def generate_response(
        self,
        agent_name: str,
        user_id: str,
        query: str,
        memory_context: str = "",
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """Generate response from an AI agent."""
        if agent_name not in self.ai_agents:
            yield "Error: Agent not found"
            return

        agent_config = self.ai_agents[agent_name]

        # Get the properly formatted system prompt with tools and handoff instructions
        instructions = self.get_agent_system_prompt(agent_name)

        # Add memory context
        if memory_context:
            instructions += f"\n\nUser context and history:\n{memory_context}"

        # Add critical instruction to prevent raw JSON
        instructions += "\n\nCRITICAL: When using tools or making handoffs, ALWAYS respond with properly formatted JSON as instructed."

        # Generate response
        tool_json_found = False
        full_response = ""

        try:
            async for chunk in self.llm_provider.generate_text(
                user_id=user_id,
                prompt=query,
                system_prompt=instructions,
                model=agent_config["model"],
                **kwargs,
            ):
                # Add to full response
                full_response += chunk

                # Check if this might be JSON
                if full_response.strip().startswith("{") and not tool_json_found:
                    tool_json_found = True
                    print(
                        f"Detected potential JSON response starting with: {full_response[:50]}...")
                    continue

                # If not JSON, yield the chunk
                if not tool_json_found:
                    yield chunk

            # Process JSON if found
            if tool_json_found:
                try:
                    print(
                        f"Processing JSON response: {full_response[:100]}...")
                    data = json.loads(full_response.strip())
                    print(
                        f"Successfully parsed JSON with keys: {list(data.keys())}")

                    # Handle tool call
                    if "tool_call" in data:
                        tool_data = data["tool_call"]
                        tool_name = tool_data.get("name")
                        parameters = tool_data.get("parameters", {})

                        print(
                            f"Processing tool call: {tool_name} with parameters: {parameters}")

                        if tool_name:
                            try:
                                # Execute tool
                                print(f"Executing tool: {tool_name}")
                                tool_result = self.execute_tool(
                                    agent_name, tool_name, parameters)
                                print(
                                    f"Tool execution result status: {tool_result.get('status')}")

                                if tool_result.get("status") == "success":
                                    print(
                                        f"Tool executed successfully - yielding result")
                                    yield tool_result.get('result', '')
                                else:
                                    print(
                                        f"Tool execution failed: {tool_result.get('message')}")
                                    yield f"Error: {tool_result.get('message', 'Unknown error')}"
                            except Exception as e:
                                print(f"Tool execution exception: {str(e)}")
                                yield f"Error executing tool: {str(e)}"

                    # Handle handoff
                    elif "handoff" in data:
                        print(
                            f"Processing handoff to: {data['handoff'].get('target_agent')}")
                        # Store handoff data but don't yield anything
                        self._last_handoff = data["handoff"]
                        return

                    # If we got JSON but it's not a tool call or handoff, yield it as text
                    else:
                        print(
                            f"Received JSON but not a tool call or handoff. Keys: {list(data.keys())}")
                        yield full_response

                except json.JSONDecodeError as e:
                    # Not valid JSON, yield it as is
                    print(f"JSON parse error: {str(e)} - yielding as text")
                    yield full_response

            # If nothing has been yielded yet (e.g., failed JSON parsing), yield the full response
            if not tool_json_found:
                print(f"Non-JSON response handled normally")

        except Exception as e:
            print(f"Error in generate_response: {str(e)}")
            import traceback
            print(traceback.format_exc())
            yield f"I'm sorry, I encountered an error: {str(e)}"
