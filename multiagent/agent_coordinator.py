import json
from typing import List, Dict, Any
import os
import requests

class AgentCoordinator:
    def __init__(self, pipeline_instance: Any):
        """
        Initializes the AgentCoordinator.

        Args:
            pipeline_instance: The instance of the main pipeline, providing access to valves and other context.
        """
        self.pipeline = pipeline_instance
        self.valves = pipeline_instance.valves
        self.agent_configs = {}  # Initialize as empty dict

        print(f"AgentCoordinator initialized with pipeline ID: {self.pipeline.id if hasattr(self.pipeline, 'id') else 'Unknown'}")

        agent_configs_json_str = getattr(self.valves, "AGENT_CONFIGS_JSON", "{}")
        try:
            self.agent_configs = json.loads(agent_configs_json_str)
            print(f"AgentCoordinator: Successfully loaded {len(self.agent_configs)} agent configurations.")
        except json.JSONDecodeError as e:
            print(f"AgentCoordinator: Error decoding AGENT_CONFIGS_JSON: {e}")
            print(f"AgentCoordinator: AGENT_CONFIGS_JSON string was: {agent_configs_json_str}")
            self.agent_configs = {}
        except AttributeError:
            print("AgentCoordinator: Critical Error - AGENT_CONFIGS_JSON valve attribute not found in pipeline_instance.valves. Agent configurations will be empty.")
            self.agent_configs = {}


    async def _call_llm(self, messages: List[Dict], model_id: str) -> str:
        """
        Calls an OpenAI-compatible LLM.
        """
        print(f"AgentCoordinator: Calling LLM (model: {model_id}) with messages: {messages}")

        api_base_url = getattr(self.valves, "OPENAI_API_BASE_URL", "https://api.openai.com/v1")
        api_key = getattr(self.valves, "OPENAI_API_KEY", None)

        if not api_key:
            print("AgentCoordinator: Error - OPENAI_API_KEY not found in valves.")
            return "Error: OPENAI_API_KEY not configured."

        request_url = f"{api_base_url.rstrip('/')}/chat/completions"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model_id,
            "messages": messages,
            "stream": False,
        }

        try:
            response = requests.post(request_url, json=payload, headers=headers)
            response.raise_for_status()

            response_json = response.json()

            if response_json.get("choices") and response_json["choices"][0].get("message"):
                content = response_json["choices"][0]["message"].get("content", "")
                print(f"AgentCoordinator: LLM response content: {content}")
                return content.strip()
            else:
                error_msg = f"Error: Unexpected LLM response structure. Full response: {json.dumps(response_json)}"
                print(f"AgentCoordinator: {error_msg}")
                return error_msg

        except requests.exceptions.HTTPError as http_err:
            error_detail = http_err.response.text if http_err.response else "No response body"
            error_msg = f"Error: HTTP error {http_err.response.status_code} - {error_detail}"
            print(f"AgentCoordinator: {error_msg}")
            return error_msg
        except Exception as e:
            error_msg = f"Error: An unexpected error occurred during LLM call: {str(e)}"
            print(f"AgentCoordinator: {error_msg}")
            return error_msg

    async def manage_multi_agent_flow(self, user_query: str, chat_history: List[Dict], agent_ids: List[str], base_model_id: str) -> List[Dict[str, Any]]:
        """
        Manages the conversation flow for a list of specified agents.
        Each agent processes the query and history independently.
        """
        print(f"AgentCoordinator: Starting dynamic flow for query '{user_query}' with agent IDs: {agent_ids}.")

        all_agent_results = []

        if not self.agent_configs:
            print("AgentCoordinator: Error - No agent configurations loaded. Cannot process agents.")
            for agent_id in agent_ids:
                 all_agent_results.append({
                    "agent_id": agent_id,
                    "response": "Error: Agent configurations not loaded in coordinator. Check pipeline valve setup."
                })
            return all_agent_results

        for agent_id in agent_ids:
            agent_config = self.agent_configs.get(agent_id)

            if not agent_config:
                print(f"AgentCoordinator: Warning - No configuration found for agent_id '{agent_id}'. Skipping.")
                all_agent_results.append({
                    "agent_id": agent_id,
                    "response": f"Error: No configuration found for agent_id '{agent_id}'."
                })
                continue

            system_prompt = agent_config.get("system_prompt", "You are a helpful assistant.")

            agent_model_id = agent_config.get("model_id")
            if not agent_model_id:
                agent_model_id = base_model_id
            if not agent_model_id:
                 agent_model_id = getattr(self.valves, "DEFAULT_AGENT_MODEL_ID", "gpt-3.5-turbo")

            print(f"AgentCoordinator: Processing Agent ID: '{agent_id}' with Model ID: '{agent_model_id}'")

            current_agent_messages = [{"role": "system", "content": system_prompt}]

            history_to_include = chat_history[-4:]
            for msg in history_to_include:
                current_agent_messages.append(msg)

            current_agent_messages.append({"role": "user", "content": user_query})

            agent_response_content = await self._call_llm(messages=current_agent_messages, model_id=agent_model_id)

            print(f"AgentCoordinator: Agent '{agent_id}' response: {agent_response_content}")

            all_agent_results.append({
                "agent_id": agent_id,
                "response": agent_response_content
            })

        print(f"AgentCoordinator: End of dynamic flow. All agent results: {all_agent_results}")
        return all_agent_results
