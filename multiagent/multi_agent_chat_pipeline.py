import os
import json
from typing import List, Dict, Any, Union, Generator, Iterator
from pydantic import BaseModel

try:
    from .agent_coordinator import AgentCoordinator
except ImportError:
    print("Attempting fallback import for AgentCoordinator due to potential ImportError")
    from agent_coordinator import AgentCoordinator


class Pipeline:
    class Valves(BaseModel):
        OPENAI_API_BASE_URL: str = os.getenv("OPENAI_API_BASE_URL", "https://api.openai.com/v1")
        OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

        DEFAULT_AGENT_MODEL_ID: str = os.getenv("DEFAULT_AGENT_MODEL_ID", "gpt-3.5-turbo")

        AGENT_CONFIGS_JSON: str = os.getenv("AGENT_CONFIGS_JSON", '''
        {
            "analyst_agent_v1": {
                "system_prompt": "You are Analyst Agent. Your role is to break down the user's query into key components and identify the core intent.",
                "model_id": "gpt-3.5-turbo"
            },
            "research_master_alpha": {
                "system_prompt": "You are Research Master Alpha. You specialize in gathering and synthesizing information based on an analysis provided to you. Focus on accuracy and comprehensive coverage.",
                "model_id": "gpt-4"
            },
            "default_responder": {
                "system_prompt": "You are a helpful general-purpose assistant.",
                "model_id": "gpt-3.5-turbo"
            }
        }
        ''')

    def __init__(self):
        self.id = "multi-agent-chat"
        self.name = "Multi-Agent Chat"
        self.valves = self.Valves()

        try:
            self.coordinator = AgentCoordinator(self)
        except Exception as e:
            print(f"Error instantiating AgentCoordinator: {e}")
            self.coordinator = None

        print(f"Initialized {self.name} pipeline with id {self.id}")
        if self.coordinator is None:
            print(f"CRITICAL ERROR: {self.name} pipeline's AgentCoordinator failed to initialize.")

    async def on_startup(self):
        print(f"on_startup:{self.name}")
        if self.coordinator is None:
             print(f"CRITICAL ERROR during on_startup: {self.name} pipeline's AgentCoordinator is not initialized.")
        pass

    async def on_shutdown(self):
        print(f"on_shutdown:{self.name}")
        pass

    async def on_valves_updated(self):
        print(f"{self.name} valves updated.")
        if self.coordinator:
            new_agent_configs_json_str = getattr(self.valves, "AGENT_CONFIGS_JSON", "{}")
            try:
                self.coordinator.agent_configs = json.loads(new_agent_configs_json_str)
                print(f"AgentCoordinator: Successfully reloaded {len(self.coordinator.agent_configs)} agent configurations after valve update.")
            except json.JSONDecodeError as e:
                print(f"AgentCoordinator: Error decoding AGENT_CONFIGS_JSON during valve update: {e}")
                self.coordinator.agent_configs = {}
            except AttributeError:
                 print("AgentCoordinator: AGENT_CONFIGS_JSON valve not found during update.")
                 self.coordinator.agent_configs = {}
        else:
            print(f"CRITICAL ERROR during on_valves_updated: {self.name} pipeline's AgentCoordinator is not initialized.")
        pass

    async def pipe(
        self, user_message: str, model_id: str, messages: List[Dict], body: Dict
    ) -> Union[List[Dict[str, Any]], str]:
        """
        Main pipeline entry point.
        """
        if self.coordinator is None:
            print(f"CRITICAL ERROR in pipe: {self.name} pipeline's AgentCoordinator is not initialized.")
            return [{"agent_id": "system_error", "response": "Coordinator not initialized. Cannot process request."}]

        print(f"{self.name} pipe received message: '{user_message}' for model_id from request: '{model_id}'")

        base_model_id = model_id if model_id else self.valves.DEFAULT_AGENT_MODEL_ID
        print(f"Base model_id for AgentCoordinator: {base_model_id}")

        default_agent_ids_to_run = ["analyst_agent_v1", "default_responder"]
        agent_ids_to_run = body.get("agent_ids", default_agent_ids_to_run)

        if not isinstance(agent_ids_to_run, list) or not all(isinstance(agent_id, str) for agent_id in agent_ids_to_run):
            print(f"Warning: 'agent_ids' in request body is not a list of strings or not provided. Using default: {default_agent_ids_to_run}")
            agent_ids_to_run = default_agent_ids_to_run

        if not agent_ids_to_run:
            print("Warning: No agent_ids specified to run. Returning an informative message.")
            return [{"agent_id": "system_message", "response": "No agents were specified to process this request."}]

        print(f"Agent IDs to run: {agent_ids_to_run}")

        all_agent_responses = await self.coordinator.manage_multi_agent_flow(
            user_query=user_message,
            chat_history=messages,
            agent_ids=agent_ids_to_run,
            base_model_id=base_model_id
        )

        return all_agent_responses
