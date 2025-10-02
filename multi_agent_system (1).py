
"""
multi_agent_system.py

Central coordinator that instantiates and organizes all major agents in the system.
"""

from integration.integration_orchestrator import IntegratedOrchestrator
from agents.ui_agents import UIAgent
from agents.data_agents import DataAgent
from agents.model_agents import ModelAgent
from agents.prompt_context_engineer_agent import PromptContextEngineerAgent
from agents.agent_optimizer import AgentOptimizer
from agents.multimodal_interaction_agent import MultiModalInteractionAgent
from agents.system_harmony_agent import SystemHarmonyAgent

def create_all_agents():
    # Initialize core orchestrator
    orchestrator = IntegratedOrchestrator()

    # Core utility agents
    optimizer = AgentOptimizer()
    prompt_engineer = PromptContextEngineerAgent("pe1", "PromptEngineer")
    ui_agent = UIAgent("ui1", "UIAgent", host="0.0.0.0", port=8001)

    # Functional agents
    data_agent = DataAgent("data1", "DataAgent")
    model_agent = ModelAgent("model1", "ModelAgent", model_path="models/model.pt")

    # Higher-level orchestrators
    multimodal_agent = MultiModalInteractionAgent("mm1", "MultiModalAgent", ui_agent, orchestrator)
    harmony_agent = SystemHarmonyAgent("harmony1", "SystemHarmonyAgent")

    # Initialize all agents
    for agent in [
        data_agent, model_agent, ui_agent,
        prompt_engineer, optimizer, multimodal_agent, harmony_agent
    ]:
        try:
            agent.initialize()
        except Exception as e:
            print(f"[WARN] Failed to initialize {agent.name}: {e}")

    return {
        "orchestrator": orchestrator,
        "ui": ui_agent,
        "data": data_agent,
        "model": model_agent,
        "prompt": prompt_engineer,
        "optimizer": optimizer,
        "multimodal": multimodal_agent,
        "harmony": harmony_agent,
    }

if __name__ == "__main__":
    agents = create_all_agents()
    print("All agents initialized.")
    print("Ready for integration or API exposure.")
