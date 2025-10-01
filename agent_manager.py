from typing import Dict, Optional

class Agent:
    """
    Abstract base class for all specialized agents.
    Defines the interface that the orchestrator expects.
    """
    def __init__(self, name: str):
        self.name = name

    async def execute(self, query: str, metadata: Optional[Dict] = None) -> any:
        """
        Executes a task based on the given query and returns a result.
        This method must be implemented by all subclasses.
        """
        raise NotImplementedError("Each agent must implement the 'execute' method.")

class AgentManager:
    """
    Manages the lifecycle and selection of specialized agents.

    In this initial version, it acts as a simple registry. In a more advanced
    implementation, it could handle dynamic loading, performance monitoring,
    and adaptive selection of agents.
    """
    def __init__(self):
        print("Initializing AgentManager...")
        self._agents: Dict[str, Agent] = {}

    def register_agent(self, agent: Agent):
        """
        Registers a new agent with the manager.
        """
        if agent.name in self._agents:
            print(f"Warning: Agent '{agent.name}' is already registered. Overwriting.")
        print(f"Registering agent: {agent.name}")
        self._agents[agent.name] = agent

    def get_agent(self, name: str) -> Optional[Agent]:
        """
        Retrieves a registered agent by its name.
        """
        print(f"Attempting to retrieve agent: {name}")
        return self._agents.get(name)

    def list_agents(self) -> list[str]:
        """
        Returns a list of all registered agent names.
        """
        return list(self._agents.keys())

# Example of how it might be run (for illustration purposes)
async def main():
    # Define a couple of mock agents for demonstration
    class MockMathAgent(Agent):
        def __init__(self):
            super().__init__("math_agent")
        async def execute(self, query, metadata=None):
            return f"Mock response from {self.name} for query: '{query}'"

    class MockRLAgent(Agent):
        def __init__(self):
            super().__init__("rl_agent")
        async def execute(self, query, metadata=None):
            return f"Mock response from {self.name} for query: '{query}'"

    # Initialize the manager and register agents
    agent_manager = AgentManager()
    agent_manager.register_agent(MockMathAgent())
    agent_manager.register_agent(MockRLAgent())

    # List registered agents
    print("\nRegistered agents:", agent_manager.list_agents())

    # Retrieve and use an agent
    math_agent = agent_manager.get_agent("math_agent")
    if math_agent:
        result = await math_agent.execute("calculate 2+2")
        print("\nResult from math_agent:", result)

    # Try to get a non-existent agent
    unknown_agent = agent_manager.get_agent("unknown_agent")
    print("\nResult for unknown_agent:", unknown_agent)

if __name__ == "__main__":
    # In a real application, this file would be imported.
    # We run the async main function for demonstration.
    import asyncio
    print("agent_manager.py created successfully and running example.")
    asyncio.run(main())