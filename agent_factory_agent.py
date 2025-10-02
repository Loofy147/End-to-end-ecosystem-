
from typing import Any, Dict, List
import os
import asyncio
from core_agents import CoreAgent
from prompt_context_engineer_agent import PromptContextEngineerAgent
from agents.agent_optimizer import AgentOptimizer

class AgentFactoryAgent(CoreAgent):
    """
    AgentFactoryAgent:
    - Dynamically generates new agent classes, files, and roles.
    - Collaborates with PromptContextEngineerAgent for context-aware code templates.
    - Uses AgentOptimizer to refine generated code.
    - Registers new agents automatically into the system.
    """

    def __init__(
        self,
        agent_id: str = "factory1",
        name: str = "AgentFactoryAgent",
        base_dir: str = "src/agents/generated"
    ):
        super().__init__(agent_id, name)
        self.base_dir = base_dir
        self.prompt_engineer = PromptContextEngineerAgent("pe_factory", "PromptEngineer")
        self.optimizer = AgentOptimizer()
        # Ensure base directory exists
        os.makedirs(self.base_dir, exist_ok=True)

    def initialize(self) -> None:
        self.state["initialized"] = True
        self.prompt_engineer.initialize()
        self.optimizer.initialize()

    def shutdown(self) -> None:
        self.prompt_engineer.shutdown()
        self.optimizer.shutdown()
        self.state.clear()

    def handle(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        input_data:
          - 'role_name': str, e.g., 'NotificationAgent'
          - 'capabilities': List[str], e.g., ['send_email', 'log_event']
          - 'author': Optional[str], e.g., 'Hicham'
        Returns:
          - 'file_path': path to generated file
          - 'suggestions': optimizer feedback
          - 'class_name': generated class name
        """
        role = input_data["role_name"]
        caps = input_data.get("capabilities", [])
        author = input_data.get("author", "AutoGen")
        # Generate context
        task = type("T", (), {
            "task_id": f"factory_{role}",
            "input": {"role_name": role, "capabilities": caps, "author": author}
        })
        prompt_meta = self.prompt_engineer.build_prompt(
            task=task,
            capability="coding",
            additional_context=f"Generate agent class {role} with methods for {', '.join(caps)}."
        )
        prompt = prompt_meta["prompt"]
        # Simulate code generation (stub)
        generated_code = self._generate_agent_code(role, caps, author)
        # Optimize code
        opt_result = self.optimizer.handle({
            "source_code": generated_code,
            "apply_fixes": True
        })
        final_code = opt_result["optimized_code"]
        # Write file
        filename = f"{role.lower()}.py"
        path = os.path.join(self.base_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(final_code)
        # Return metadata
        return {
            "class_name": role,
            "file_path": path,
            "suggestions": opt_result.get("suggestions", []),
            "prompt": prompt
        }

    def _generate_agent_code(self, role: str, caps: List[str], author: str) -> str:
        """
        Build basic agent class skeleton.
        """
        imports = "from core_agents import CoreAgent\nfrom typing import Any, Dict\n\n"
        doc = f'"""\nAuto-generated agent {role}\nAuthor: {author}\nCapabilities: {", ".join(caps)}\n"""\n\n'
        class_def = f"class {role}(CoreAgent):\n"
        init = (
            "    def __init__(self, agent_id: str, name: str):\n"
            "        super().__init__(agent_id, name)\n\n"
        )
        init += "    def initialize(self) -> None:\n        pass\n\n"
        init += "    def shutdown(self) -> None:\n        pass\n\n"
        init += "    async def handle(self, input_data: Dict[str, Any]) -> Any:\n"
        init += "        # TODO: implement task routing\n"
        init += "        return {'status': 'ok', 'handled': input_data}\n\n"
        methods = ""
        for c in caps:
            methods += f"    async def {c}(self, payload: Any) -> Any:\n"
            methods += f"        # TODO: implement {c}\n"
            methods += f"        return {{'{c}': payload}}\n\n"
        return imports + doc + class_def + init + methods
