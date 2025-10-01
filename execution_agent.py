import io
import sys
from agent_manager import Agent
from typing import Dict, Optional, Any

class ExecutionAgent(Agent):
    """
    A specialized agent for executing arbitrary Python code.

    WARNING: This agent is NOT secure and should not be used in a
    production environment. It uses `exec` to run code, which can be
    dangerous. A real implementation would require a proper sandbox.
    """
    def __init__(self):
        super().__init__("execution_agent")
        print("Initializing ExecutionAgent...")

    async def execute(self, query: str, metadata: Optional[Dict[str, Any]] = None) -> Any:
        """
        Executes the given query as a Python script.

        The agent captures and returns the standard output of the script.
        """
        print(f"ExecutionAgent executing code: \n---\n{query}\n---")

        # Create a string buffer to capture stdout
        old_stdout = sys.stdout
        redirected_output = io.StringIO()
        sys.stdout = redirected_output

        try:
            # Execute the code
            exec(query)
            # Restore stdout
            sys.stdout = old_stdout
            # Get the output
            output = redirected_output.getvalue()
            return f"Execution successful. Output:\n{output}"
        except Exception as e:
            # Restore stdout in case of an error
            sys.stdout = old_stdout
            error_message = f"An error occurred during execution: {e}"
            print(error_message)
            return error_message

# Example usage
async def main():
    agent = ExecutionAgent()

    # Example 1: Simple print
    code_to_run = "print('Hello from the execution agent!')"
    result = await agent.execute(code_to_run)
    print(result)

    # Example 2: A loop
    code_to_run_2 = """
for i in range(5):
    print(f"Line {i+1}")
"""
    result_2 = await agent.execute(code_to_run_2)
    print(result_2)

    # Example 3: An error
    code_with_error = "print(undefined_variable)"
    result_3 = await agent.execute(code_with_error)
    print(result_3)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())