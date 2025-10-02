import os
import sys
import logging
from tuber_orchestrator import TuberOrchestratorAI
from config import Config

# Configure logging
logging.basicConfig(level=Config.LOG_LEVEL, format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    # Ensure .env file exists and API keys are set
    if not os.path.exists(".env"):
        logging.warning("No .env file found. Please create one with your LLM_PROVIDER and API_KEY.")
        logging.warning("Example .env content:\nOPENAI_API_KEY=\"your_key_here\"\nLLM_PROVIDER=\"openai\"")
        sys.exit(1)

    try:
        orchestrator = TuberOrchestratorAI()
    except ValueError as e:
        logging.error(f"Initialization error: {e}. Please check your .env file and configuration.")
        sys.exit(1)

    # Seed the root vision (the core philosophy of your system)
    root_vision = "دردتنا مبنية على Umbrella Architecture كـ Collaborative Ecosystem، تعمل كـ Meta-System ينسج Social Fabric مرنًا بين المشاركين؛ كل ذلك موجهًا من خلال Holistic Framework ومنسقًا عبر Holarchy of Communities of Practice، مما يعزز Collective Intelligence و Peer-to-Peer collaboration."
    orchestrator.seed_root_vision(root_vision)

    print("\n--- TuberOrchestratorAI Initialized ---")
    print("Type 'help' for available commands or 'exit' to quit.")
    print("---------------------------------------")

    while True:
        try:
            user_input = input("\nDeveloper> ").strip()
            if user_input.lower() == 'exit':
                print("Exiting TuberOrchestratorAI. Goodbye!")
                break
            elif user_input.lower() == 'help':
                print("\nAvailable Commands:")
                print("  exit                                  - Exit the application.")
                print("  help                                  - Show this help message.")
                print("  status                                - Get current system status.")
                print("  chat <message>                        - Send a general message to the AI.")
                print("  suggest_code <problem_desc>           - Ask AI to suggest code for a problem.")
                print("  propose_experiment <goal>             - Ask AI to propose an automated experiment.")
                print("  run_experiment <exp_id>               - Run a proposed experiment.")
                print("  validate_code <suggestion_id>         - Validate a cached code suggestion.")
                print("  list_experiments                      - List all proposed and completed experiments.")
                print("  health                                - Get a system health report.")
                print("\nExample Usage:")
                print("  Developer> chat How can I improve tuber pruning?")
                print("  Developer> suggest_code \"Implement a new reward function\"\n")
                continue
            elif user_input.lower() == 'status':
                status = orchestrator.get_system_status()
                print("\n--- System Status ---")
                for key, value in status.items():
                    print(f"  {key}: {value}")
                print("---------------------")
                continue
            elif user_input.lower() == 'health':
                health_report = orchestrator.developer_converse(message="", action_type="analyze_system_health")
                print(health_report)
                continue
            elif user_input.lower() == 'list_experiments':
                exp_list = orchestrator.developer_converse(message="", action_type="list_experiments")
                print(exp_list)
                continue

            # Parse commands with arguments
            parts = user_input.split(" ", 1)
            command = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if command == 'chat':
                response = orchestrator.developer_converse(message=arg)
                print(f"\nAI Response:\n{response}")
            elif command == 'suggest_code':
                response = orchestrator.developer_converse(message="", action_type="suggest_code_change", problem_description=arg)
                print(f"\nAI Response:\n{response}")
            elif command == 'propose_experiment':
                response = orchestrator.developer_converse(message="", action_type="propose_automated_experiment", goal=arg)
                print(f"\nAI Response:\n{response}")
            elif command == 'run_experiment':
                response = orchestrator.developer_converse(message="", action_type="run_automated_experiment", experiment_id=arg)
                print(f"\nAI Response:\n{response}")
            elif command == 'validate_code':
                response = orchestrator.developer_converse(message="", action_type="validate_code_suggestion", suggestion_id=arg)
                print(f"\nAI Response:\n{response}")
            else:
                print("Unknown command. Type 'help' for available commands.")

        except Exception as e:
            logging.error(f"An error occurred: {e}")
            print(f"An error occurred: {e}. Check logs for details.")

if __name__ == "__main__":
    main()


