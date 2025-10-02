# Project: Agents Refactoring & Execution Orchestration

# ✅ Summary (as of July 2025)
# نظام متعدد الوكلاء modular architecture باستخدام PyTorch، FastAPI، ngrok، والموجه للتنفيذ والتنسيق مع بيانات حقيقية من مولدات أو مجمعات واقعية. الهدف: تشغيل النظام بدون تدريب فعلي، مع تغذية فعلية للبيانات.

# ✅ Directory Structure
# .
# ├── main.py                          # Main entrypoint (FastAPI)
# ├── config.py                        # Central settings & modes
# ├── database_helpers.py             # DB abstraction layer
# ├── agents/                         # Agent logic
# │   ├── data_agents.py              # Handles real/generator input
# │   ├── model_agents.py
# │   ├── core_agents.py
# ├── orchestrator/
# │   ├── base_agent.py
# │   └── orchestrator.py
# ├── api/
# │   ├── ui_agents.py
# │   └── routes.py
# ├── knowledge/
# │   └── knowledge_graph.py
# ├── data/                           # Output real/generated data
# ├── real_data_collector.py          # Real-world data acquisition
# ├── data_generator.py               # Synthetic data based on rules
# ├── tests/
# │   └── test_agents.py
# └── requirements.txt


# ✅ Execution Modes in config.py
# config.py
SIMULATION_MODE = False
USE_REAL_DATA = True
DATA_SOURCE = "generator"  # "generator" or "collector"
SKIP_TRAINING = True


# ✅ DataPreprocessingAgent Logic
```python
from config import USE_REAL_DATA, DATA_SOURCE

if USE_REAL_DATA:
    if DATA_SOURCE == "generator":
        from data_generator import DataGenerator
        gen = DataGenerator(output_dir="data")
        data_file = gen.generate_jsonl_training_data(count=500)
        self.data = load_jsonl(data_file)
    elif DATA_SOURCE == "collector":
        from real_data_collector import collect_data
        self.data = collect_data()
else:
    # fallback to simulation / preloaded sample
    self.data = []
```


# ✅ TrainingAgent Logic
```python
from config import SKIP_TRAINING
if SKIP_TRAINING:
    print("🧪 Skipping training – using real data only.")
else:
    self.train_model(self.data)
```


# ✅ Orchestrator Behavior
- Reads config, detects `USE_REAL_DATA`
- Initializes pipeline: [DataPreprocessingAgent → ModelConstructionAgent → TrainingAgent → EvaluationAgent]
- Runs agents in sequence


# ✅ Optional Enhancements
- Log all agent transitions with timestamps
- Add `/run-all` endpoint in FastAPI to trigger full orchestration
- Visualize data loading status in frontend


# ✅ Next Implementation Step
- ✅ Create `config.py` with all flags
- ✅ Modify `DataPreprocessingAgent` to use `data_generator` and `real_data_collector`
- ✅ Update `TrainingAgent` to skip training when needed
- ✅ Add testing log to `tests/test_agents.py`


# ✅ Ready for execution 🚀
