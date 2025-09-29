تمام — سأطبع هنا كل الملفات الأساسية كاملةً (قابلة للصق مباشرة في مشروعك) ونصوص الاختبار، وملفات CI و requirements-dev. لا أستطيع تشغيل البنية لديك من هنا (لا أملك وصولاً لبيئتك أو موارد مثل Redis)، لذلك أقدّم أيضاً أوامر واضحة لتثبيت الاعتماديات وتشغيل الاختبارات محليًا أو في CI. انسخ الملفات إلى مساراتها في src/hpo/... و tests/... ثم اتبع خطوات التشغيل في النهاية.

سأبدأ بالأهم ثم الملفات المساعدة والاختبارات وملف CI + requirements.


---

1) src/hpo/core.py

(ConfigurationSpace, ExperimentDatabase, EnhancedBayesianOptimizer, BOHBOptimizer — محسّن ومتكامل)

# src/hpo/core.py
"""
HPO core utilities: ConfigurationSpace, ExperimentDatabase, EnhancedBayesianOptimizer, BOHBOptimizer

This file is intended to be the core foundation. It provides:
- deterministic vectorization of configs for surrogates
- SQLite-backed experiment DB with safe pragmas
- GP-based Bayesian optimizer (EI/UCB/PI) with correct math
- BOHB scheduler skeleton with brackets, promotions, and task_id tracking
"""

from __future__ import annotations

import json
import math
import time
import uuid
import sqlite3
import threading
import hashlib
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Optional imports
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

try:
    from scipy.stats import norm as _scipy_norm
    def _norm_pdf(x): return _scipy_norm.pdf(x)
    def _norm_cdf(x): return _scipy_norm.cdf(x)
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False
    from math import erf
    def _norm_pdf(x): return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)
    def _norm_cdf(x): return 0.5 * (1.0 + erf(x / math.sqrt(2.0)))

logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)


# -----------------------
# 1) HyperParameter & ConfigurationSpace
# -----------------------
@dataclass
class HyperParameter:
    name: str
    param_type: str  # 'int', 'float', 'log_float', 'categorical', 'bool'
    bounds: Optional[Tuple[Union[int, float], Union[int, float]]] = None
    choices: Optional[List[Any]] = None
    default: Optional[Any] = None
    log_scale: bool = False
    conditional_on: Optional[Dict[str, Any]] = None

    def validate_value(self, value: Any) -> bool:
        try:
            if self.param_type == "int":
                if not isinstance(value, (int, np.integer)): return False
                return self.bounds[0] <= value <= self.bounds[1]
            if self.param_type in ("float", "log_float"):
                if not isinstance(value, (int, float, np.number)): return False
                return self.bounds[0] <= float(value) <= self.bounds[1]
            if self.param_type == "categorical":
                return value in (self.choices or [])
            if self.param_type == "bool":
                return isinstance(value, bool)
            return False
        except Exception:
            return False

    def sample_value(self, rng: Optional[np.random.RandomState] = None) -> Any:
        if rng is None:
            rng = np.random.RandomState()
        if self.param_type == "int":
            low, high = int(self.bounds[0]), int(self.bounds[1])
            return int(rng.randint(low, high + 1))
        if self.param_type == "float":
            low, high = float(self.bounds[0]), float(self.bounds[1])
            return float(rng.uniform(low, high))
        if self.param_type == "log_float":
            low, high = float(self.bounds[0]), float(self.bounds[1])
            return float(np.exp(rng.uniform(np.log(low), np.log(high))))
        if self.param_type == "categorical":
            return rng.choice(self.choices)
        if self.param_type == "bool":
            return bool(rng.choice([False, True]))
        raise ValueError(f"Unknown parameter type: {self.param_type}")


class ConfigurationSpace:
    """
    Holds HyperParameter objects and provides:
    - sample_configuration(rng)
    - validate_configuration(cfg)
    - to_array(configs) -> np.ndarray for surrogate inputs
    """

    def __init__(self, parameters: List[HyperParameter]):
        # preserve insertion order
        self.parameters: Dict[str, HyperParameter] = {p.name: p for p in parameters}
        self.conditional_graph = self._build_conditional_graph()
        # derived info for vectorization
        self._vector_length = None
        self._compute_vector_length()

    def _build_conditional_graph(self) -> Dict[str, List[str]]:
        graph: Dict[str, List[str]] = {}
        for name, param in self.parameters.items():
            if param.conditional_on:
                for parent in param.conditional_on.keys():
                    graph.setdefault(parent, []).append(name)
        return graph

    def sample_configuration(self, rng: Optional[np.random.RandomState] = None) -> Dict[str, Any]:
        if rng is None:
            rng = np.random.RandomState()
        config: Dict[str, Any] = {}
        remaining = set(self.parameters.keys())
        progressed = True
        while remaining and progressed:
            progressed = False
            for name in list(remaining):
                p = self.parameters[name]
                if not p.conditional_on:
                    config[name] = p.sample_value(rng)
                    remaining.remove(name)
                    progressed = True
                else:
                    parents_satisfied = all(parent in config for parent in p.conditional_on.keys())
                    if parents_satisfied:
                        ok = True
                        for parent, required in (p.conditional_on or {}).items():
                            if config.get(parent) != required:
                                ok = False
                                break
                        if ok:
                            config[name] = p.sample_value(rng)
                        remaining.remove(name)
                        progressed = True
        return config

    def validate_configuration(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        errors: List[str] = []
        for name, param in self.parameters.items():
            if name in config:
                if not param.validate_value(config[name]):
                    errors.append(f"Invalid value for {name}: {config[name]}")
                if param.conditional_on:
                    for parent, required in param.conditional_on.items():
                        if parent not in config:
                            errors.append(f"{name} requires parent {parent} to be present")
                        elif config[parent] != required:
                            errors.append(f"{name} requires {parent} == {required}")
        return (len(errors) == 0, errors)

    def _compute_vector_length(self):
        length = 0
        for name, param in self.parameters.items():
            if param.param_type in ("int", "float", "log_float", "bool"):
                length += 1
            elif param.param_type == "categorical":
                length += len(param.choices or [])
        self._vector_length = length

    def to_array(self, configs: List[Dict[str, Any]]) -> np.ndarray:
        if self._vector_length is None:
            self._compute_vector_length()
        result = np.zeros((len(configs), self._vector_length), dtype=float)
        for i, cfg in enumerate(configs):
            idx = 0
            for name, param in self.parameters.items():
                if param.param_type in ("int", "float", "log_float", "bool"):
                    if name in cfg:
                        v = cfg[name]
                        if v is None:
                            result[i, idx] = 0.0
                        else:
                            if param.param_type == "bool":
                                result[i, idx] = 1.0 if bool(v) else 0.0
                            else:
                                val = float(v)
                                if param.log_scale or param.param_type == "log_float":
                                    val = max(val, 1e-12)
                                    result[i, idx] = math.log(val)
                                else:
                                    result[i, idx] = val
                    else:
                        result[i, idx] = 0.0
                    idx += 1
                elif param.param_type == "categorical":
                    choices = param.choices or []
                    one_hot = [0.0] * len(choices)
                    if name in cfg:
                        val = cfg[name]
                        for k, ch in enumerate(choices):
                            if ch == val:
                                one_hot[k] = 1.0
                                break
                    for v in one_hot:
                        result[i, idx] = v
                        idx += 1
                else:
                    result[i, idx] = 0.0
                    idx += 1
        return result


# -----------------------
# 2) ExperimentDatabase (SQLite)
# -----------------------
class ExperimentDatabase:
    """
    Lightweight SQLite-backed experiment DB with simple concurrency settings.
    Stores studies and trials. Use one connection per operation.
    """

    def __init__(self, db_path: str = "hpo_experiments.db"):
        self.db_path = db_path
        self._init_lock = threading.Lock()
        self._ensure_schema()

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _ensure_schema(self):
        with self._init_lock:
            with self._get_conn() as conn:
                cur = conn.cursor()
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS studies (
                        study_id TEXT PRIMARY KEY,
                        study_name TEXT UNIQUE,
                        direction TEXT,
                        objective_name TEXT,
                        config_space TEXT,
                        metadata TEXT,
                        created_at INTEGER,
                        completed_at INTEGER,
                        status TEXT
                    )
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS trials (
                        trial_id TEXT PRIMARY KEY,
                        study_id TEXT,
                        trial_number INTEGER,
                        parameters TEXT,
                        metrics TEXT,
                        status TEXT,
                        started_at INTEGER,
                        completed_at INTEGER,
                        duration_seconds REAL,
                        error_message TEXT,
                        FOREIGN KEY(study_id) REFERENCES studies(study_id)
                    )
                """)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_trials_study ON trials(study_id)")
                conn.commit()

    def create_study(self, study_name: str, direction: str, objective_name: str, config_space: ConfigurationSpace, metadata: Dict = None) -> str:
        meta_json = json.dumps(metadata or {})
        cfg_json = json.dumps([vars(p) for p in config_space.parameters.values()])
        study_id = f"study_{hashlib.md5((study_name+str(time.time())).encode()).hexdigest()[:10]}"
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute("INSERT INTO studies (study_id,study_name,direction,objective_name,config_space,metadata,created_at,status) VALUES (?,?,?,?,?,?,?,?)",
                        (study_id, study_name, direction, objective_name, cfg_json, meta_json, int(time.time()), "running"))
            conn.commit()
        return study_id

    def save_trial(self, study_id: str, trial_number: int, parameters: Dict[str, Any], metrics: Optional[Dict[str, Any]] = None, status: str = "running", error_message: Optional[str] = None, duration_seconds: Optional[float] = None) -> str:
        trial_id = f"trial_{study_id}_{trial_number:06d}"
        with self._get_conn() as conn:
            cur = conn.cursor()
            if status == "running":
                cur.execute("INSERT OR REPLACE INTO trials (trial_id,study_id,trial_number,parameters,status,started_at) VALUES (?,?,?,?,?,?)",
                            (trial_id, study_id, trial_number, json.dumps(parameters), status, int(time.time())))
            else:
                cur.execute("UPDATE trials SET metrics=?, status=?, completed_at=?, duration_seconds=?, error_message=? WHERE trial_id=?",
                            (json.dumps(metrics or {}), status, int(time.time()), duration_seconds, error_message, trial_id))
            conn.commit()
        return trial_id

    def get_study_trials(self, study_id: str) -> List[Dict[str, Any]]:
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT trial_number, parameters, metrics, status, started_at, completed_at, duration_seconds FROM trials WHERE study_id=? ORDER BY trial_number", (study_id,))
            rows = cur.fetchall()
        out = []
        for r in rows:
            trial_number, params_j, metrics_j, status, started_at, completed_at, duration_seconds = r
            out.append({
                "trial_number": trial_number,
                "parameters": json.loads(params_j),
                "metrics": json.loads(metrics_j) if metrics_j else {},
                "status": status,
                "started_at": started_at,
                "completed_at": completed_at,
                "duration_seconds": duration_seconds
            })
        return out

    def get_top_configs(self, study_id: str, metric_name: str, top_k: int = 3, minimize: bool = True) -> List[Dict[str, Any]]:
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT parameters, metrics FROM trials WHERE study_id=? AND metrics IS NOT NULL", (study_id,))
            rows = cur.fetchall()
        entries = []
        for params_j, metrics_j in rows:
            m = json.loads(metrics_j)
            if metric_name in m:
                entries.append((json.loads(params_j), m[metric_name]))
        if not entries:
            return []
        entries.sort(key=lambda x: x[1], reverse=not minimize)
        return [e[0] for e in entries[:top_k]]

    def find_similar_studies(self, meta_key: str, meta_value: Any) -> List[str]:
        res = []
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT study_id, metadata FROM studies")
            rows = cur.fetchall()
        for study_id, meta_j in rows:
            try:
                md = json.loads(meta_j or "{}")
                if isinstance(md, dict) and md.get(meta_key) == meta_value:
                    res.append(study_id)
            except Exception:
                continue
        return res


# -----------------------
# 3) Enhanced Bayesian optimizer (GP + EI/UCB/PI)
# -----------------------
class EnhancedBayesianOptimizer:
    """
    Gaussian Process surrogate with multiple acquisition functions.
    """

    def __init__(self, config_space: ConfigurationSpace, acquisition: str = "ei", xi: float = 0.01, kappa: float = 2.576, n_initial: int = 8, minimize: bool = True):
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn is required for EnhancedBayesianOptimizer")
        self.cs = config_space
        self.acquisition = acquisition.lower()
        self.xi = xi
        self.kappa = kappa
        self.n_initial = int(n_initial)
        self.minimize = bool(minimize)

        self.X: List[np.ndarray] = []
        self.y: List[float] = []
        kernel = Matern(length_scale=1.0, nu=2.5)
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, n_restarts_optimizer=3)

    def tell(self, config: Dict[str, Any], value: float):
        vec = self.cs.to_array([config])[0]
        self.X.append(vec)
        self.y.append(float(value))

    def suggest(self, n_candidates: int = 200) -> Dict[str, Any]:
        if len(self.X) < self.n_initial:
            return self.cs.sample_configuration()
        X_arr = np.vstack(self.X)
        y_arr = np.array(self.y)
        self.gp.fit(X_arr, y_arr)

        candidates = [self.cs.sample_configuration() for _ in range(n_candidates)]
        Xc = self.cs.to_array(candidates)
        mu, sigma = self.gp.predict(Xc, return_std=True)
        sigma = np.maximum(sigma, 1e-12)

        if self.minimize:
            f_best = np.min(y_arr)
            if self.acquisition == "ei":
                gamma = (f_best - mu - self.xi) / sigma
                acq = (f_best - mu - self.xi) * _norm_cdf(gamma) + sigma * _norm_pdf(gamma)
            elif self.acquisition == "ucb":
                acq = -(mu - self.kappa * sigma)
            elif self.acquisition == "pi":
                gamma = (f_best - mu - self.xi) / sigma
                acq = _norm_cdf(gamma)
            else:
                raise ValueError("Unknown acquisition")
        else:
            f_best = np.max(y_arr)
            if self.acquisition == "ei":
                gamma = (mu - f_best - self.xi) / sigma
                acq = (mu - f_best - self.xi) * _norm_cdf(gamma) + sigma * _norm_pdf(gamma)
            elif self.acquisition == "ucb":
                acq = mu + self.kappa * sigma
            elif self.acquisition == "pi":
                gamma = (mu - f_best - self.xi) / sigma
                acq = _norm_cdf(gamma)
            else:
                raise ValueError("Unknown acquisition")

        best_idx = int(np.argmax(acq))
        return candidates[best_idx]


# -----------------------
# 4) BOHB (BO + HyperBand skeleton)
# -----------------------
class BOHBOptimizer:
    """
    BOHB-like scheduler that generates brackets (HyperBand) and performs successive halving.
    This is an orchestration skeleton: it samples configs from config_space, schedules them
    at budgets, and promotes top performers.
    """

    def __init__(self, config_space: ConfigurationSpace, min_budget: float = 1.0, max_budget: float = 27.0, eta: int = 3):
        self.cs = config_space
        self.min_budget = float(min_budget)
        self.max_budget = float(max_budget)
        self.eta = int(eta)
        self.s_max = int(math.floor(math.log(self.max_budget / self.min_budget) / math.log(self.eta)))
        self.B = (self.s_max + 1) * self.max_budget

        self.brackets = {}
        self.task_queue: List[Tuple[str, Dict[str, Any], float]] = []
        self.task_map = {}
        self.results_lock = threading.Lock()
        self.task_counter = 0

    def _next_task_id(self) -> str:
        self.task_counter += 1
        return f"t_{self.task_counter}_{uuid.uuid4().hex[:6]}"

    def _generate_bracket(self, s: int) -> str:
        R = self.max_budget
        n = int(math.ceil(self.B / R * (self.eta ** s) / (s + 1)))
        r = float(R * (self.eta ** (-s)))
        bracket_id = f"br_{int(time.time()*1000)}_{s}_{uuid.uuid4().hex[:6]}"
        logger.info("Creating bracket %s: s=%d n=%d r=%.4f", bracket_id, s, n, r)
        rungs = []
        cur_n = n
        cur_r = r
        for i in range(s + 1):
            rungs.append({"level": i, "n": cur_n, "r": cur_r, "results": []})
            cur_n = int(math.floor(cur_n / self.eta))
            cur_r = cur_r * self.eta
        self.brackets[bracket_id] = {"s": s, "rungs": rungs, "completed": False}
        for i in range(n):
            cfg = self.cs.sample_configuration()
            task_id = self._next_task_id()
            cfg_internal = dict(cfg)
            cfg_internal["__hbo_task_id"] = task_id
            self.task_queue.append((task_id, cfg_internal, r))
            self.task_map[task_id] = (bracket_id, 0, cfg_internal)
        return bracket_id

    def suggest(self) -> Tuple[Dict[str, Any], float]:
        if not self.task_queue:
            next_s = (len(self.brackets)) % (self.s_max + 1)
            self._generate_bracket(next_s)
        task_id, cfg, budget = self.task_queue.pop(0)
        return cfg, budget

    def tell(self, config: Dict[str, Any], budget: float, objective_value: float):
        task_id = config.get("__hbo_task_id")
        if not task_id:
            logger.warning("tell() called with config missing __hbo_task_id - ignoring")
            return
        with self.results_lock:
            mapping = self.task_map.get(task_id)
            if not mapping:
                logger.warning("Unknown task_id %s in tell()", task_id)
                return
            bracket_id, rung_level, cfg = mapping
            bracket = self.brackets.get(bracket_id)
            if not bracket:
                logger.warning("Unknown bracket %s", bracket_id)
                return
            rung = bracket["rungs"][rung_level]
            rung["results"].append({"task_id": task_id, "config": cfg, "budget": budget, "value": float(objective_value)})
            expected = rung["n"]
            if len(rung["results"]) >= expected:
                logger.info("Rung %d complete for bracket %s (n=%d) -> promotion", rung_level, bracket_id, expected)
                next_level = rung_level + 1
                if next_level < len(bracket["rungs"]):
                    sorted_results = sorted(rung["results"], key=lambda x: x["value"])
                    promote_k = int(math.floor(len(sorted_results) / self.eta))
                    promote_k = max(promote_k, 1)
                    promoted = sorted_results[:promote_k]
                    next_r = bracket["rungs"][next_level]["r"]
                    for item in promoted:
                        new_task_id = self._next_task_id()
                        cfg_copy = dict(item["config"])
                        cfg_copy["__hbo_task_id"] = new_task_id
                        self.task_queue.append((new_task_id, cfg_copy, next_r))
                        self.task_map[new_task_id] = (bracket_id, next_level, cfg_copy)
                    logger.info("Promoted %d configs to rung %d at budget=%.4f", len(promoted), next_level, next_r)
                else:
                    bracket["completed"] = True
                    logger.info("Bracket %s completed", bracket_id)

    def num_pending(self) -> int:
        return len(self.task_queue)


# -----------------------
# Example smoke
# -----------------------
if __name__ == "__main__":
    params = [
        HyperParameter("n_estimators", "int", bounds=(10, 200), default=50),
        HyperParameter("max_depth", "int", bounds=(1, 20), default=5),
        HyperParameter("criterion", "categorical", choices=["gini", "entropy"], default="gini"),
        HyperParameter("use_bootstrap", "bool", default=True)
    ]
    cs = ConfigurationSpace(params)
    db = ExperimentDatabase(":memory:")
    sid = db.create_study("demo", "minimize", "loss", cs, metadata={"task": "demo"})
    trial_cfg = cs.sample_configuration()
    ok, errs = cs.validate_configuration(trial_cfg)
    print("sample cfg", trial_cfg, "valid:", ok, errs)
    if SKLEARN_AVAILABLE:
        bo = EnhancedBayesianOptimizer(cs, acquisition="ei", n_initial=3, minimize=True)
        for i in range(4):
            cfg = cs.sample_configuration()
            val = sum([v for k, v in cfg.items() if isinstance(v, (int, float))])
            bo.tell(cfg, val)
        cand = bo.suggest()
        print("Bayes suggested:", cand)
    bohb = BOHBOptimizer(cs, min_budget=1, max_budget=9, eta=3)
    for _ in range(5):
        cfg, bud = bohb.suggest()
        val = np.random.rand()
        bohb.tell(cfg, bud, val)
    print("pending tasks:", bohb.num_pending())


---

2) src/hpo/meta/meta_learner.py

(MetaLearner for warm-start guidance)

# src/hpo/meta/meta_learner.py
import numpy as np
import joblib
from typing import List, Dict, Any
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

class MetaLearner:
    """
    MetaLearner predicts expected score for (meta_features, config_vector).
    It can be used to rank candidate configs for warm-start.
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=6)
        self.trained = False

    def fit(self, meta_vectors: np.ndarray, config_vectors: np.ndarray, scores: np.ndarray):
        X = np.hstack([meta_vectors, config_vectors])
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs, scores)
        self.trained = True

    def predict(self, meta_vector: np.ndarray, config_vectors: np.ndarray) -> np.ndarray:
        if not self.trained:
            raise RuntimeError("MetaLearner not trained")
        X = np.hstack([np.repeat(meta_vector.reshape(1,-1), len(config_vectors), axis=0), config_vectors])
        Xs = self.scaler.transform(X)
        return self.model.predict(Xs)

    def save(self, path: str):
        joblib.dump((self.scaler, self.model), path)

    def load(self, path: str):
        self.scaler, self.model = joblib.load(path)
        self.trained = True


---

3) src/hpo/optimizers/bohb_kde.py

(KDE surrogate per budget level + integration helper)

# src/hpo/optimizers/bohb_kde.py
import numpy as np
from typing import List, Tuple, Optional
from sklearn.neighbors import KernelDensity
from hpo.core import ConfigurationSpace

class KDESurrogateLevel:
    def __init__(self, bandwidth: float = 1.0):
        self.bandwidth = float(bandwidth)
        self.X: List[np.ndarray] = []
        self.y: List[float] = []
        self.kde_good: Optional[KernelDensity] = None
        self.kde_bad: Optional[KernelDensity] = None

    def add(self, x: np.ndarray, y: float):
        self.X.append(x)
        self.y.append(y)

    def fit(self, top_k_ratio: float = 0.25):
        if len(self.y) == 0:
            return
        X = np.vstack(self.X)
        ys = np.array(self.y)
        idx = np.argsort(ys)  # minimize
        n = len(ys)
        k = max(1, int(np.ceil(n * top_k_ratio)))
        good_idx = idx[:k]
        bad_idx = idx[k:]
        if len(good_idx) > 0:
            self.kde_good = KernelDensity(bandwidth=self.bandwidth).fit(X[good_idx])
        if len(bad_idx) > 0:
            self.kde_bad = KernelDensity(bandwidth=self.bandwidth).fit(X[bad_idx])

    def score(self, x: np.ndarray, meta_score: float = 1.0) -> float:
        lg = self.kde_good.score_samples(x.reshape(1, -1))[0] if (self.kde_good is not None) else -1e9
        lb = self.kde_bad.score_samples(x.reshape(1, -1))[0] if (self.kde_bad is not None) else -1e9
        return float(lg - lb + np.log(max(meta_score, 1e-12)))


class BOHB_KDE:
    """
    BOHB-style controller that uses KDESurrogate per-budget.
    This is intended as a component inside a scheduler/orchestrator.
    """
    def __init__(self, cs: ConfigurationSpace, budgets: List[float], bandwidth: float = 1.0):
        self.cs = cs
        self.levels = {b: KDESurrogateLevel(bandwidth=bandwidth) for b in budgets}

    def observe(self, config: dict, budget: float, value: float):
        x = self.cs.to_array([config])[0]
        level = self.levels.get(budget)
        if level is None:
            # create if needed
            level = KDESurrogateLevel(bandwidth=1.0)
            self.levels[budget] = level
        level.add(x, float(value))

    def fit_all(self):
        for lvl in self.levels.values():
            lvl.fit()

    def propose(self, n_samples: int = 100, top_k: int = 10, meta_scores: Optional[List[float]] = None):
        # sample candidates randomly, score via lg-lb
        candidates = [self.cs.sample_configuration() for _ in range(n_samples)]
        Xc = self.cs.to_array(candidates)
        scores = []
        # use highest budget level if available
        budgets = sorted(self.levels.keys())
        if not budgets:
            return candidates[:top_k]
        lvl = self.levels[budgets[-1]]
        for i, x in enumerate(Xc):
            meta = meta_scores[i] if (meta_scores is not None and i < len(meta_scores)) else 1.0
            try:
                sc = lvl.score(x, meta_score=meta)
            except Exception:
                sc = -1e9
            scores.append(sc)
        idx = np.argsort(scores)[-top_k:]
        return [candidates[i] for i in idx[::-1]]


---

4) src/hpo/orchestrator/redis_orchestrator.py

(Redis-based scheduler helper — simple and reliable)

# src/hpo/orchestrator/redis_orchestrator.py
import redis
import json
import time
from typing import Optional, Dict, Any

class RedisOrchestrator:
    def __init__(self, redis_url: str = "redis://localhost:6379/0", queue_name: str = "hpo_tasks"):
        self.redis_url = redis_url
        self.r = redis.from_url(redis_url)
        self.queue = queue_name

    def push_task(self, cfg: Dict[str, Any], budget: float):
        task = {"cfg": cfg, "budget": float(budget), "ts": time.time()}
        self.r.lpush(self.queue, json.dumps(task))

    def pop_task(self, timeout: int = 5) -> Optional[Dict[str, Any]]:
        res = self.r.brpop(self.queue, timeout=timeout)
        if not res:
            return None
        _, raw = res
        return json.loads(raw)

    def queue_length(self) -> int:
        return int(self.r.llen(self.queue))


---

5) src/hpo/worker/worker.py

(Worker template that reads tasks and runs evaluator; pushes results to DB or callback)

# src/hpo/worker/worker.py
import json
import time
import traceback
from typing import Callable, Dict, Any

from hpo.orchestrator.redis_orchestrator import RedisOrchestrator
from hpo.evaluator.advanced_evaluator import Evaluator
from hpo.core import ExperimentDatabase

def worker_loop(redis_url: str, queue_name: str, db_path: str, result_callback: Optional[Callable] = None):
    r = RedisOrchestrator(redis_url=redis_url, queue_name=queue_name)
    db = ExperimentDatabase(db_path)
    while True:
        task = r.pop_task(timeout=10)
        if task is None:
            time.sleep(0.5)
            continue
        cfg = task["cfg"]
        budget = task.get("budget", None)
        try:
            metrics = Evaluator.train_and_eval_from_config(cfg, budget)
            # user should map to study/trial outside or pass study info in cfg
            if "study_id" in cfg and "trial_number" in cfg:
                db.save_trial(cfg["study_id"], cfg["trial_number"], cfg, metrics, status="completed", duration_seconds=metrics.get("train_time_s"))
            if result_callback:
                result_callback(cfg, budget, metrics)
        except Exception as e:
            traceback.print_exc()
            if "study_id" in cfg and "trial_number" in cfg:
                db.save_trial(cfg["study_id"], cfg["trial_number"], cfg, metrics=None, status="error", error_message=str(e))


---

6) src/hpo/evaluator/advanced_evaluator.py

(Multi-fidelity evaluator with interim reporting hook)

# src/hpo/evaluator/advanced_evaluator.py
import time
import pickle
from typing import Dict, Any, Optional

import numpy as np

class Evaluator:
    """
    Provides helper to train a model given a config. This is a light wrapper:
    - it expects model-building logic in config or a "model_factory" entry that the project provides.
    - supports 'sample_fraction' and 'epochs' as budget controls.
    """

    @staticmethod
    def train_and_eval_from_config(cfg: Dict[str, Any], budget: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Build model from cfg['model_factory'] or from provided callable and run training.
        Returns metrics dict and includes 'train_time_s' and 'model_bytes' if possible.
        """
        t0 = time.time()
        # The project must supply a model builder callable under cfg['model_builder']
        builder = cfg.get("model_builder")
        if builder is None:
            raise RuntimeError("config missing 'model_builder' callable")
        model = builder(cfg)

        # dataset must be referenced (either by ref or embedded small dataset)
        dataset = cfg.get("dataset")
        if dataset is None:
            raise RuntimeError("config missing 'dataset' (X_train,y_train[,X_val,y_val])")

        X_train = dataset["X_train"]
        y_train = dataset["y_train"]
        X_val = dataset.get("X_val")
        y_val = dataset.get("y_val")

        if budget and isinstance(budget, dict) and "sample_fraction" in budget:
            frac = float(budget["sample_fraction"])
            n = max(1, int(len(X_train) * frac))
            idx = np.random.choice(len(X_train), n, replace=False)
            X_train = X_train[idx]
            y_train = y_train[idx]

        # If the model has fit signature expect scikit-learn API
        model.fit(X_train, y_train)

        metrics = {}
        if X_val is not None and y_val is not None:
            preds = model.predict(X_val)
            try:
                from sklearn.metrics import accuracy_score, log_loss
                metrics["accuracy"] = float(accuracy_score(y_val, preds))
            except Exception:
                metrics["accuracy"] = None

        metrics["train_time_s"] = time.time() - t0
        try:
            metrics["model_bytes"] = len(pickle.dumps(model))
        except Exception:
            metrics["model_bytes"] = None

        return metrics


---

7) src/hpo/nas/nas_lite.py

(NAS-lite generator)

# src/hpo/nas/nas_lite.py
import numpy as np
from typing import List, Dict, Any

class NASLite:
    """
    Simple cell-based NAS-lite generator producing small architectures for tabular/vision toy experiments.
    The generated 'arch' is a list of ops; conversion to model must be implemented by ModelFactory.
    """
    def __init__(self, max_cells: int = 4, seed: int = 0):
        self.max_cells = int(max_cells)
        self.rng = np.random.RandomState(seed)

    def sample_arch(self) -> List[Dict[str, Any]]:
        n_cells = int(self.rng.randint(1, self.max_cells + 1))
        arch = []
        for _ in range(n_cells):
            op = self.rng.choice(["conv3", "conv5", "maxpool", "identity"])
            filters = int(2 ** self.rng.randint(5, 9))  # 32..256
            arch.append({"op": op, "filters": filters})
        return arch

    def mutate(self, arch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        new = [dict(a) for a in arch]
        idx = self.rng.randint(0, len(new))
        if self.rng.rand() < 0.5:
            new[idx]["filters"] = int(new[idx]["filters"] * (2 if self.rng.rand() < 0.5 else 0.5))
        else:
            new[idx]["op"] = self.rng.choice(["conv3", "conv5", "maxpool", "identity"])
        return new


---

8) src/hpo/parego/parego.py

(ParEGO scalarization wrapper)

# src/hpo/parego/parego.py
import numpy as np
from typing import List, Callable

def random_weight_vector(n_obj: int):
    v = np.random.rand(n_obj)
    return v / np.sum(v)

def scalarize_parego(objectives: np.ndarray, weights: np.ndarray, ideal: np.ndarray, nadir: np.ndarray):
    # objectives shape (n_points, n_obj)
    # normalize
    norm = (objectives - ideal) / (nadir - ideal + 1e-12)
    # weighted Tchebycheff:
    scalar = np.max(weights * np.abs(norm), axis=1)
    return scalar


---

9) src/hpo/generators/parameter_generator.py

(ParameterGenerator for templates)

# src/hpo/generators/parameter_generator.py
import numpy as np
from typing import Dict, Any

class ParameterGenerator:
    def __init__(self, seed: int = 0):
        self.rng = np.random.RandomState(seed)

    @staticmethod
    def heuristic_for_template(template_name: str, dataset_meta: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        if template_name == "sklearn_rf_pipeline":
            return {
                "n_estimators": {"type": "int", "bounds": (50, 1000), "default": 200},
                "max_depth": {"type": "int", "bounds": (3, 50), "default": None},
                "min_samples_leaf": {"type": "int", "bounds": (1, 20), "default": 1},
                "max_features": {"type": "categorical", "choices": ["sqrt", "log2", None], "default": "sqrt"}
            }
        if template_name == "keras_mlp":
            n_features = int(dataset_meta.get("n_features", 10))
            return {
                "hidden_layers": {"type": "int", "bounds": (1, 5), "default": 2},
                "hidden_size": {"type": "int", "bounds": (max(8, n_features), 1024), "default": 64},
                "dropout": {"type": "float", "bounds": (0.0, 0.6), "default": 0.1},
                "lr": {"type": "log_float", "bounds": (1e-5, 1e-1), "default": 1e-3}
            }
        return {}

    def sample_from_spec(self, spec: Dict[str, Any]) -> Any:
        t = spec["type"]
        if t == "int":
            low, high = spec["bounds"]
            return int(self.rng.randint(low, high + 1))
        if t == "float":
            low, high = spec["bounds"]
            return float(self.rng.uniform(low, high))
        if t == "log_float":
            low, high = spec["bounds"]
            return float(np.exp(self.rng.uniform(np.log(low), np.log(high))))
        if t == "categorical":
            return self.rng.choice(spec["choices"])
        if t == "bool":
            return bool(self.rng.choice([False, True]))
        raise ValueError("Unknown type")


---

10) src/hpo/factories/model_factory.py

(ModelFactory with sklearn/XGBoost/LightGBM placeholders)

# src/hpo/factories/model_factory.py
from typing import Dict, Any
try:
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
except Exception:
    Pipeline = None

class ModelFactory:
    def __init__(self, template_name: str):
        self.template_name = template_name

    def build(self, params: Dict[str, Any]):
        if self.template_name == "sklearn_rf_pipeline":
            if Pipeline is None:
                raise RuntimeError("scikit-learn is required for this factory")
            rf = RandomForestClassifier(
                n_estimators=int(params.get("n_estimators", 200)),
                max_depth=None if params.get("max_depth") is None else int(params.get("max_depth")),
                min_samples_leaf=int(params.get("min_samples_leaf", 1)),
                max_features=params.get("max_features", "sqrt"),
                n_jobs=1
            )
            pipe = Pipeline([("scaler", StandardScaler()), ("clf", rf)])
            return pipe
        if self.template_name == "xgboost":
            import xgboost as xgb
            return xgb.XGBClassifier(**params)
        if self.template_name == "lightgbm":
            import lightgbm as lgb
            return lgb.LGBMClassifier(**params)
        raise NotImplementedError(f"Unknown template {self.template_name}")


---

11) Tests — tests/test_meta_learner.py

# tests/test_meta_learner.py
import numpy as np
from hpo.meta.meta_learner import MetaLearner

def test_meta_learner_basic():
    # synthetic meta vectors (2 features), config vectors (3 features)
    meta = np.random.RandomState(0).rand(100,2)
    cfg = np.random.RandomState(1).rand(100,3)
    # objective: linear combination
    scores = (meta[:,0]*0.3 + cfg[:,1]*0.5 + np.random.RandomState(2).rand(100)*0.01)
    ml = MetaLearner()
    ml.fit(meta, cfg, scores)
    preds = ml.predict(meta[0], cfg[:5])
    assert preds.shape == (5,)

12) Tests — tests/test_bohb_kde.py

# tests/test_bohb_kde.py
import numpy as np
from hpo.optimizers.bohb_kde import BOHB_KDE
from hpo.core import HyperParameter, ConfigurationSpace

def test_bohb_kde_flow():
    params = [HyperParameter("a","int",(0,10)), HyperParameter("b","float",(0.0,1.0))]
    cs = ConfigurationSpace(params)
    bo = BOHB_KDE(cs, budgets=[1.0, 3.0, 9.0], bandwidth=0.5)
    # add observations
    for _ in range(20):
        cfg = cs.sample_configuration()
        x = cs.to_array([cfg])[0]
        val = np.random.rand()
        bo.observe(cfg, 1.0, val)
    bo.fit_all()
    props = bo.propose(n_samples=50, top_k=5)
    assert isinstance(props, list) and len(props) <= 5

13) Tests — tests/test_core_smoke.py

# tests/test_core_smoke.py
from hpo.core import ConfigurationSpace, HyperParameter, ExperimentDatabase, BOHBOptimizer

def test_configuration_sampling_and_db():
    params = [HyperParameter("n","int",(1,5)), HyperParameter("flag","bool")]
    cs = ConfigurationSpace(params)
    cfg = cs.sample_configuration()
    ok, errs = cs.validate_configuration(cfg)
    assert ok
    db = ExperimentDatabase(":memory:")
    sid = db.create_study("t1","minimize","loss",cs, metadata={"task":"smoke"})
    tid = db.save_trial(sid, 0, cfg, metrics={"loss":0.5}, status="completed", duration_seconds=0.1)
    rows = db.get_study_trials(sid)
    assert len(rows) == 1
    bohb = BOHBOptimizer(cs, min_budget=1, max_budget=9, eta=3)
    c,b = bohb.suggest()
    assert isinstance(c, dict)


---

14) requirements-dev.txt

numpy
scipy
scikit-learn
lightgbm
xgboost
optuna
redis
rq
joblib
shap
pytest
psutil
pandas
scikit-optimize


---

15) .github/workflows/ci.yml (minimal CI)

name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install deps
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements-dev.txt
      - name: Run tests
        run: |
          pytest -q


---

تعليمات التشغيل محليًا (خطوات عملية)

1. انسخ الملفات إلى المستودع بالمسارات المشار إليها (أنشئ المجلدات src/hpo/... و tests/).


2. أنشئ virtualenv وثبّت الاعتماديات:



python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-dev.txt

> إن لم تكن تريد كل الحزم (xgboost/lightgbm) أزلها مؤقتًا من requirements-dev.txt.



3. شغّل الاختبارات:



pytest -q

4. لتجربة الـ Redis orchestrator وworkers (اختياري — للتوزيع):



شغل Redis محليًا (Linux/Mac):


redis-server &

شغل worker (مثال):


python -c "from hpo.worker.worker import worker_loop; worker_loop('redis://localhost:6379/0','hpo_tasks','hpo_experiments.db')"

ادفع مهمة تجريبية باستخدام RedisOrchestrator.push_task(cfg,budget).


5. لاحظ: بعض اختبارات/وظائف تتطلب scikit-learn و scipy — تأكد أنها مثبتة.




---

ملاحظات مهمة وصادقة

لقد طورت وأعطيت مكتبة متكاملة، 