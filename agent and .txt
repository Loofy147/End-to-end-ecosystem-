# Fixing bugs (import math, correct grid_search_weights) and re-running tuning + generation.
import time, ast, textwrap, json, random, os, math
from typing import Dict, Any, List, Tuple
import networkx as nx

# Reuse PROCESS_TESTS and utilities from previous cell
PROCESS_TESTS = [
    ("normal_list", [1, 2, 3], "len_exact"),
    ("empty_list", [], "len_exact"),
    ("none", None, "total_nonneg"),
    ("wrong_type", 'notalist', "raises"),
    ("large_list", list(range(2000)), "len_exact"),
    ("list_with_none", [1, None, 3], "len_exact"),
    ("nested_list", [[1,2], [3]], "len_exact"),
    ("mixed_elements", [1, 'a', None], "len_exact"),
    ("generator_input", (i for i in range(5)), "raises"),
]

def safe_compile(code: str) -> bool:
    try:
        compile(code, '<string>', 'exec')
        return True
    except Exception:
        return False

class CyclomaticComplexityVisitor(ast.NodeVisitor):
    def __init__(self):
        self.count = 0
    def generic_visit(self, node):
        from ast import If, For, While, With, Try, BoolOp, IfExp, Compare
        if isinstance(node, (If, For, While, Try, With, IfExp)):
            self.count += 1
        if isinstance(node, ast.BoolOp):
            self.count += len(node.values) - 1
        super().generic_visit(node)

def cyclomatic_complexity(code: str) -> int:
    try:
        tree = ast.parse(code)
    except Exception:
        return 0
    v = CyclomaticComplexityVisitor()
    v.visit(tree)
    return max(1, 1 + v.count)

SECURITY_PATTERNS = ['exec(', 'eval(', 'os.system', 'subprocess', 'open(', 'pickle.loads', 'requests.']
def detect_security_issues(code: str):
    issues = []
    lower = code.lower()
    for p in SECURITY_PATTERNS:
        if p in lower:
            issues.append(p)
    return issues

# Graph features
def spec_to_nx_graph(spec: str) -> nx.DiGraph:
    words = [w.strip(',.()') for w in spec.lower().split()]
    nodes = list(dict.fromkeys([w for w in words if len(w) > 2]))
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    edges = []
    for i in range(len(nodes)-1):
        a, b = nodes[i], nodes[i+1]
        edges.append((a,b))
    G.add_edges_from(edges)
    return G

def graph_features(G: nx.DiGraph) -> Dict[str, Any]:
    node_count = G.number_of_nodes()
    edge_count = G.number_of_edges()
    degrees = dict(G.degree())
    avg_deg = sum(degrees.values())/node_count if node_count>0 else 0.0
    pr_max = 0.0; pr_entropy = 0.0
    if node_count > 0:
        try:
            pr = nx.pagerank(G)
            pr_vals = list(pr.values())
            pr_max = max(pr_vals) if pr_vals else 0.0
            pr_entropy = -sum([p * (0 if p<=0 else math.log(p)) for p in pr_vals]) if pr_vals else 0.0
        except Exception:
            pr_max = 0.0; pr_entropy = 0.0
    return {'node_count': node_count, 'edge_count': edge_count, 'avg_deg': avg_deg, 'pagerank_max': pr_max, 'pagerank_entropy': pr_entropy}

# Fitness
class FitnessEvaluatorTunable:
    def __init__(self, coverage_w=0.6, cyc_w=0.2, exec_w=0.1, sec_penalty=0.25, timeout=0.05):
        self.coverage_w = coverage_w
        self.cyc_w = cyc_w
        self.exec_w = exec_w
        self.sec_penalty = sec_penalty
        self.timeout = timeout

    def _check_test(self, func, inp, expect_type):
        try:
            if expect_type == "len_exact":
                res = func(inp)
                return isinstance(res, dict) and 'total' in res and res['total'] == len(inp)
            elif expect_type == "total_nonneg":
                res = func(inp)
                return isinstance(res, dict) and 'total' in res and isinstance(res['total'], int) and res['total'] >= 0
            elif expect_type == "raises":
                try:
                    func(inp)
                    return False
                except Exception:
                    return True
            else:
                res = func(inp)
                return isinstance(res, dict) and 'total' in res
        except Exception:
            return expect_type == "raises"

    def evaluate(self, code: str) -> Dict[str, Any]:
        metrics = {'syntax_ok': False, 'cyclomatic':0, 'exec_time':None, 'test_coverage':0.0, 'security_issues':[], 'score':0.0, 'per_test':{}}
        metrics['syntax_ok'] = safe_compile(code)
        if not metrics['syntax_ok']:
            return metrics
        try:
            tree = ast.parse(code)
            v = CyclomaticComplexityVisitor(); v.visit(tree); metrics['cyclomatic'] = max(1,1+v.count)
        except Exception:
            metrics['cyclomatic']=0
        metrics['security_issues'] = detect_security_issues(code)
        exec_env={}
        try:
            exec(code, exec_env)
        except Exception:
            return metrics
        func = exec_env.get('process_interactions') or exec_env.get('process_interaction')
        total_tests = len(PROCESS_TESTS)
        passed = 0; total_time=0.0; per_test={}
        if func and callable(func):
            for name, inp, expect_type in PROCESS_TESTS:
                start=time.time()
                ok=False
                try:
                    ok = self._check_test(func, inp, expect_type)
                except Exception:
                    ok=False
                took=time.time()-start; total_time+=took; per_test[name]=ok
                if ok: passed+=1
        else:
            for name,_,_ in PROCESS_TESTS: per_test[name]=False
            passed=0; total_time=0.0
        metrics['exec_time']=total_time; metrics['per_test']=per_test; metrics['test_coverage']=passed/total_tests
        score=0.0
        score += self.coverage_w * metrics['test_coverage']
        cyc = metrics['cyclomatic']; cyc_penalty = min(self.cyc_w, (cyc-1)*0.03); score += max(0.0, self.cyc_w - cyc_penalty)
        if metrics['exec_time'] is not None:
            exec_penalty = min(self.exec_w, metrics['exec_time']/(self.timeout*20+1e-9)); score += max(0.0, self.exec_w - exec_penalty)
        sec = len(metrics['security_issues']); score -= self.sec_penalty * sec
        metrics['score']=max(0.0, min(1.0, score))
        return metrics

# Candidate generator influenced by graph features + some extra original excerpts if present
def generate_candidates_from_spec(spec: str) -> Tuple[List[str], Dict[str,Any]]:
    G = spec_to_nx_graph(spec)
    feats = graph_features(G)
    candidates = []
    base = textwrap.dedent("""\
        def process_interactions(records):
            \"\"\"Process list of user interaction records and return a summary\"\"\"
            if not isinstance(records, list):
                raise ValueError('records must be a list')
            total = len(records)
            return {'total': total}
    """)
    candidates.append(base)
    if feats['avg_deg'] > 0.5:
        graph_code = textwrap.dedent(f"""\
            def process_interactions(records):
                if records is None:
                    return {{'total': 0}}
                total = len(records)
                anomalies = []
                # graph-guided: avg_deg={feats['avg_deg']:.2f}
                if total > 500:
                    anomalies.append('too_large')
                return {{'total': total, 'anomalies': anomalies}}
        """)
    else:
        graph_code = textwrap.dedent("""\
            def process_interactions(records):
                if records is None:
                    return {'total': 0}
                total = len(records)
                anomalies = []
                return {'total': total, 'anomalies': anomalies}
        """)
    candidates.append(graph_code)
    # include excerpts from original file (larger)
    paths = ['/mnt/data/breakthrough generator .py', '/mnt/data/breakthrough_generator.py']
    for p in paths:
        try:
            if os.path.exists(p):
                with open(p, 'r', encoding='utf-8', errors='ignore') as f:
                    txt = f.read(20000)
                    if txt and txt not in candidates:
                        candidates.append(txt)
                        # also extract function snippets
                        try:
                            tree = ast.parse(txt)
                            funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
                            for f in funcs[:5]:
                                try:
                                    s = ast.unparse(f)
                                except Exception:
                                    s = None
                                if s and s not in candidates:
                                    candidates.append(s)
                        except Exception:
                            pass
        except Exception:
            pass
    # create variants and hybrids
    for i in range(6):
        if i%2==0:
            v = base.replace("\n    total = len(records)\n", "\n    if records is None:\n        return {'total': 0}\n    if not isinstance(records, list):\n        return {'total': 0}\n    total = len(records)\n")
        else:
            v = base + f"\n# variant {i}\n"
        candidates.append(v)
    candidates.append(ast_hybridize(base, graph_code))
    return candidates, feats

# Grid search function corrected
def grid_search_weights(candidates: List[str], feas: List[Tuple[float,float,float,float]]) -> Tuple[Tuple[float,float,float,float], Dict]:
    best_conf=None; best_score=-1; best_report=None
    for (cov_w, cyc_w, exec_w, sec_pen) in feas:
        fe = FitnessEvaluatorTunable(coverage_w=cov_w, cyc_w=cyc_w, exec_w=exec_w, sec_penalty=sec_pen)
        evals = [(fe.evaluate(c), i, c) for i,c in enumerate(candidates)]
        evals.sort(key=lambda x: x[0]['score'], reverse=True)
        topn = evals[:3] if len(evals)>=3 else evals
        if not topn:
            continue
        avg_top = sum([t[0]['score'] for t in topn])/len(topn)
        if avg_top > best_score:
            best_score = avg_top
            best_conf = (cov_w, cyc_w, exec_w, sec_pen)
            best_report = {'conf':best_conf, 'avg_top':avg_top, 'top':[(t[0]['score'], t[1]) for t in topn], 'evals': [(m['score'], idx) for m,idx,_ in evals]}
    return best_conf, best_report

# Run
spec = ("Process a large dataset of user interactions, run ML-based anomaly detection, "
        "generate a report, be secure and efficient. Use graph-aware heuristics.")

candidates, feats = generate_candidates_from_spec(spec)
print("Graph features:", feats)
print("Candidates count:", len(candidates))

cov_w_list = [0.5, 0.6, 0.7, 0.8]
cyc_w_list = [0.1, 0.15, 0.2]
exec_w_list = [0.05, 0.1]
sec_pen_list = [0.25, 0.5]
feas = [(a,b,c,d) for a in cov_w_list for b in cyc_w_list for c in exec_w_list for d in sec_pen_list]

best_conf, best_report = grid_search_weights(candidates, feas)
if best_conf is None:
    best_conf = (0.6, 0.2, 0.1, 0.25)

fe_final = FitnessEvaluatorTunable(coverage_w=best_conf[0], cyc_w=best_conf[1], exec_w=best_conf[2], sec_penalty=best_conf[3])
final_evals = [(fe_final.evaluate(c), i, c) for i,c in enumerate(candidates)]
final_evals.sort(key=lambda x: x[0]['score'], reverse=True)

summary = {
    'graph_features': feats,
    'generated_count': len(candidates),
    'best_weight_config': {'coverage_w':best_conf[0], 'cyc_w':best_conf[1], 'exec_w':best_conf[2], 'sec_penalty':best_conf[3]},
    'best_top3_avg_score': best_report['avg_top'] if best_report else None,
    'top_candidates': [
        {'rank': r+1, 'idx': i, 'score': m['score'], 'coverage': m['test_coverage'], 'cyclomatic': m['cyclomatic'], 'sec_issues': m['security_issues'], 'per_test': m.get('per_test')} 
        for r,(m,i,c) in enumerate(final_evals[:5])
    ]
}

out_path = '/mnt/data/gnn_tuning_report_fixed.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2)

print("Best config:", summary['best_weight_config'])
print("Best top3 avg score:", summary['best_top3_avg_score'])
print("\nTop candidates:")
for r,t in enumerate(summary['top_candidates'], start=1):
    print(f"Rank {r}: idx={t['idx']} score={t['score']:.3f} coverage={t['coverage']:.2f} cyclomatic={t['cyclomatic']} sec_issues={len(t['sec_issues'])}")
    print("Per-test:", t['per_test'])
    print("-"*60)

print("\nReport saved to:", out_path)
