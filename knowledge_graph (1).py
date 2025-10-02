
# Combined knowledge_graph.py module (for demonstration/export only)

# === schema.py ===
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, List

class NodeType(Enum):
    PROJECT = "Project"
    TASK = "Task"
    PERSON = "Person"
    TOOL = "Tool"
    METRIC = "Metric"

@dataclass
class NodeSchema:
    type: NodeType
    properties: Dict[str, Any]
    version: int = 1
    description: str = ""
    parent: NodeType = None

@dataclass
class RelationshipSchema:
    name: str
    source: NodeType
    target: NodeType
    metadata: Dict[str, Any] = field(default_factory=dict)
    directed: bool = True

ONTOLOGY = {
    "nodes": [
        NodeSchema(NodeType.PROJECT, {"name": "string", "deadline": "date"}),
        NodeSchema(NodeType.TASK, {"title": "string", "status": "string"}, parent=NodeType.PROJECT),
        NodeSchema(NodeType.PERSON, {"name": "string", "role": "string"}),
    ],
    "relationships": [
        RelationshipSchema("depends_on", NodeType.TASK, NodeType.TASK),
        RelationshipSchema("owned_by", NodeType.TASK, NodeType.PERSON),
    ]
}

# === graph_engine.py ===
from py2neo import Graph, Node, Relationship

class GraphEngine:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", pwd="password"):
        self.graph = Graph(uri, auth=(user, pwd))
        self._ensure_constraints()

    def _ensure_constraints(self):
        for schema in ONTOLOGY["nodes"]:
            label = schema.type.value
            self.graph.run(f"CREATE CONSTRAINT IF NOT EXISTS ON (n:{label}) ASSERT n.id IS UNIQUE")

    def create_node(self, node_id: str, schema: NodeSchema, properties: Dict[str, Any]):
        node = Node(schema.type.value, id=node_id, **properties)
        self.graph.create(node)
        return node

    def create_relationship(self, source_id: str, rel_schema: RelationshipSchema, target_id: str, metadata: Dict[str, Any] = None):
        src = self.graph.nodes.match(rel_schema.source.value, id=source_id).first()
        tgt = self.graph.nodes.match(rel_schema.target.value, id=target_id).first()
        rel = Relationship(src, rel_schema.name, tgt, **(metadata or {}))
        self.graph.create(rel)
        return rel

# === inference_engine.py ===
from torch_geometric.nn import GCNConv
import torch

class InferenceEngine:
    def __init__(self, model=None):
        self.model = model or GCNConv(16, 16)

    def predict_links(self, embeddings, edges):
        out = self.model(embeddings, edges)
        return torch.sigmoid(out)

    def rule_based_checks(self, graph_engine):
        query = "MATCH (t:Task) WHERE NOT (t)-[:owned_by]->() RETURN t.id"
        return graph_engine.graph.run(query).data()

# === integration.py ===
class KnowledgeGraphCoordinator:
    def __init__(self, **cfg):
        self.ge = GraphEngine(**cfg)
        self.ie = InferenceEngine()

    def bootstrap_entities(self, model_spec):
        for idx, task in enumerate(model_spec.get('tasks', [])):
            self.ge.create_node(f"task_{idx}", ONTOLOGY['nodes'][1], task)

    def log_results(self, collab_result):
        self.ge.create_node("hpo_result", ONTOLOGY['nodes'][4], {
            'score': collab_result['hpo']['best_params'].get('lr', 0.001)
        })

    def analyze_and_predict(self):
        rules = self.ie.rule_based_checks(self.ge)
        return {'rules': rules}

    def coverage_report(self):
        return {
            'nodes': len(list(self.ge.graph.nodes)),
            'rels': len(list(self.ge.graph.relationships))
        }
