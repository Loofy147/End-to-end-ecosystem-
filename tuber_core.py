import uuid
import networkx as nx
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import logging

from config import Config

logging.basicConfig(level=Config.LOG_LEVEL, format="%(asctime)s - %(levelname)s - %(message)s")

@dataclass
class TuberNode:
    """
    Represents a 'drna' (tuber) in the metaphorical model.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    node_type: str = "subtuber"  # 'root', 'subtuber', 'latent', 'meta_knowledge', 'decision_log', 'challenge_discovery'
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None

    def __post_init__(self):
        # Initialize metadata with default values if not present
        self.metadata.setdefault('predicted_future_reward', Config.INITIAL_REWARD_PREDICTION)
        self.metadata.setdefault('prediction_confidence', 0.5)
        self.metadata.setdefault('short_term_reward', 0.0)
        self.metadata.setdefault('long_term_reward', 0.0)
        self.metadata.setdefault('last_prediction_error', 0.0)
        self.metadata.setdefault('value_score', 0.0) # Overall value for pruning
        self.metadata.setdefault('expected_regret_cost', 0.0)
        self.metadata.setdefault('transport_speed', Config.DEFAULT_TRANSPORT_SPEED)
        self.metadata.setdefault('transport_cost', Config.DEFAULT_TRANSPORT_COST)
        self.metadata.setdefault('transport_capacity', Config.DEFAULT_TRANSPORT_CAPACITY)
        self.metadata.setdefault('last_llm_insight_time', 0.0)
        self.metadata.setdefault('creation_time', time.time())


class MetaphoricalTuberingModel:
    """
    A dynamic, branching knowledge graph inspired by tubers:
    - Nodes grow in depth and breadth
    - Can seed ideas, expand, prune, and query latent knowledge
    - Implements the Active Knowledge Triangle (AKT) for dynamic focus.
    """
    def __init__(self):
        self.graph = nx.DiGraph()  # Directed graph: edges point from parent to child
        self.root_id: Optional[str] = None
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2') # For semantic similarity
        self.reward_history = defaultdict(list) # Track reward history per tuber
        self.prediction_confidence = defaultdict(float) # Track prediction certainty
        self.active_knowledge_triangle: List[str] = [] # Stores IDs of the 3 active tubers
        self.llm_interface = None # Will be set by TuberOrchestratorAI

    def set_llm_interface(self, llm_interface_instance):
        self.llm_interface = llm_interface_instance

    def _embed_content(self, content: str) -> List[float]:
        """Generates an embedding for the given content."""
        return self.embedder.encode(content).tolist()

    def _calculate_semantic_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculates cosine similarity between two embeddings."""
        if not embedding1 or not embedding2:
            return 0.0
        return cosine_similarity(np.array(embedding1).reshape(1, -1), np.array(embedding2).reshape(1, -1))[0][0]

    def seed_idea(self, content: str, node_type: str = 'root') -> str:
        """
        Create the root tuber (الفكرة الأصلية).
        """
        embedding = self._embed_content(content)
        node = TuberNode(content=content, node_type=node_type, embedding=embedding)
        self.graph.add_node(node.id, data=node)
        self.root_id = node.id
        self.active_knowledge_triangle = [node.id] * 3 # Initialize AKT with root
        logging.info(f"Root tuber seeded: {node.id} - '{node.content[:50]}...' ")
        return node.id

    def expand(self, parent_id: str, directions: List[str], expansion_type: str = 'general') -> List[str]:
        """
        Expand a parent tuber into new child tubers in different conceptual directions.
        Uses LLM to generate content for new tubers based on parent and direction.
        """
        if parent_id not in self.graph:
            logging.warning(f"Parent tuber {parent_id} not found for expansion.")
            return []

        parent_node = self.graph.nodes[parent_id]['data']
        new_tuber_ids = []

        logging.info(f"Expanding tuber {parent_id} ('{parent_node.content[:30]}...') in directions: {directions}")

        for direction in directions:
            prompt = f"""
            Given the parent concept: '{parent_node.content}'
            Expand on this concept in the following direction/aspect: '{direction}'.
            Focus on a '{expansion_type}' type of expansion.
            Provide a concise, single-paragraph content for the new tuber.
            """
            if self.llm_interface:
                new_content, cost = self.llm_interface.generate_text(prompt, max_tokens=200, temperature=0.7)
                if new_content.startswith("Error:"):
                    logging.error(f"LLM error during expansion: {new_content}")
                    continue
            else:
                new_content = f"Expansion of '{parent_node.content[:20]}...' in direction '{direction}'"
                cost = 0.0

            embedding = self._embed_content(new_content)
            new_node = TuberNode(content=new_content, node_type='subtuber', embedding=embedding)
            self.graph.add_node(new_node.id, data=new_node)
            self.graph.add_edge(parent_id, new_node.id, type=expansion_type, direction=direction, cost=cost)
            new_tuber_ids.append(new_node.id)
            logging.info(f"  - Created child tuber {new_node.id} for direction '{direction}'")

        return new_tuber_ids

    def prune(self) -> List[str]:
        """
        Prune tubers with low value scores, excluding root and active knowledge triangle.
        """
        pruned_ids = []
        for node_id in list(self.graph.nodes):
            if node_id == self.root_id or node_id in self.active_knowledge_triangle:
                continue

            node_data = self.graph.nodes[node_id]['data']
            if node_data.metadata.get('value_score', 0.0) < Config.PRUNING_THRESHOLD:
                self.graph.remove_node(node_id)
                pruned_ids.append(node_id)
                logging.info(f"Pruned tuber: {node_id} - '{node_data.content[:30]}...' (Value: {node_data.metadata.get('value_score'):.2f})")
        return pruned_ids

    def traverse(self, start_node_id: Optional[str] = None, depth: int = 3) -> Dict:
        """
        Traverse the tuber network from a start node (or root) up to a certain depth.
        """
        if not self.graph.nodes:
            return {"nodes": [], "edges": []}

        if start_node_id is None:
            start_node_id = self.root_id
            if start_node_id is None:
                return {"nodes": [], "edges": []}

        visited_nodes = set()
        nodes_to_visit = [(start_node_id, 0)]
        traversed_nodes_data = []
        traversed_edges_data = []

        while nodes_to_visit:
            current_node_id, current_depth = nodes_to_visit.pop(0)

            if current_node_id in visited_nodes or current_depth > depth:
                continue

            visited_nodes.add(current_node_id)
            node_data = self.graph.nodes[current_node_id]['data']
            traversed_nodes_data.append({
                'id': node_data.id,
                'content': node_data.content,
                'type': node_data.node_type,
                'metadata': node_data.metadata
            })

            for neighbor_id in self.graph.neighbors(current_node_id):
                edge_data = self.graph.get_edge_data(current_node_id, neighbor_id)
                traversed_edges_data.append({
                    'source': current_node_id,
                    'target': neighbor_id,
                    'type': edge_data.get('type', 'relates_to'),
                    'direction': edge_data.get('direction', 'N/A'),
                    'cost': edge_data.get('cost', 0.0)
                })
                nodes_to_visit.append((neighbor_id, current_depth + 1))

        return {'nodes': traversed_nodes_data, 'edges': traversed_edges_data}

    def to_dict(self) -> Dict:
        """
        Serialize the graph to a dictionary for easy representation/storage.
        """
        nodes = [{
            'id': nid,
            'content': self.graph.nodes[nid]['data'].content,
            'type': self.graph.nodes[nid]['data'].node_type,
            'metadata': self.graph.nodes[nid]['data'].metadata,
            'embedding': self.graph.nodes[nid]['data'].embedding # Include embedding for external use if needed
        } for nid in self.graph.nodes]
        edges = [{
            'source': u,
            'target': v,
            'type': self.graph.get_edge_data(u, v).get('type', 'relates_to'),
            'direction': self.graph.get_edge_data(u, v).get('direction', 'N/A'),
            'cost': self.graph.get_edge_data(u, v).get('cost', 0.0)
        } for u, v in self.graph.edges]
        return {'nodes': nodes, 'edges': edges, 'root_id': self.root_id, 'active_knowledge_triangle': self.active_knowledge_triangle}

    def update_active_knowledge_triangle(self, new_tuber_id: str):
        """
        Updates the Active Knowledge Triangle (AKT) by replacing one of its members.
        The replacement strategy can be simple (e.g., random, least relevant) or complex (LLM-driven).
        For now, a simple replacement of the oldest/least recently used tuber.
        """
        if len(self.active_knowledge_triangle) < 3:
            self.active_knowledge_triangle.append(new_tuber_id)
            logging.info(f"AKT: Added {new_tuber_id}. Current AKT: {self.active_knowledge_triangle}")
        else:
            # Simple strategy: replace the first (oldest) element
            old_tuber_id = self.active_knowledge_triangle.pop(0)
            self.active_knowledge_triangle.append(new_tuber_id)
            logging.info(f"AKT: Replaced {old_tuber_id} with {new_tuber_id}. Current AKT: {self.active_knowledge_triangle}")

    def find_similar_tubers(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Finds tubers semantically similar to a query embedding.
        Returns a list of (tuber_id, similarity_score) tuples.
        """
        similarities = []
        for node_id in self.graph.nodes:
            node_data = self.graph.nodes[node_id]['data']
            if node_data.embedding is not None:
                similarity = self._calculate_semantic_similarity(query_embedding, node_data.embedding)
                similarities.append((node_id, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def get_tuber_content(self, tuber_id: str) -> Optional[str]:
        """
        Retrieves the content of a specific tuber.
        """
        if tuber_id in self.graph.nodes:
            return self.graph.nodes[tuber_id]['data'].content
        return None

    def get_tuber_data(self, tuber_id: str) -> Optional[TuberNode]:
        """
        Retrieves the TuberNode object for a specific tuber.
        """
        if tuber_id in self.graph.nodes:
            return self.graph.nodes[tuber_id]['data']
        return None

    def update_tuber_metadata(self, tuber_id: str, key: str, value: Any):
        """
        Updates a specific metadata field for a tuber.
        """
        if tuber_id in self.graph.nodes:
            self.graph.nodes[tuber_id]['data'].metadata[key] = value
            logging.debug(f"Updated metadata for {tuber_id}: {key} = {value}")
        else:
            logging.warning(f"Tuber {tuber_id} not found for metadata update.")


# Example usage (for testing):
if __name__ == "__main__":
    # This part requires internet access to download the sentence-transformers model
    # and a working LLM setup if you want to test expansion.
    
    # Mock LLMInterface for basic testing without actual API calls
    class MockLLMInterface:
        def generate_text(self, prompt: str, max_tokens: Optional[int] = None, temperature: Optional[float] = None) -> Tuple[str, float]:
            if "expand" in prompt.lower():
                return "This is a mock expansion content.", 0.001
            return "Mock response.", 0.001

    model = MetaphoricalTuberingModel()
    model.set_llm_interface(MockLLMInterface())

    # Seed the root idea
    root_id = model.seed_idea("The core vision of a self-evolving AI system.")
    print(f"Root Tuber ID: {root_id}")

    # Expand the root idea
    child_ids = model.expand(root_id, ['architecture', 'learning_mechanisms', 'interaction_patterns'])
    print(f"Child Tuber IDs: {child_ids}")

    # Update AKT (simulated new tuber discovery)
    if child_ids:
        model.update_active_knowledge_triangle(child_ids[0])

    # Traverse the graph
    graph_data = model.traverse()
    print("\n--- Traversed Graph ---")
    print(json.dumps(graph_data, indent=2))

    # Prune (will not prune root or AKT members)
    print("\n--- Pruning (if any) ---")
    pruned = model.prune()
    print(f"Pruned tubers: {pruned}")

    # Find similar tubers
    query_embedding = model._embed_content("How does the system learn?")
    similar_tubers = model.find_similar_tubers(query_embedding)
    print("\n--- Similar Tubers ---")
    for tid, score in similar_tubers:
        content = model.get_tuber_content(tid)
        print(f"ID: {tid}, Similarity: {score:.4f}, Content: '{content[:50]}...' ")

    print("\n--- Full Graph (Serialized) ---")
    full_graph_dict = model.to_dict()
    print(json.dumps(full_graph_dict, indent=2))

    print(f"\nActive Knowledge Triangle: {model.active_knowledge_triangle}")




