import networkx as nx
import json
import ast
import os
import re
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import sqlite3
import hashlib

@dataclass
class Entity:
    id: str
    type: str  # 'function', 'class', 'module', 'requirement', 'test', 'bug', 'concept', 'outcome'
    name: str
    properties: Dict[str, Any]
    created_at: str = ""

@dataclass
class Relationship:
    source_id: str
    target_id: str
    relationship_type: str  # 'calls', 'implements', 'tests', 'depends_on', 'expects', 'causes', etc.
    properties: Dict[str, Any]
    created_at: str = ""

class KnowledgeGraph:
    """
    Enhanced Knowledge Graph for Unified Orchestrator AI
    A web of entities and relationships, context, logics, connected concepts, 
    outcomes, expectations, and exceptions
    """
    
    def __init__(self, db_path: str = "knowledge_graph.db"):
        self.graph = nx.DiGraph()
        self.db_path = db_path
        self.init_database()
        self._setup_core_concepts()
    
    def init_database(self):
        """Initialize the enhanced Knowledge Graph database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Entities table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                name TEXT NOT NULL,
                properties TEXT,
                created_at TEXT,
                updated_at TEXT,
                access_count INTEGER DEFAULT 0
            )
        ''')
        
        # Relationships table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id TEXT,
                target_id TEXT,
                relationship_type TEXT,
                properties TEXT,
                strength REAL DEFAULT 1.0,
                created_at TEXT,
                FOREIGN KEY (source_id) REFERENCES entities (id),
                FOREIGN KEY (target_id) REFERENCES entities (id)
            )
        ''')
        
        # Outcomes table (for learning from results)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_id TEXT,
                outcome_type TEXT,  -- 'test_result', 'static_analysis', 'performance', 'user_feedback'
                outcome_data TEXT,
                success BOOLEAN,
                timestamp TEXT,
                FOREIGN KEY (entity_id) REFERENCES entities (id)
            )
        ''')
        
        # Expectations table (for tracking what should happen)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS expectations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_id TEXT,
                expectation_type TEXT,  -- 'performance', 'behavior', 'output', 'error'
                expectation_data TEXT,
                met BOOLEAN,
                timestamp TEXT,
                FOREIGN KEY (entity_id) REFERENCES entities (id)
            )
        ''')
        
        # Context table (for storing contextual information)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS context (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_id TEXT,
                context_type TEXT,  -- 'usage', 'environment', 'dependencies', 'constraints'
                context_data TEXT,
                relevance_score REAL DEFAULT 1.0,
                timestamp TEXT,
                FOREIGN KEY (entity_id) REFERENCES entities (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _setup_core_concepts(self):
        """Setup core concepts and relationships for the AI system"""
        core_concepts = [
            Entity("concept_code_quality", "concept", "Code Quality", 
                   {"description": "Measures of code maintainability, readability, and efficiency"}),
            Entity("concept_test_coverage", "concept", "Test Coverage", 
                   {"description": "Percentage of code covered by automated tests"}),
            Entity("concept_performance", "concept", "Performance", 
                   {"description": "Runtime efficiency and resource usage"}),
            Entity("concept_security", "concept", "Security", 
                   {"description": "Protection against vulnerabilities and threats"}),
            Entity("concept_maintainability", "concept", "Maintainability", 
                   {"description": "Ease of modifying and extending code"}),
            Entity("concept_user_experience", "concept", "User Experience", 
                   {"description": "Quality of interaction between user and system"}),
        ]
        
        for concept in core_concepts:
            self.add_entity(concept)
    
    def add_entity(self, entity: Entity) -> bool:
        """Add an entity to the knowledge graph"""
        try:
            now = datetime.now().isoformat()
            entity.created_at = entity.created_at or now
            
            # Add to NetworkX graph
            self.graph.add_node(entity.id, **{
                'type': entity.type,
                'name': entity.name,
                'properties': entity.properties,
                'created_at': entity.created_at
            })
            
            # Add to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO entities (id, type, name, properties, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (entity.id, entity.type, entity.name, json.dumps(entity.properties), 
                  entity.created_at, now))
            conn.commit()
            conn.close()
            
            return True
        except Exception as e:
            print(f"Error adding entity: {e}")
            return False
    
    def add_relationship(self, relationship: Relationship, strength: float = 1.0) -> bool:
        """Add a relationship to the knowledge graph with strength weighting"""
        try:
            relationship.created_at = datetime.now().isoformat()
            
            # Add to NetworkX graph
            self.graph.add_edge(
                relationship.source_id,
                relationship.target_id,
                relationship_type=relationship.relationship_type,
                properties=relationship.properties,
                strength=strength,
                created_at=relationship.created_at
            )
            
            # Add to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO relationships (source_id, target_id, relationship_type, properties, strength, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (relationship.source_id, relationship.target_id, relationship.relationship_type, 
                  json.dumps(relationship.properties), strength, relationship.created_at))
            conn.commit()
            conn.close()
            
            return True
        except Exception as e:
            print(f"Error adding relationship: {e}")
            return False
    
    def parse_python_file(self, file_path: str) -> List[Entity]:
        """Enhanced Python file parsing with more detailed entity extraction"""
        entities = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Extract module entity
            module_name = os.path.basename(file_path).replace('.py', '')
            content_hash = hashlib.md5(content.encode()).hexdigest()
            
            module_entity = Entity(
                id=f"module_{module_name}",
                type="module",
                name=module_name,
                properties={
                    "file_path": file_path,
                    "content_hash": content_hash,
                    "line_count": len(content.split('\n')),
                    "imports": self._extract_imports(tree),
                    "complexity": self._calculate_complexity(tree)
                }
            )
            entities.append(module_entity)
            
            # Extract functions and classes with enhanced properties
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_entity = Entity(
                        id=f"function_{module_name}_{node.name}",
                        type="function",
                        name=node.name,
                        properties={
                            "module": module_name,
                            "line_number": node.lineno,
                            "args": [arg.arg for arg in node.args.args],
                            "docstring": ast.get_docstring(node),
                            "is_async": isinstance(node, ast.AsyncFunctionDef),
                            "complexity": len([n for n in ast.walk(node) if isinstance(n, (ast.If, ast.For, ast.While))]),
                            "return_type": self._extract_return_type(node),
                            "decorators": [d.id for d in node.decorator_list if isinstance(d, ast.Name)]
                        }
                    )
                    entities.append(func_entity)
                    
                elif isinstance(node, ast.ClassDef):
                    class_entity = Entity(
                        id=f"class_{module_name}_{node.name}",
                        type="class",
                        name=node.name,
                        properties={
                            "module": module_name,
                            "line_number": node.lineno,
                            "bases": [base.id for base in node.bases if isinstance(base, ast.Name)],
                            "docstring": ast.get_docstring(node),
                            "methods": [n.name for n in node.body if isinstance(n, ast.FunctionDef)],
                            "decorators": [d.id for d in node.decorator_list if isinstance(d, ast.Name)]
                        }
                    )
                    entities.append(class_entity)
            
        except Exception as e:
            print(f"Error parsing file {file_path}: {e}")
        
        return entities
    
    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """Extract import statements from AST"""
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend([alias.name for alias in node.names])
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                imports.extend([f"{module}.{alias.name}" for alias in node.names])
        return imports
    
    def _calculate_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity"""
        complexity = 1  # Base complexity
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor, 
                               ast.ExceptHandler, ast.With, ast.AsyncWith)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        return complexity
    
    def _extract_return_type(self, node: ast.FunctionDef) -> Optional[str]:
        """Extract return type annotation if present"""
        if node.returns:
            if isinstance(node.returns, ast.Name):
                return node.returns.id
            elif isinstance(node.returns, ast.Constant):
                return str(node.returns.value)
        return None
    
    def find_enhanced_relationships(self, file_path: str) -> List[Relationship]:
        """Find enhanced relationships including dependencies, calls, and patterns"""
        relationships = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            module_name = os.path.basename(file_path).replace('.py', '')
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    current_func_id = f"function_{module_name}_{node.name}"
                    
                    # Find function calls
                    for child in ast.walk(node):
                        if isinstance(child, ast.Call):
                            if isinstance(child.func, ast.Name):
                                called_func_name = child.func.id
                                called_func_id = f"function_{module_name}_{called_func_name}"
                                
                                relationship = Relationship(
                                    source_id=current_func_id,
                                    target_id=called_func_id,
                                    relationship_type="calls",
                                    properties={"line_number": child.lineno, "call_type": "direct"}
                                )
                                relationships.append(relationship)
                            
                            elif isinstance(child.func, ast.Attribute):
                                # Method calls
                                if isinstance(child.func.value, ast.Name):
                                    obj_name = child.func.value.id
                                    method_name = child.func.attr
                                    
                                    relationship = Relationship(
                                        source_id=current_func_id,
                                        target_id=f"method_{obj_name}_{method_name}",
                                        relationship_type="calls",
                                        properties={"line_number": child.lineno, "call_type": "method"}
                                    )
                                    relationships.append(relationship)
                    
                    # Find exception handling patterns
                    for child in ast.walk(node):
                        if isinstance(child, ast.ExceptHandler):
                            if child.type and isinstance(child.type, ast.Name):
                                exception_type = child.type.id
                                relationship = Relationship(
                                    source_id=current_func_id,
                                    target_id=f"exception_{exception_type}",
                                    relationship_type="handles",
                                    properties={"line_number": child.lineno}
                                )
                                relationships.append(relationship)
            
        except Exception as e:
            print(f"Error finding relationships in {file_path}: {e}")
        
        return relationships
    
    def ingest_codebase(self, directory_path: str):
        """Enhanced codebase ingestion with progress tracking"""
        print(f"Ingesting codebase from {directory_path}...")
        
        total_files = sum(1 for root, dirs, files in os.walk(directory_path) 
                         for file in files if file.endswith('.py'))
        processed_files = 0
        
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    print(f"Processing {file_path} ({processed_files + 1}/{total_files})...")
                    
                    # Extract entities
                    entities = self.parse_python_file(file_path)
                    for entity in entities:
                        self.add_entity(entity)
                    
                    # Extract relationships
                    relationships = self.find_enhanced_relationships(file_path)
                    for relationship in relationships:
                        self.add_relationship(relationship)
                    
                    processed_files += 1
        
        print(f"Ingestion complete. Processed {processed_files} files.")
    
    def find_similar_entities(self, entity_type: str, query: str, limit: int = 3) -> List[Entity]:
        """Enhanced similarity search with semantic matching"""
        similar_entities = []
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Update access count
        cursor.execute('''
            UPDATE entities SET access_count = access_count + 1 
            WHERE type = ? AND (name LIKE ? OR properties LIKE ?)
        ''', (entity_type, f"%{query}%", f"%{query}%"))
        
        cursor.execute('''
            SELECT id, type, name, properties, created_at, access_count
            FROM entities 
            WHERE type = ? AND (name LIKE ? OR properties LIKE ?)
            ORDER BY access_count DESC, name
            LIMIT ?
        ''', (entity_type, f"%{query}%", f"%{query}%", limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        for row in rows:
            entity = Entity(
                id=row[0],
                type=row[1],
                name=row[2],
                properties=json.loads(row[3]) if row[3] else {},
                created_at=row[4]
            )
            similar_entities.append(entity)
        
        return similar_entities
    
    def get_entity_context(self, entity_id: str) -> Dict[str, Any]:
        """Enhanced context retrieval with expectations and outcomes"""
        context = {
            "entity": None,
            "incoming_relationships": [],
            "outgoing_relationships": [],
            "related_entities": [],
            "outcomes": [],
            "expectations": [],
            "context_data": []
        }
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get the entity
        cursor.execute('SELECT * FROM entities WHERE id = ?', (entity_id,))
        entity_row = cursor.fetchone()
        
        if entity_row:
            context["entity"] = Entity(
                id=entity_row[0],
                type=entity_row[1],
                name=entity_row[2],
                properties=json.loads(entity_row[3]) if entity_row[3] else {},
                created_at=entity_row[4]
            )
        
        # Get relationships with strength
        cursor.execute('''
            SELECT source_id, target_id, relationship_type, properties, strength
            FROM relationships 
            WHERE source_id = ? OR target_id = ?
            ORDER BY strength DESC
        ''', (entity_id, entity_id))
        
        relationships = cursor.fetchall()
        for rel in relationships:
            rel_data = {
                "type": rel[2],
                "properties": json.loads(rel[3]) if rel[3] else {},
                "strength": rel[4]
            }
            
            if rel[0] == entity_id:  # Outgoing
                rel_data["target_id"] = rel[1]
                context["outgoing_relationships"].append(rel_data)
            else:  # Incoming
                rel_data["source_id"] = rel[0]
                context["incoming_relationships"].append(rel_data)
        
        # Get outcomes
        cursor.execute('''
            SELECT outcome_type, outcome_data, success, timestamp
            FROM outcomes 
            WHERE entity_id = ?
            ORDER BY timestamp DESC
            LIMIT 10
        ''', (entity_id,))
        
        outcomes = cursor.fetchall()
        for outcome in outcomes:
            context["outcomes"].append({
                "type": outcome[0],
                "data": json.loads(outcome[1]) if outcome[1] else {},
                "success": bool(outcome[2]),
                "timestamp": outcome[3]
            })
        
        # Get expectations
        cursor.execute('''
            SELECT expectation_type, expectation_data, met, timestamp
            FROM expectations 
            WHERE entity_id = ?
            ORDER BY timestamp DESC
            LIMIT 10
        ''', (entity_id,))
        
        expectations = cursor.fetchall()
        for expectation in expectations:
            context["expectations"].append({
                "type": expectation[0],
                "data": json.loads(expectation[1]) if expectation[1] else {},
                "met": bool(expectation[2]),
                "timestamp": expectation[3]
            })
        
        conn.close()
        return context
    
    def record_outcome(self, entity_id: str, outcome_type: str, outcome_data: Dict[str, Any], success: bool = True):
        """Record an outcome for learning purposes"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO outcomes (entity_id, outcome_type, outcome_data, success, timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (entity_id, outcome_type, json.dumps(outcome_data), success, datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
    
    def record_expectation(self, entity_id: str, expectation_type: str, expectation_data: Dict[str, Any], met: bool = False):
        """Record an expectation for tracking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO expectations (entity_id, expectation_type, expectation_data, met, timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (entity_id, expectation_type, json.dumps(expectation_data), met, datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights from outcomes and expectations for continuous learning"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Success rates by entity type
        cursor.execute('''
            SELECT e.type, 
                   COUNT(o.id) as total_outcomes,
                   SUM(CASE WHEN o.success = 1 THEN 1 ELSE 0 END) as successful_outcomes
            FROM entities e
            LEFT JOIN outcomes o ON e.id = o.entity_id
            WHERE o.id IS NOT NULL
            GROUP BY e.type
        ''')
        
        success_rates = {}
        for row in cursor.fetchall():
            entity_type = row[0]
            total = row[1]
            successful = row[2]
            success_rates[entity_type] = {
                "total_outcomes": total,
                "successful_outcomes": successful,
                "success_rate": successful / total if total > 0 else 0
            }
        
        # Expectation fulfillment rates
        cursor.execute('''
            SELECT expectation_type,
                   COUNT(*) as total_expectations,
                   SUM(CASE WHEN met = 1 THEN 1 ELSE 0 END) as met_expectations
            FROM expectations
            GROUP BY expectation_type
        ''')
        
        expectation_rates = {}
        for row in cursor.fetchall():
            exp_type = row[0]
            total = row[1]
            met = row[2]
            expectation_rates[exp_type] = {
                "total_expectations": total,
                "met_expectations": met,
                "fulfillment_rate": met / total if total > 0 else 0
            }
        
        conn.close()
        
        return {
            "success_rates": success_rates,
            "expectation_fulfillment": expectation_rates,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Enhanced graph statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM entities')
        entity_count = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM relationships')
        relationship_count = cursor.fetchone()[0]
        
        cursor.execute('SELECT type, COUNT(*) FROM entities GROUP BY type')
        entity_types = dict(cursor.fetchall())
        
        cursor.execute('SELECT relationship_type, COUNT(*) FROM relationships GROUP BY relationship_type')
        relationship_types = dict(cursor.fetchall())
        
        cursor.execute('SELECT COUNT(*) FROM outcomes')
        outcome_count = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM expectations')
        expectation_count = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "total_entities": entity_count,
            "total_relationships": relationship_count,
            "total_outcomes": outcome_count,
            "total_expectations": expectation_count,
            "entity_types": entity_types,
            "relationship_types": relationship_types,
            "graph_density": nx.density(self.graph) if self.graph.number_of_nodes() > 0 else 0,
            "connected_components": nx.number_weakly_connected_components(self.graph)
        }

# Example usage and testing
if __name__ == "__main__":
    kg = KnowledgeGraph()
    
    # Ingest the unified orchestrator codebase
    kg.ingest_codebase("/home/ubuntu/unified_orchestrator")
    
    # Print enhanced statistics
    stats = kg.get_graph_stats()
    print("Enhanced Knowledge Graph Statistics:")
    print(json.dumps(stats, indent=2))
    
    # Get learning insights
    insights = kg.get_learning_insights()
    print("\nLearning Insights:")
    print(json.dumps(insights, indent=2))
    
    # Find similar functions
    similar_funcs = kg.find_similar_entities("function", "init", limit=3)
    print(f"\nFound {len(similar_funcs)} similar functions:")
    for func in similar_funcs:
        print(f"- {func.name} ({func.id})")

