#!/usr/bin/env python3
"""
Advanced Python Code Agent v2.0 - AI-Powered Code Generation & Optimization
Features machine learning-based code suggestions, advanced static analysis, and intelligent execution
"""

import ast
import sys
import io
import traceback
import subprocess
import tempfile
import os
import inspect
import types
import time
import hashlib
import pickle
import threading
import queue
import multiprocessing
import sqlite3
import logging
from contextual import redirect_stdout, redirect_stderr
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Set, Iterator
from dataclasses import dataclass, field, asdict
from collections import defaultdict, Counter, deque
from functools import lru_cache, wraps, partial
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path
import json
import re
import difflib
import tokenize
from enum import Enum, auto
import heapq
import networkx as nx
import warnings
from abc import ABC, abstractmethod
import weakref
import gc
import psutil
import resource

# Advanced imports for ML-based features
try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CodeComplexity(Enum):
    """Enhanced code complexity levels with numerical values"""
    TRIVIAL = (1, "Very simple, single operation")
    SIMPLE = (2, "Basic logic, few branches")
    MODERATE = (3, "Multiple functions, some complexity")
    COMPLEX = (4, "Advanced algorithms, many branches")
    EXPERT = (5, "Highly complex, requires deep expertise")
    CRITICAL = (6, "System-critical, maximum complexity")

class OptimizationStrategy(Enum):
    """Code optimization strategies"""
    PERFORMANCE = auto()
    MEMORY = auto()
    READABILITY = auto()
    SECURITY = auto()
    MAINTAINABILITY = auto()
    ALL = auto()

class ExecutionMode(Enum):
    """Code execution modes"""
    SAFE = auto()      # Sandboxed execution
    NORMAL = auto()    # Standard execution
    PARALLEL = auto()  # Multi-threaded execution
    DISTRIBUTED = auto() # Multi-process execution

@dataclass
class PerformanceMetrics:
    """Detailed performance metrics"""
    execution_time: float = 0.0
    memory_usage: int = 0
    cpu_usage: float = 0.0
    io_operations: int = 0
    network_calls: int = 0
    database_queries: int = 0
    cache_hits: int = 0
    cache_misses: int = 0

@dataclass
class SecurityAnalysis:
    """Security analysis results"""
    vulnerability_count: int = 0
    risk_level: str = "LOW"
    vulnerabilities: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    secure_alternatives: Dict[str, str] = field(default_factory=dict)

@dataclass
class CodeAnalysis:
    """Comprehensive code analysis results"""
    complexity: CodeComplexity
    cyclomatic_complexity: int
    lines_of_code: int
    maintainability_index: float
    security_analysis: SecurityAnalysis
    performance_metrics: PerformanceMetrics
    suggestions: List[str] = field(default_factory=list)
    code_smells: List[str] = field(default_factory=list)
    refactoring_opportunities: List[str] = field(default_factory=list)
    test_coverage_estimate: float = 0.0
    documentation_score: float = 0.0

@dataclass
class ExecutionContext:
    """Enhanced execution context with dependency tracking"""
    variables: Dict[str, Any] = field(default_factory=dict)
    imports: Set[str] = field(default_factory=set)
    functions: Set[str] = field(default_factory=set)
    classes: Set[str] = field(default_factory=set)
    dependencies: Set[str] = field(default_factory=set)
    resource_usage: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    execution_mode: ExecutionMode = ExecutionMode.NORMAL
    timeout: Optional[float] = None
    memory_limit: Optional[int] = None

class CodeDatabase:
    """Persistent storage for code analysis and learning"""
    
    def __init__(self, db_path: str = "code_agent.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript('''
                CREATE TABLE IF NOT EXISTS code_executions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    code_hash TEXT UNIQUE,
                    code TEXT,
                    execution_count INTEGER DEFAULT 1,
                    success_rate REAL DEFAULT 1.0,
                    avg_execution_time REAL DEFAULT 0.0,
                    complexity_score INTEGER DEFAULT 0,
                    tags TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS optimization_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_name TEXT,
                    original_pattern TEXT,
                    optimized_pattern TEXT,
                    improvement_factor REAL,
                    usage_count INTEGER DEFAULT 0,
                    success_rate REAL DEFAULT 1.0
                );
                
                CREATE TABLE IF NOT EXISTS code_suggestions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    context_hash TEXT,
                    suggestion TEXT,
                    confidence REAL,
                    usage_count INTEGER DEFAULT 0,
                    success_rate REAL DEFAULT 1.0
                );
                
                CREATE INDEX IF NOT EXISTS idx_code_hash ON code_executions(code_hash);
                CREATE INDEX IF NOT EXISTS idx_context_hash ON code_suggestions(context_hash);
            ''')
    
    def record_execution(self, code: str, success: bool, execution_time: float, complexity: int):
        """Record code execution for learning"""
        code_hash = hashlib.sha256(code.encode()).hexdigest()
        
        with sqlite3.connect(self.db_path) as conn:
            # Update or insert execution record
            conn.execute('''
                INSERT OR REPLACE INTO code_executions 
                (code_hash, code, execution_count, success_rate, avg_execution_time, complexity_score, last_used)
                VALUES (?, ?, 
                    COALESCE((SELECT execution_count FROM code_executions WHERE code_hash = ?) + 1, 1),
                    COALESCE(
                        (SELECT (success_rate * execution_count + ?) / (execution_count + 1) 
                         FROM code_executions WHERE code_hash = ?), ?
                    ),
                    COALESCE(
                        (SELECT (avg_execution_time * execution_count + ?) / (execution_count + 1)
                         FROM code_executions WHERE code_hash = ?), ?
                    ),
                    ?, CURRENT_TIMESTAMP)
            ''', (code_hash, code, code_hash, 1 if success else 0, code_hash, 1 if success else 0,
                  execution_time, code_hash, execution_time, complexity))
    
    def get_similar_code(self, code: str, limit: int = 5) -> List[Dict]:
        """Find similar code patterns"""
        if not ML_AVAILABLE:
            return []
        
        code_hash = hashlib.sha256(code.encode()).hexdigest()
        
        with sqlite3.connect(self.db_path) as conn:
            # Get all code samples
            cursor = conn.execute('SELECT code, success_rate, avg_execution_time FROM code_executions')
            samples = cursor.fetchall()
            
            if not samples:
                return []
            
            # Use TF-IDF for similarity
            vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
            codes = [sample[0] for sample in samples] + [code]
            
            try:
                tfidf_matrix = vectorizer.fit_transform(codes)
                similarities = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1]).flatten()
                
                # Get top similar codes
                top_indices = similarities.argsort()[-limit:][::-1]
                
                return [{
                    'code': samples[i][0],
                    'similarity': similarities[i],
                    'success_rate': samples[i][1],
                    'avg_execution_time': samples[i][2]
                } for i in top_indices if similarities[i] > 0.1]
            except Exception as e:
                logger.warning(f"Similarity calculation failed: {e}")
                return []

class SmartCodeGenerator:
    """AI-powered code generation with learning capabilities"""
    
    def __init__(self):
        self.db = CodeDatabase()
        self.patterns = self._load_enhanced_patterns()
        self.algorithms = self._load_advanced_algorithms()
        self.ml_model = self._init_ml_model() if ML_AVAILABLE else None
    
    def _init_ml_model(self):
        """Initialize machine learning model for code suggestions"""
        # In a real implementation, this would load a pre-trained model
        return {
            'vectorizer': TfidfVectorizer(max_features=10000, ngram_range=(1, 3)),
            'patterns': defaultdict(list),
            'confidence_threshold': 0.7
        }
    
    def generate_intelligent_code(self, description: str, context: ExecutionContext, 
                                strategy: OptimizationStrategy = OptimizationStrategy.ALL) -> str:
        """Generate code using AI and historical patterns"""
        # Analyze description for intent
        intent = self._analyze_intent(description)
        
        # Get similar historical patterns
        similar_patterns = self._get_similar_patterns(description, intent)
        
        # Generate base code
        base_code = self._generate_base_code(description, intent, context)
        
        # Apply optimization strategy
        optimized_code = self._apply_strategy_optimization(base_code, strategy, context)
        
        # Add error handling and logging
        robust_code = self._add_robustness(optimized_code, context)
        
        return robust_code
    
    def _analyze_intent(self, description: str) -> Dict[str, Any]:
        """Analyze user intent from description"""
        intents = {
            'data_processing': ['process', 'analyze', 'filter', 'transform', 'clean'],
            'algorithm': ['sort', 'search', 'optimize', 'calculate', 'compute'],
            'web': ['request', 'scrape', 'api', 'http', 'download'],
            'file_ops': ['read', 'write', 'file', 'directory', 'path'],
            'database': ['query', 'insert', 'update', 'delete', 'database', 'sql'],
            'automation': ['automate', 'schedule', 'task', 'job', 'cron'],
            'visualization': ['plot', 'chart', 'graph', 'visualize', 'display'],
            'machine_learning': ['predict', 'train', 'model', 'classify', 'cluster']
        }
        
        description_lower = description.lower()
        detected_intents = {}
        
        for intent, keywords in intents.items():
            score = sum(1 for keyword in keywords if keyword in description_lower)
            if score > 0:
                detected_intents[intent] = score / len(keywords)
        
        primary_intent = max(detected_intents, key=detected_intents.get) if detected_intents else 'general'
        
        return {
            'primary': primary_intent,
            'secondary': [k for k, v in detected_intents.items() if v > 0.2 and k != primary_intent],
            'confidence': detected_intents.get(primary_intent, 0.1),
            'complexity_estimate': min(len(description.split()) / 10, 5)
        }
    
    def _get_similar_patterns(self, description: str, intent: Dict) -> List[str]:
        """Get similar code patterns from database"""
        return self.db.get_similar_code(description, limit=3)
    
    def _generate_base_code(self, description: str, intent: Dict, context: ExecutionContext) -> str:
        """Generate base code structure"""
        primary_intent = intent['primary']
        
        if primary_intent in self.patterns:
            template = self.patterns[primary_intent]
            return self._customize_template(template, description, context)
        
        # Fallback to general template
        return self._generate_general_template(description, intent, context)
    
    def _customize_template(self, template: str, description: str, context: ExecutionContext) -> str:
        """Customize template based on description and context"""
        # Extract parameters from description
        params = self._extract_parameters(description)
        
        # Replace placeholders in template
        customized = template
        for param, value in params.items():
            customized = customized.replace(f"{{{{ {param} }}}}", str(value))
        
        return customized
    
    def _extract_parameters(self, description: str) -> Dict[str, str]:
        """Extract parameters from natural language description"""
        params = {}
        
        # Extract numbers
        numbers = re.findall(r'\b\d+\b', description)
        if numbers:
            params['number'] = numbers[0]
            params['count'] = numbers[0]
        
        # Extract file extensions
        extensions = re.findall(r'\.([a-zA-Z0-9]+)', description)
        if extensions:
            params['file_extension'] = extensions[0]
        
        # Extract URLs
        urls = re.findall(r'https?://[^\s]+', description)
        if urls:
            params['url'] = urls[0]
        
        # Extract file paths
        paths = re.findall(r'[/\\][\w/\\.-]+', description)
        if paths:
            params['file_path'] = paths[0]
        
        return params
    
    def _apply_strategy_optimization(self, code: str, strategy: OptimizationStrategy, 
                                   context: ExecutionContext) -> str:
        """Apply optimization strategy to code"""
        if strategy == OptimizationStrategy.PERFORMANCE:
            return self._optimize_for_performance(code)
        elif strategy == OptimizationStrategy.MEMORY:
            return self._optimize_for_memory(code)
        elif strategy == OptimizationStrategy.SECURITY:
            return self._optimize_for_security(code)
        elif strategy == OptimizationStrategy.READABILITY:
            return self._optimize_for_readability(code)
        elif strategy == OptimizationStrategy.MAINTAINABILITY:
            return self._optimize_for_maintainability(code)
        else:
            # Apply all optimizations
            optimized = self._optimize_for_performance(code)
            optimized = self._optimize_for_memory(optimized)
            optimized = self._optimize_for_security(optimized)
            return optimized
    
    def _optimize_for_performance(self, code: str) -> str:
        """Optimize code for performance"""
        optimizations = [
            # Use list comprehensions
            (r'result = \[\]\nfor (\w+) in (.+?):\n    if (.+?):\n        result\.append\((.+?)\)',
             r'result = [\4 for \1 in \2 if \3]'),
            
            # Use generator expressions for memory efficiency
            (r'sum\(\[(.*?) for (.*?) in (.*?)\]\)',
             r'sum(\1 for \2 in \3)'),
            
            # Use dict.get() instead of key checking
            (r'if (\w+) in (\w+):\n    (.+?) = \2\[\1\]\nelse:\n    \3 = (.+?))',
             r'\3 = \2.get(\1, \4)'),
            
            # Use enumerate instead of range(len())
            (r'for i in range\(len\((.+?)\)\):\n    (.+?) = \1\[i\]',
             r'for i, \2 in enumerate(\1):'),
        ]
        
        optimized_code = code
        for pattern, replacement in optimizations:
            optimized_code = re.sub(pattern, replacement, optimized_code, flags=re.MULTILINE)
        
        return optimized_code
    
    def _optimize_for_memory(self, code: str) -> str:
        """Optimize code for memory usage"""
        # Replace large data structures with generators where appropriate
        optimizations = [
            (r'result = \[(.*?) for (.*?) in (.*?)\]\nfor (.+?) in result:',
             r'# Using generator for memory efficiency\nresult = (\1 for \2 in \3)\nfor \4 in result:'),
            
            (r'data = open\((.+?)\)\.read\(\)',
             r'# Using context manager for better memory management\nwith open(\1) as f:\n    data = f.read()'),
        ]
        
        optimized_code = code
        for pattern, replacement in optimizations:
            optimized_code = re.sub(pattern, replacement, optimized_code, flags=re.MULTILINE)
        
        return optimized_code
    
    def _optimize_for_security(self, code: str) -> str:
        """Optimize code for security"""
        security_fixes = [
            # Replace eval with safer alternatives
            (r'eval\((.+?)\)', r'# Security: Avoid eval(), use ast.literal_eval() for literals\n# ast.literal_eval(\1)'),
            
            # Add input validation
            (r'input\((.+?)\)', r'# Security: Validate input\nuser_input = input(\1)\nif not user_input.replace(" ", "").replace("-", "").isalnum():\n    raise ValueError("Invalid input")\nuser_input'),
            
            # Secure file operations
            (r'open\((.+?), ["\']w["\']\)', r'# Security: Validate file path\nfrom pathlib import Path\nfile_path = Path(\1)\nif not file_path.parent.exists():\n    file_path.parent.mkdir(parents=True)\nopen(file_path, "w")'),
        ]
        
        secure_code = code
        for pattern, replacement in security_fixes:
            secure_code = re.sub(pattern, replacement, secure_code, flags=re.MULTILINE)
        
        return secure_code
    
    def _optimize_for_readability(self, code: str) -> str:
        """Optimize code for readability"""
        # Add type hints and docstrings
        if 'def ' in code and '"""' not in code:
            # Add basic docstring template
            code = re.sub(r'def (\w+)\((.*?)\):', 
                         r'def \1(\2):\n    """\n    Function: \1\n    """', code)
        
        return code
    
    def _optimize_for_maintainability(self, code: str) -> str:
        """Optimize code for maintainability"""
        # Break long functions into smaller ones
        # Add configuration constants
        # Add proper error handling
        maintainable_code = code
        
        # Add constants for magic numbers
        numbers = re.findall(r'\b(\d{2,})\b', code)
        if numbers:
            constants = []
            for i, num in enumerate(set(numbers)):
                const_name = f'CONSTANT_{i}'
                constants.append(f'{const_name} = {num}')
                maintainable_code = maintainable_code.replace(num, const_name)
            
            if constants:
                maintainable_code = '# Configuration constants\n' + '\n'.join(constants) + '\n\n' + maintainable_code
        
        return maintainable_code
    
    def _add_robustness(self, code: str, context: ExecutionContext) -> str:
        """Add error handling and logging to code"""
        robust_code = f'''
import logging
import traceback
from typing import Optional, Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def execute_safely() -> Optional[Any]:
    """Execute code with proper error handling"""
    try:
        logger.info("Starting execution...")
        
        # Generated code begins here
{self._indent_code(code, 8)}
        
        logger.info("Execution completed successfully")
        return locals().get('result', None)
        
    except Exception as e:
        logger.error(f"Execution failed: {{e}}")
        logger.error(f"Traceback: {{traceback.format_exc()}}")
        raise
    finally:
        logger.info("Execution finished")

# Execute the function
if __name__ == "__main__":
    result = execute_safely()
    if result is not None:
        print(f"Result: {{result}}")
'''
        return robust_code
    
    def _indent_code(self, code: str, spaces: int) -> str:
        """Indent code by specified number of spaces"""
        indent = ' ' * spaces
        return '\n'.join(indent + line for line in code.split('\n'))
    
    def _generate_general_template(self, description: str, intent: Dict, context: ExecutionContext) -> str:
        """Generate a general purpose template"""
        return f'''
def generated_function():
    """
    Generated function based on: {description}
    Intent: {intent['primary']} (confidence: {intent['confidence']:.2f})
    """
    # TODO: Implement functionality based on description
    # Primary intent: {intent['primary']}
    # Secondary intents: {', '.join(intent['secondary'])}
    
    result = None
    
    # Add your implementation here
    
    return result

# Execute the function
result = generated_function()
print(f"Result: {{result}}")
'''
    
    def _load_enhanced_patterns(self) -> Dict[str, str]:
        """Load enhanced code patterns with ML integration"""
        return {
            'data_processing': '''
import pandas as pd
import numpy as np
from typing import Union, List, Optional
import logging

logger = logging.getLogger(__name__)

def process_data(data_source: Union[str, pd.DataFrame], 
                operations: List[str] = None) -> pd.DataFrame:
    """
    Advanced data processing with automatic optimization
    """
    logger.info(f"Processing data from: {data_source}")
    
    # Load data intelligently
    if isinstance(data_source, str):
        if data_source.endswith('.csv'):
            df = pd.read_csv(data_source, 
                           dtype_backend='nullable_int64',  # Use nullable dtypes
                           engine='pyarrow' if 'pyarrow' in sys.modules else 'python')
        elif data_source.endswith('.parquet'):
            df = pd.read_parquet(data_source)
        elif data_source.endswith('.json'):
            df = pd.read_json(data_source)
        else:
            df = pd.read_excel(data_source)
    else:
        df = data_source.copy()
    
    logger.info(f"Loaded data shape: {df.shape}")
    
    # Automatic data profiling
    profile = {
        'shape': df.shape,
        'memory_usage': df.memory_usage(deep=True).sum(),
        'null_counts': df.isnull().sum().to_dict(),
        'dtypes': df.dtypes.to_dict(),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist()
    }
    
    logger.info(f"Data profile: {profile}")
    
    # Apply operations if specified
    if operations:
        for operation in operations:
            if operation == 'clean':
                df = df.dropna()
            elif operation == 'normalize':
                numeric_columns = df.select_dtypes(include=[np.number]).columns
                df[numeric_columns] = (df[numeric_columns] - df[numeric_columns].mean()) / df[numeric_columns].std()
            elif operation == 'deduplicate':
                df = df.drop_duplicates()
    
    return df
''',
            
            'algorithm': '''
from typing import List, Tuple, Optional, TypeVar, Generic, Protocol
from abc import ABC, abstractmethod
import heapq
import bisect
from collections import defaultdict, deque
import time

T = TypeVar('T')

class Comparable(Protocol):
    def __lt__(self, other: 'Comparable') -> bool: ...

class Algorithm(ABC, Generic[T]):
    """Base class for algorithms with performance tracking"""
    
    def __init__(self):
        self.execution_time = 0.0
        self.comparisons = 0
        self.memory_usage = 0
    
    @abstractmethod
    def execute(self, data: List[T]) -> List[T]:
        """Execute the algorithm"""
        pass
    
    def benchmark(self, data: List[T], iterations: int = 1) -> dict:
        """Benchmark algorithm performance"""
        times = []
        for _ in range(iterations):
            start_time = time.perf_counter()
            result = self.execute(data)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        return {
            'avg_time': sum(times) / len(times),
            'min_time': min(times),
            'max_time': max(times),
            'comparisons': self.comparisons,
            'result_size': len(result) if result else 0
        }

class AdaptiveSort(Algorithm[T]):
    """Adaptive sorting algorithm that chooses best strategy"""
    
    def execute(self, data: List[T]) -> List[T]:
        n = len(data)
        
        if n <= 1:
            return data.copy()
        elif n <= 10:
            return self._insertion_sort(data)
        elif self._is_nearly_sorted(data):
            return self._tim_sort(data)
        elif n <= 1000:
            return self._quick_sort(data)
        else:
            return self._merge_sort(data)
    
    def _is_nearly_sorted(self, data: List[T]) -> bool:
        """Check if data is nearly sorted"""
        inversions = 0
        for i in range(len(data) - 1):
            if data[i] > data[i + 1]:
                inversions += 1
                if inversions > len(data) * 0.1:  # More than 10% inversions
                    return False
        return True
    
    def _insertion_sort(self, data: List[T]) -> List[T]:
        """Insertion sort for small arrays"""
        result = data.copy()
        for i in range(1, len(result)):
            key = result[i]
            j = i - 1
            while j >= 0 and result[j] > key:
                self.comparisons += 1
                result[j + 1] = result[j]
                j -= 1
            result[j + 1] = key
        return result
    
    def _tim_sort(self, data: List[T]) -> List[T]:
        """Use Python's built-in Timsort for nearly sorted data"""
        return sorted(data)
    
    def _quick_sort(self, data: List[T]) -> List[T]:
        """Optimized quicksort with median-of-three pivot"""
        if len(data) <= 1:
            return data.copy()
        
        # Median-of-three pivot selection
        first, middle, last = 0, len(data) // 2, len(data) - 1
        if data[middle] < data[first]:
            data[first], data[middle] = data[middle], data[first]
        if data[last] < data[first]:
            data[first], data[last] = data[last], data[first]
        if data[last] < data[middle]:
            data[middle], data[last] = data[last], data[middle]
        
        pivot = data[middle]
        left = [x for x in data if x < pivot]
        middle_vals = [x for x in data if x == pivot]
        right = [x for x in data if x > pivot]
        
        return self._quick_sort(left) + middle_vals + self._quick_sort(right)
    
    def _merge_sort(self, data: List[T]) -> List[T]:
        """Merge sort for large arrays"""
        if len(data) <= 1:
            return data.copy()
        
        mid = len(data) // 2
        left = self._merge_sort(data[:mid])
        right = self._merge_sort(data[mid:])
        
        return self._merge(left, right)
    
    def _merge(self, left: List[T], right: List[T]) -> List[T]:
        """Merge two sorted arrays"""
        result = []
        i = j = 0
        
        while i < len(left) and j < len(right):
            self.comparisons += 1
            if left[i] <= right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        
        result.extend(left[i:])
        result.extend(right[j:])
        return result
''',
            
            'machine_learning': '''
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import joblib
from typing import Tuple, Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

class AutoML:
    """Automated machine learning pipeline"""
    
    def __init__(self, problem_type: str = 'classification'):
        self.problem_type = problem_type
        self.models = {}
        self.best_model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
    
    def prepare_data(self, X: pd.DataFrame, y: pd.Series, 
                    test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data with automatic preprocessing"""
        logger.info("Preparing data...")
        
        # Handle categorical variables
        X_processed = X.copy()
        for column in X_processed.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X_processed[column] = le.fit_transform(X_processed[column].astype(str))
        
        # Handle missing values
        X_processed = X_processed.fillna(X_processed.mean() if X_processed.select_dtypes(include=[np.number]).shape[1] > 0 else 0)
        
        # Encode target variable if needed
        if y.dtype == 'object':
            y_processed = self.label_encoder.fit_transform(y)
        else:
            y_processed = y.values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_processed, test_size=test_size, random_state=42, stratify=y_processed
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info(f"Data prepared: Train shape {X_train_scaled.shape}, Test shape {X_test_scaled.shape}")
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Dict]:
        """Train multiple models and compare performance"""
        logger.info("Training multiple models...")
        
        models_config = {
            'random_forest': {
                'model': RandomForestClassifier(n_estimators=100, random_state=42),
                'params': {'n_estimators': [50, 100, 200], 'max_depth': [10, 20, None]}
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {'n_estimators': [50, 100], 'learning_rate': [0.1, 0.01]}
            },
            'logistic_regression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {'C': [0.1, 1, 10], 'penalty': ['l2']}
            },
            'svm': {
                'model': SVC(random_state=42),
                'params': {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']}
            }
        }
        
        results = {}
        
        for name, config in models_config.items():
            logger.info(f"Training {name}...")
            
            # Grid search with cross-validation
            grid_search = GridSearchCV(
                config['model'], 
                config['params'], 
                cv=5, 
                scoring='accuracy',
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            
            # Cross-validation scores
            cv_scores = cross_val_score(grid_search.best_estimator_, X_train, y_train, cv=5)
            
            results[name] = {
                'model': grid_search.best_estimator_,
                'best_params': grid_search.best_params_,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'best_score': grid_search.best_score_
            }
            
            logger.info(f"{name} - CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Select best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['cv_mean'])
        self.best_model = results[best_model_name]['model']
        self.models = results
        self.is_fitted = True
        
        logger.info(f"Best model: {best_model_name}")
        return results
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate the best model"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        y_pred = self.best_model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        cm = confusion_matrix(y_test, y_pred)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm.tolist(),
            'predictions': y_pred.tolist()
        }
        
        logger.info(f"Model evaluation - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        return results
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        model_data = {
            'model': self.best_model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'models': self.models
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")

# Example usage
def auto_ml_pipeline(data_path: str, target_column: str):
    """Complete AutoML pipeline"""
    # Load data
    df = pd.read_csv(data_path)
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Initialize AutoML
    automl = AutoML()
    
    # Prepare data
    X_train, X_test, y_train, y_test = automl.prepare_data(X, y)
    
    # Train models
    results = automl.train_models(X_train, y_train)
    
    # Evaluate
    evaluation = automl.evaluate(X_test, y_test)
    
    # Save model
    automl.save_model('best_model.joblib')
    
    return {
        'training_results': results,
        'evaluation': evaluation,
        'model': automl
    }
'''
        }
    
    def _load_advanced_algorithms(self) -> Dict[str, Callable]:
        """Load advanced algorithm implementations"""
        algorithms = {}
        
        # Graph algorithms
        algorithms.update({
            'dijkstra': self._dijkstra_implementation,
            'a_star': self._a_star_implementation,
            'floyd_warshall': self._floyd_warshall_implementation,
            'topological_sort': self._topological_sort_implementation,
            'strongly_connected_components': self._scc_implementation,
        })
        
        # String algorithms
        algorithms.update({
            'kmp_search': self._kmp_implementation,
            'boyer_moore': self._boyer_moore_implementation,
            'rabin_karp': self._rabin_karp_implementation,
            'suffix_array': self._suffix_array_implementation,
        })
        
        # Advanced data structures
        algorithms.update({
            'segment_tree': self._segment_tree_implementation,
            'fenwick_tree': self._fenwick_tree_implementation,
            'trie': self._trie_implementation,
            'bloom_filter': self._bloom_filter_implementation,
            'lru_cache': self._lru_cache_implementation,
            'consistent_hashing': self._consistent_hashing_implementation,
        })
        
        # Mathematical algorithms
        algorithms.update({
            'fft': self._fft_implementation,
            'matrix_multiplication': self._matrix_mult_implementation,
            'prime_sieve': self._sieve_implementation,
            'number_theory': self._number_theory_implementation,
        })
        
        return algorithms
    
    def _dijkstra_implementation(self) -> str:
        return '''
import heapq
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

class Graph:
    """Advanced graph implementation with multiple algorithms"""
    
    def __init__(self, directed: bool = False):
        self.graph = defaultdict(list)
        self.directed = directed
        self.vertices = set()
    
    def add_edge(self, u: int, v: int, weight: float = 1.0):
        """Add edge to graph"""
        self.graph[u].append((v, weight))
        self.vertices.update([u, v])
        
        if not self.directed:
            self.graph[v].append((u, weight))
    
    def dijkstra(self, start: int, end: Optional[int] = None) -> Dict[int, Tuple[float, List[int]]]:
        """Dijkstra's shortest path with path reconstruction"""
        distances = {vertex: float('infinity') for vertex in self.vertices}
        distances[start] = 0
        previous = {vertex: None for vertex in self.vertices}
        pq = [(0, start)]
        visited = set()
        
        while pq:
            current_distance, current = heapq.heappop(pq)
            
            if current in visited:
                continue
                
            visited.add(current)
            
            if end and current == end:
                break
            
            for neighbor, weight in self.graph[current]:
                distance = current_distance + weight
                
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous[neighbor] = current
                    heapq.heappush(pq, (distance, neighbor))
        
        # Reconstruct paths
        paths = {}
        for vertex in self.vertices:
            if distances[vertex] != float('infinity'):
                path = []
                current = vertex
                while current is not None:
                    path.append(current)
                    current = previous[current]
                path.reverse()
                paths[vertex] = (distances[vertex], path)
        
        return paths
    
    def a_star(self, start: int, goal: int, heuristic_func) -> Tuple[float, List[int]]:
        """A* pathfinding algorithm"""
        open_set = [(0, start)]
        came_from = {}
        g_score = {vertex: float('infinity') for vertex in self.vertices}
        g_score[start] = 0
        f_score = {vertex: float('infinity') for vertex in self.vertices}
        f_score[start] = heuristic_func(start, goal)
        
        while open_set:
            current_f, current = heapq.heappop(open_set)
            
            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return g_score[goal], path
            
            for neighbor, weight in self.graph[current]:
                tentative_g = g_score[current] + weight
                
                if tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic_func(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return float('infinity'), []
    
    def floyd_warshall(self) -> Dict[Tuple[int, int], float]:
        """All-pairs shortest path"""
        vertices = list(self.vertices)
        n = len(vertices)
        
        # Initialize distance matrix
        dist = {}
        for i in vertices:
            for j in vertices:
                if i == j:
                    dist[(i, j)] = 0
                else:
                    dist[(i, j)] = float('infinity')
        
        # Set direct edge distances
        for u in self.graph:
            for v, weight in self.graph[u]:
                dist[(u, v)] = weight
        
        # Floyd-Warshall algorithm
        for k in vertices:
            for i in vertices:
                for j in vertices:
                    if dist[(i, k)] + dist[(k, j)] < dist[(i, j)]:
                        dist[(i, j)] = dist[(i, k)] + dist[(k, j)]
        
        return dist
    
    def topological_sort(self) -> List[int]:
        """Topological sort using DFS"""
        if not self.directed:
            raise ValueError("Topological sort requires a directed graph")
        
        visited = set()
        temp_mark = set()
        result = []
        
        def visit(node):
            if node in temp_mark:
                raise ValueError("Graph has cycles")
            if node not in visited:
                temp_mark.add(node)
                for neighbor, _ in self.graph[node]:
                    visit(neighbor)
                temp_mark.remove(node)
                visited.add(node)
                result.append(node)
        
        for vertex in self.vertices:
            if vertex not in visited:
                visit(vertex)
        
        return result[::-1]
'''
    
    def _segment_tree_implementation(self) -> str:
        return '''
from typing import List, Callable, Any

class SegmentTree:
    """Segment tree for range queries and updates"""
    
    def __init__(self, arr: List[int], operation: Callable[[int, int], int] = max, identity: int = 0):
        """
        Initialize segment tree
        operation: function for combining values (max, min, sum, etc.)
        identity: identity element for the operation
        """
        self.n = len(arr)
        self.operation = operation
        self.identity = identity
        self.tree = [identity] * (4 * self.n)
        self.lazy = [0] * (4 * self.n)
        
        if arr:
            self._build(arr, 0, 0, self.n - 1)
    
    def _build(self, arr: List[int], node: int, start: int, end: int):
        """Build the segment tree"""
        if start == end:
            self.tree[node] = arr[start]
        else:
            mid = (start + end) // 2
            self._build(arr, 2 * node + 1, start, mid)
            self._build(arr, 2 * node + 2, mid + 1, end)
            self.tree[node] = self.operation(self.tree[2 * node + 1], self.tree[2 * node + 2])
    
    def _push(self, node: int, start: int, end: int):
        """Push lazy propagation"""
        if self.lazy[node] != 0:
            self.tree[node] += self.lazy[node]
            if start != end:  # Not a leaf
                self.lazy[2 * node + 1] += self.lazy[node]
                self.lazy[2 * node + 2] += self.lazy[node]
            self.lazy[node] = 0
    
    def update_range(self, l: int, r: int, value: int):
        """Update range [l, r] with value"""
        self._update_range(0, 0, self.n - 1, l, r, value)
    
    def _update_range(self, node: int, start: int, end: int, l: int, r: int, value: int):
        """Internal range update"""
        self._push(node, start, end)
        
        if start > end or start > r or end < l:
            return
        
        if start >= l and end <= r:
            self.lazy[node] += value
            self._push(node, start, end)
            return
        
        mid = (start + end) // 2
        self._update_range(2 * node + 1, start, mid, l, r, value)
        self._update_range(2 * node + 2, mid + 1, end, l, r, value)
        
        self._push(2 * node + 1, start, mid)
        self._push(2 * node + 2, mid + 1, end)
        self.tree[node] = self.operation(self.tree[2 * node + 1], self.tree[2 * node + 2])
    
    def query_range(self, l: int, r: int) -> int:
        """Query range [l, r]"""
        return self._query_range(0, 0, self.n - 1, l, r)
    
    def _query_range(self, node: int, start: int, end: int, l: int, r: int) -> int:
        """Internal range query"""
        if start > end or start > r or end < l:
            return self.identity
        
        self._push(node, start, end)
        
        if start >= l and end <= r:
            return self.tree[node]
        
        mid = (start + end) // 2
        left_val = self._query_range(2 * node + 1, start, mid, l, r)
        right_val = self._query_range(2 * node + 2, mid + 1, end, l, r)
        
        return self.operation(left_val, right_val)

class FenwickTree:
    """Fenwick Tree (Binary Indexed Tree) for prefix sum queries"""
    
    def __init__(self, size: int):
        self.size = size
        self.tree = [0] * (size + 1)
    
    def update(self, idx: int, delta: int):
        """Add delta to element at index idx"""
        while idx <= self.size:
            self.tree[idx] += delta
            idx += idx & (-idx)
    
    def query(self, idx: int) -> int:
        """Get prefix sum up to index idx"""
        result = 0
        while idx > 0:
            result += self.tree[idx]
            idx -= idx & (-idx)
        return result
    
    def range_query(self, left: int, right: int) -> int:
        """Get sum of range [left, right]"""
        return self.query(right) - self.query(left - 1)
    
    @classmethod
    def from_array(cls, arr: List[int]) -> 'FenwickTree':
        """Create Fenwick tree from array"""
        ft = cls(len(arr))
        for i, val in enumerate(arr):
            ft.update(i + 1, val)
        return ft
'''

class AdvancedCodeOptimizer(CodeOptimizer):
    """Enhanced code optimizer with ML-based suggestions"""
    
    def __init__(self):
        super().__init__()
        self.db = CodeDatabase()
        self.optimization_history = []
    
    def optimize_with_learning(self, code: str, context: ExecutionContext) -> Tuple[str, List[str], float]:
        """Optimize code using ML-based learning from past optimizations"""
        # Get similar code patterns
        similar_codes = self.db.get_similar_code(code)
        
        # Apply learned optimizations
        optimized_code = code
        applied_optimizations = []
        confidence_score = 0.5  # Base confidence
        
        # Apply historical optimizations with high success rates
        for similar in similar_codes:
            if similar['success_rate'] > 0.8 and similar['similarity'] > 0.6:
                # Apply successful patterns from similar code
                confidence_score += similar['similarity'] * similar['success_rate'] * 0.1
        
        # Standard optimizations
        standard_optimized, standard_opts = self.optimize_code(code, optimization_level=3)
        optimized_code = standard_optimized
        applied_optimizations.extend(standard_opts)
        
        # Advanced pattern-based optimizations
        pattern_optimized, pattern_opts = self._apply_pattern_optimizations(optimized_code, context)
        optimized_code = pattern_optimized
        applied_optimizations.extend(pattern_opts)
        
        # Domain-specific optimizations
        domain_optimized, domain_opts = self._apply_domain_optimizations(optimized_code, context)
        optimized_code = domain_optimized
        applied_optimizations.extend(domain_opts)
        
        return optimized_code, applied_optimizations, min(confidence_score, 1.0)
    
    def _apply_pattern_optimizations(self, code: str, context: ExecutionContext) -> Tuple[str, List[str]]:
        """Apply advanced pattern-based optimizations"""
        optimized = code
        applied = []
        
        patterns = [
            # Vectorization opportunities
            {
                'pattern': r'for i in range\(len\((.+?)\)\):\n\s+(.+?)\[i\] = (.+?)\[i\] ([+\-*/]) (.+)',
                'replacement': r'# Vectorized operation\n\2 = \3 \4 \5',
                'description': 'Vectorize element-wise operations',
                'condition': lambda: 'numpy' in context.imports or 'np' in str(context.variables)
            },
            
            # Memory-efficient generators
            {
                'pattern': r'return \[(.+?) for (.+?) in (.+?) if (.+?)\]',
                'replacement': r'# Memory-efficient generator\nreturn (\1 for \2 in \3 if \4)',
                'description': 'Use generator for memory efficiency',
                'condition': lambda: True
            },
            
            # Database query optimization
            {
                'pattern': r'for (.+?) in (.+?):\n\s+cursor\.execute\((.+?)\)',
                'replacement': r'# Batch database operation\ncursor.executemany(\3, \2)',
                'description': 'Batch database operations',
                'condition': lambda: any('sql' in imp.lower() or 'db' in imp.lower() for imp in context.imports)
            },
            
            # Parallel processing opportunities
            {
                'pattern': r'result = \[\]\nfor (.+?) in (.+?):\n\s+result\.append\((.+?)\((.+?)\)\)',
                'replacement': r'# Parallel processing\nfrom concurrent.futures import ProcessPoolExecutor\nwith ProcessPoolExecutor() as executor:\n    result = list(executor.map(\3, \2))',
                'description': 'Parallelize independent operations',
                'condition': lambda: context.execution_mode == ExecutionMode.PARALLEL
            }
        ]
        
        for pattern_def in patterns:
            if pattern_def['condition']() and re.search(pattern_def['pattern'], optimized, re.MULTILINE):
                optimized = re.sub(pattern_def['pattern'], pattern_def['replacement'], optimized, flags=re.MULTILINE)
                applied.append(f"Applied {pattern_def['description']}")
        
        return optimized, applied
    
    def _apply_domain_optimizations(self, code: str, context: ExecutionContext) -> Tuple[str, List[str]]:
        """Apply domain-specific optimizations based on context"""
        optimized = code
        applied = []
        
        # Data science optimizations
        if any(lib in context.imports for lib in ['pandas', 'numpy', 'sklearn']):
            data_science_opts = [
                (r'df\.iterrows\(\)', 'df.itertuples()', 'Use itertuples() instead of iterrows()'),
                (r'pd\.concat\(\[(.+?)\]\)', r'pd.concat(\1, ignore_index=True)', 'Optimize pandas concat'),
                (r'df\.apply\(lambda x: (.+?)\)', r'# Vectorized operation\ndf.\1', 'Vectorize pandas operations'),
            ]
            
            for pattern, replacement, description in data_science_opts:
                if re.search(pattern, optimized):
                    optimized = re.sub(pattern, replacement, optimized)
                    applied.append(f"Data Science: {description}")
        
        # Web development optimizations
        if any(lib in context.imports for lib in ['requests', 'urllib', 'aiohttp']):
            web_opts = [
                (r'requests\.get\((.+?)\)', r'session.get(\1)', 'Use session for multiple requests'),
                (r'time\.sleep\((.+?)\)', r'await asyncio.sleep(\1)', 'Use async sleep'),
            ]
            
            for pattern, replacement, description in web_opts:
                if re.search(pattern, optimized):
                    optimized = re.sub(pattern, replacement, optimized)
                    applied.append(f"Web: {description}")
        
        return optimized, applied

class ExecutionEngine:
    """Advanced execution engine with multiple modes and monitoring"""
    
    def __init__(self):
        self.db = CodeDatabase()
        self.resource_monitor = ResourceMonitor()
        self.execution_history = []
    
    def execute_with_monitoring(self, code: str, context: ExecutionContext) -> Dict[str, Any]:
        """Execute code with comprehensive monitoring"""
        execution_id = hashlib.md5(f"{code}{time.time()}".encode()).hexdigest()[:8]
        
        # Start resource monitoring
        self.resource_monitor.start_monitoring()
        
        try:
            if context.execution_mode == ExecutionMode.SAFE:
                result = self._execute_sandboxed(code, context)
            elif context.execution_mode == ExecutionMode.PARALLEL:
                result = self._execute_parallel(code, context)
            elif context.execution_mode == ExecutionMode.DISTRIBUTED:
                result = self._execute_distributed(code, context)
            else:
                result = self._execute_normal(code, context)
            
            # Stop monitoring and get metrics
            metrics = self.resource_monitor.stop_monitoring()
            
            # Record execution
            self.db.record_execution(code, result['success'], metrics.execution_time, 
                                   self._estimate_complexity(code))
            
            result['performance_metrics'] = metrics
            result['execution_id'] = execution_id
            
            self.execution_history.append({
                'id': execution_id,
                'code': code,
                'result': result,
                'context': context,
                'timestamp': time.time()
            })
            
            return result
            
        except Exception as e:
            metrics = self.resource_monitor.stop_monitoring()
            logger.error(f"Execution failed: {e}")
            
            return {
                'success': False,
                'error': str(e),
                'execution_id': execution_id,
                'performance_metrics': metrics
            }
    
    def _execute_sandboxed(self, code: str, context: ExecutionContext) -> Dict[str, Any]:
        """Execute code in sandboxed environment"""
        # Implement sandboxing logic
        restricted_builtins = {
            '__builtins__': {
                'print': print,
                'len': len,
                'range': range,
                'enumerate': enumerate,
                'zip': zip,
                'map': map,
                'filter': filter,
                'sorted': sorted,
                'sum': sum,
                'max': max,
                'min': min,
                'abs': abs,
                'round': round,
                'int': int,
                'float': float,
                'str': str,
                'list': list,
                'dict': dict,
                'set': set,
                'tuple': tuple,
            }
        }
        
        # Execute with restricted environment
        local_vars = context.variables.copy()
        local_vars.update(restricted_builtins)
        
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, local_vars)
            
            return {
                'success': True,
                'output': stdout_capture.getvalue(),
                'variables': {k: v for k, v in local_vars.items() 
                            if not k.startswith('__') and k not in restricted_builtins},
                'execution_mode': 'sandboxed'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'output': stdout_capture.getvalue(),
                'execution_mode': 'sandboxed'
            }
    
    def _execute_parallel(self, code: str, context: ExecutionContext) -> Dict[str, Any]:
        """Execute code with parallel processing"""
        # Analyze code for parallelizable sections
        parallel_opportunities = self._find_parallel_opportunities(code)
        
        if not parallel_opportunities:
            return self._execute_normal(code, context)
        
        # Transform code for parallel execution
        parallel_code = self._transform_for_parallel(code, parallel_opportunities)
        
        return self._execute_normal(parallel_code, context)
    
    def _execute_distributed(self, code: str, context: ExecutionContext) -> Dict[str, Any]:
        """Execute code in distributed manner"""
        # This would implement distributed execution
        # For now, fall back to parallel execution
        return self._execute_parallel(code, context)
    
    def _execute_normal(self, code: str, context: ExecutionContext) -> Dict[str, Any]:
        """Standard code execution"""
        local_vars = context.variables.copy()
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        start_time = time.perf_counter()
        
        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, local_vars)
            
            end_time = time.perf_counter()
            
            return {
                'success': True,
                'output': stdout_capture.getvalue(),
                'error': stderr_capture.getvalue(),
                'execution_time': end_time - start_time,
                'variables': {k: v for k, v in local_vars.items() 
                            if not k.startswith('__')},
                'execution_mode': 'normal'
            }
        except Exception as e:
            end_time = time.perf_counter()
            return {
                'success': False,
                'output': stdout_capture.getvalue(),
                'error': f"{type(e).__name__}: {str(e)}",
                'execution_time': end_time - start_time,
                'execution_mode': 'normal'
            }
    
    def _estimate_complexity(self, code: str) -> int:
        """Estimate code complexity for database recording"""
        try:
            tree = ast.parse(code)
            complexity = 1
            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
                    complexity += 1
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    complexity += 2
                elif isinstance(node, ast.ClassDef):
                    complexity += 3
            return min(complexity, 10)
        except:
            return 1
    
    def _find_parallel_opportunities(self, code: str) -> List[Dict]:
        """Find opportunities for parallel execution"""
        opportunities = []
        
        # Look for independent loops
        loop_pattern = r'for (\w+) in (.+?):\n((?:\s+.+\n)*)'
        matches = re.finditer(loop_pattern, code, re.MULTILINE)
        
        for match in matches:
            loop_var = match.group(1)
            iterable = match.group(2)
            body = match.group(3)
            
            # Check if loop body is independent (simplified check)
            if loop_var not in body.replace(f'{loop_var}', '') and 'global' not in body:
                opportunities.append({
                    'type': 'independent_loop',
                    'variable': loop_var,
                    'iterable': iterable,
                    'body': body.strip(),
                    'position': match.span()
                })
        
        return opportunities
    
    def _transform_for_parallel(self, code: str, opportunities: List[Dict]) -> str:
        """Transform code for parallel execution"""
        transformed = code
        
        for opp in opportunities:
            if opp['type'] == 'independent_loop':
                original = f"for {opp['variable']} in {opp['iterable']}:\n{opp['body']}"
                parallel_version = f"""
# Parallel execution
from concurrent.futures import ThreadPoolExecutor
import functools

def _parallel_task({opp['variable']}):
{self._indent_code(opp['body'], 4)}
    return locals()

with ThreadPoolExecutor() as executor:
    results = list(executor.map(_parallel_task, {opp['iterable']}))
"""
                transformed = transformed.replace(original, parallel_version)
        
        return transformed

class ResourceMonitor:
    """Monitor system resources during code execution"""
    
    def __init__(self):
        self.start_stats = None
        self.monitoring = False
        self.process = psutil.Process()
    
    def start_monitoring(self):
        """Start resource monitoring"""
        self.monitoring = True
        self.start_stats = {
            'cpu_percent': self.process.cpu_percent(),
            'memory_info': self.process.memory_info(),
            'time': time.perf_counter(),
            'io_counters': self.process.io_counters() if hasattr(self.process, 'io_counters') else None
        }
    
    def stop_monitoring(self) -> PerformanceMetrics:
        """Stop monitoring and return metrics"""
        if not self.monitoring or not self.start_stats:
            return PerformanceMetrics()
        
        end_time = time.perf_counter()
        end_memory = self.process.memory_info()
        end_cpu = self.process.cpu_percent()
        end_io = self.process.io_counters() if hasattr(self.process, 'io_counters') else None
        
        metrics = PerformanceMetrics(
            execution_time=end_time - self.start_stats['time'],
            memory_usage=end_memory.rss - self.start_stats['memory_info'].rss,
            cpu_usage=(end_cpu + self.start_stats['cpu_percent']) / 2,
            io_operations=0 if not end_io or not self.start_stats['io_counters'] 
                         else (end_io.read_count + end_io.write_count - 
                              self.start_stats['io_counters'].read_count - 
                              self.start_stats['io_counters'].write_count)
        )
        
        self.monitoring = False
        return metrics

class EnhancedCodeAnalyzer(CodeAnalyzer):
    """Enhanced code analyzer with ML-based insights"""
    
    def __init__(self):
        super().__init__()
        self.complexity_model = self._init_complexity_model() if ML_AVAILABLE else None
    
    def _init_complexity_model(self):
        """Initialize ML model for complexity prediction"""
        # In production, this would load a pre-trained model
        return {
            'vectorizer': TfidfVectorizer(max_features=5000),
            'threshold_complex': 0.7,
            'threshold_expert': 0.9
        }
    
    def analyze_with_ml(self, code: str) -> CodeAnalysis:
        """Analyze code using ML-enhanced techniques"""
        base_analysis = self.analyze_code(code)
        
        if ML_AVAILABLE and self.complexity_model:
            # Enhanced complexity prediction
            ml_complexity = self._predict_complexity_ml(code)
            if ml_complexity != base_analysis.complexity:
                base_analysis.suggestions.append(
                    f"ML model suggests complexity level: {ml_complexity.name}"
                )
        
        # Advanced code smell detection
        code_smells = self._detect_advanced_code_smells(code)
        base_analysis.code_smells.extend(code_smells)
        
        # Refactoring opportunities
        refactoring_ops = self._identify_refactoring_opportunities(code)
        base_analysis.refactoring_opportunities.extend(refactoring_ops)
        
        # Test coverage estimate
        base_analysis.test_coverage_estimate = self._estimate_test_coverage(code)
        
        # Documentation score
        base_analysis.documentation_score = self._calculate_documentation_score(code)
        
        return base_analysis
    
    def _predict_complexity_ml(self, code: str) -> CodeComplexity:
        """Predict code complexity using ML"""
        # Feature extraction
        features = self._extract_code_features(code)
        
        # Simple heuristic-based prediction (in production, use trained model)
        if features['cyclomatic_complexity'] > 20:
            return CodeComplexity.EXPERT
        elif features['function_count'] > 10 or features['class_count'] > 5:
            return CodeComplexity.COMPLEX
        elif features['loop_count'] > 5:
            return CodeComplexity.MODERATE
        else:
            return CodeComplexity.SIMPLE
    
    def _extract_code_features(self, code: str) -> Dict[str, int]:
        """Extract features for ML analysis"""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return {}
        
        features = {
            'lines_of_code': len(code.split('\n')),
            'function_count': 0,
            'class_count': 0,
            'if_count': 0,
            'loop_count': 0,
            'try_count': 0,
            'import_count': 0,
            'cyclomatic_complexity': 1
        }
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                features['function_count'] += 1
            elif isinstance(node, ast.ClassDef):
                features['class_count'] += 1
            elif isinstance(node, ast.If):
                features['if_count'] += 1
                features['cyclomatic_complexity'] += 1
            elif isinstance(node, (ast.For, ast.While)):
                features['loop_count'] += 1
                features['cyclomatic_complexity'] += 1
            elif isinstance(node, ast.Try):
                features['try_count'] += 1
                features['cyclomatic_complexity'] += 1
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                features['import_count'] += 1
        
        return features
    
    def _detect_advanced_code_smells(self, code: str) -> List[str]:
        """Detect advanced code smells"""
        smells = []
        
        # Long parameter lists
        if re.search(r'def \w+\([^)]{50,}\)', code):
            smells.append("Long parameter list detected")
        
        # Deep nesting
        nesting_levels = []
        for line in code.split('\n'):
            if line.strip():
                indent_level = (len(line) - len(line.lstrip())) // 4
                nesting_levels.append(indent_level)
        
        if nesting_levels and max(nesting_levels) > 4:
            smells.append("Deep nesting detected (>4 levels)")
        
        # Large methods/functions
        functions = re.findall(r'def \w+.*?(?=\ndef|\nclass|\Z)', code, re.DOTALL)
        for func in functions:
            if len(func.split('\n')) > 50:
                smells.append("Large function detected (>50 lines)")
        
        # Magic numbers
        magic_numbers = re.findall(r'\b(?<!\.)\d{3,}\b', code)
        if magic_numbers:
            smells.append(f"Magic numbers detected: {', '.join(set(magic_numbers[:3]))}")
        
        # Duplicate code patterns
        lines = [line.strip() for line in code.split('\n') if line.strip()]
        line_counts = Counter(lines)
        duplicates = [line for line, count in line_counts.items() if count > 2 and len(line) > 20]
        if duplicates:
            smells.append("Duplicate code patterns detected")
        
        return smells
    
    def _identify_refactoring_opportunities(self, code: str) -> List[str]:
        """Identify refactoring opportunities"""
        opportunities = []
        
        # Extract method opportunities
        functions = re.findall(r'def (\w+).*?(?=\ndef|\nclass|\Z)', code, re.DOTALL)
        for func in functions:
            lines = func.split('\n')
            if len(lines) > 30:
                opportunities.append(f"Consider extracting methods from large function")
        
        # Replace conditional with polymorphism
        if re.search(r'if.*isinstance.*elif.*isinstance', code, re.DOTALL):
            opportunities.append("Consider replacing type checking with polymorphism")
        
        # Replace magic numbers with constants
        if re.findall(r'\b\d{2,}\b', code):
            opportunities.append("Consider extracting magic numbers as constants")
        
        # Introduce parameter object
        param_counts = re.findall(r'def \w+\(([^)]+)\)', code)
        for params in param_counts:
            if len(params.split(',')) > 5:
                opportunities.append("Consider introducing parameter object for functions with many parameters")
        
        return opportunities
    
    def _estimate_test_coverage(self, code: str) -> float:
        """Estimate test coverage based on code structure"""
        total_testable_units = 0
        
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    total_testable_units += 1
                elif isinstance(node, ast.If):
                    total_testable_units += 1
        except SyntaxError:
            return 0.0
        
        # Look for test-like patterns
        test_indicators = len(re.findall(r'assert|test_|Test|unittest|pytest', code))
        
        if total_testable_units == 0:
            return 0.0
        
        coverage_estimate = min(test_indicators / total_testable_units, 1.0)
        return coverage_estimate * 100
    
    def _calculate_documentation_score(self, code: str) -> float:
        """Calculate documentation quality score"""
        score = 0.0
        total_items = 0
        
        # Check for module docstring
        if re.search(r'^""".*?"""', code, re.DOTALL | re.MULTILINE):
            score += 20
        
        # Check function docstrings
        functions = re.findall(r'def (\w+).*?(?=\ndef|\nclass|\Z)', code, re.DOTALL)
        for func in functions:
            total_items += 1
            if '"""' in func or "'''" in func:
                score += 15
        
        # Check class docstrings
        classes = re.findall(r'class (\w+).*?(?=\nclass|\ndef|\Z)', code, re.DOTALL)
        for cls in classes:
            total_items += 1
            if '"""' in cls or "'''" in cls:
                score += 15
        
        # Check for comments
        comment_lines = len(re.findall(r'^\s*#.*$', code, re.MULTILINE))
        total_lines = len(code.split('\n'))
        
        if total_lines > 0:
            comment_ratio = comment_lines / total_lines
            score += comment_ratio * 30
        
        # Check for type hints
        if re.search(r':\s*(int|str|float|bool|List|Dict|Optional)', code):
            score += 20
        
        return min(score, 100.0)

class PythonCodeAgent:
    """Enhanced Python Code Agent v2.0"""
    
    def __init__(self, db_path: str = "code_agent.db"):
        self.generator = SmartCodeGenerator()
        self.optimizer = AdvancedCodeOptimizer()
        self.analyzer = EnhancedCodeAnalyzer()
        self.executor = ExecutionEngine()
        self.db = CodeDatabase(db_path)
        
        # Advanced features
        self.learning_enabled = True
        self.auto_optimization = True
        self.safety_checks = True
        
        logger.info("Enhanced Python Code Agent v2.0 initialized")
    
    def generate_and_execute(self, description: str, 
                           optimization_strategy: OptimizationStrategy = OptimizationStrategy.ALL,
                           execution_mode: ExecutionMode = ExecutionMode.NORMAL,
                           auto_optimize: bool = None) -> Dict[str, Any]:
        """Generate, optimize, and execute code in one pipeline"""
        
        if auto_optimize is None:
            auto_optimize = self.auto_optimization
        
        # Create execution context
        context = ExecutionContext(execution_mode=execution_mode)
        
        # Generate code
        logger.info(f"Generating code for: {description}")
        generated_code = self.generator.generate_intelligent_code(description, context, optimization_strategy)
        
        # Analyze generated code
        analysis = self.analyzer.analyze_with_ml(generated_code)
        
        # Optimize if requested
        if auto_optimize:
            optimized_code, optimizations, confidence = self.optimizer.optimize_with_learning(
                generated_code, context
            )
            logger.info(f"Applied {len(optimizations)} optimizations with {confidence:.2f} confidence")
        else:
            optimized_code = generated_code
            optimizations = []
            confidence = 1.0
        
        # Execute code
        execution_result = self.executor.execute_with_monitoring(optimized_code, context)
        
        # Compile results
        result = {
            'description': description,
            'generated_code': generated_code,
            'optimized_code': optimized_code,
            'analysis': analysis,
            'optimizations': optimizations,
            'optimization_confidence': confidence,
            'execution_result': execution_result,
            'context': context,
            'timestamp': time.time()
        }
        
        # Learn from execution if enabled
        if self.learning_enabled and execution_result.get('success'):
            self._update_learning_data(description, optimized_code, execution_result, analysis)
        
        return result
    
    def batch_process(self, tasks: List[str], 
                     max_workers: int = 4) -> List[Dict[str, Any]]:
        """Process multiple code generation tasks in parallel"""
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self.generate_and_execute, task): task 
                for task in tasks
            }
            
            # Collect results
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"Completed task: {task}")
                except Exception as e:
                    logger.error(f"Task failed: {task}, Error: {e}")
                    results.append({
                        'description': task,
                        'error': str(e),
                        'success': False
                    })
        
        return results
    
    def interactive_mode(self):
        """Start interactive mode for code generation"""
        print("🐍 Enhanced Python Code Agent v2.0 - Interactive Mode")
        print("Type 'exit' to quit, 'help' for commands")
        
        while True:
            try:
                user_input = input("\n📝 Describe what you want to code: ").strip()
                
                if user_input.lower() == 'exit':
                    break
                elif user_input.lower() == 'help':
                    self._show_help()
                    continue
                elif user_input.lower().startswith('analyze '):
                    # Analyze existing code
                    code = user_input[8:]
                    analysis = self.analyzer.analyze_with_ml(code)
                    self._display_analysis(analysis)
                    continue
                elif user_input.lower().startswith('optimize '):
                    # Optimize existing code
                    code = user_input[9:]
                    context = ExecutionContext()
                    optimized, opts, conf = self.optimizer.optimize_with_learning(code, context)
                    print(f"\n✨ Optimized Code (confidence: {conf:.2f}):")
                    print(optimized)
                    print(f"\n🔧 Applied optimizations: {opts}")
                    continue
                
                if not user_input:
                    continue
                
                # Generate and execute code
                result = self.generate_and_execute(user_input)
                
                # Display results
                self._display_results(result)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                logger.error(f"Interactive mode error: {e}")
    
    def _show_help(self):
        """Show help information"""
        help_text = """
🔧 Available Commands:
- describe your task in natural language
- 'analyze <code>' - analyze existing code
- 'optimize <code>' - optimize existing code  
- 'help' - show this help
- 'exit' - quit interactive mode

🎯 Example tasks:
- "Create a function to sort a list of numbers"
- "Build a web scraper for product prices"
- "Generate a data analysis script for CSV files"
- "Create an API client for REST services"
"""
        print(help_text)
    
    def _display_results(self, result: Dict[str, Any]):
        """Display execution results in a formatted way"""
        print(f"\n🎯 Task: {result['description']}")
        
        # Show analysis
        analysis = result['analysis']
        print(f"📊 Analysis: Complexity={analysis.complexity.name}, "
              f"Lines={analysis.lines_of_code}, "
              f"Maintainability={analysis.maintainability_index:.1f}")
        
        # Show optimizations
        if result['optimizations']:
            print(f"⚡ Optimizations ({result['optimization_confidence']:.2f} confidence):")
            for opt in result['optimizations'][:3]:  # Show first 3
                print(f"  • {opt}")
        
        # Show execution result
        exec_result = result['execution_result']
        if exec_result['success']:
            print("✅ Execution: SUCCESS")
            if exec_result.get('output'):
                print(f"📤 Output:\n{exec_result['output']}")
            
            # Show performance
            if 'performance_metrics' in exec_result:
                metrics = exec_result['performance_metrics']
                print(f"⚡ Performance: {metrics.execution_time:.4f}s, "
                      f"Memory: {metrics.memory_usage/1024/1024:.2f}MB")
        else:
            print("❌ Execution: FAILED")
            print(f"💥 Error: {exec_result.get('error', 'Unknown error')}")
        
        print(f"\n💻 Generated Code:")
        print("```python")
        print(result['optimized_code'])
        print("```")
    
    def _display_analysis(self, analysis: CodeAnalysis):
        """Display code analysis results"""
        print(f"\n📊 Code Analysis Results:")
        print(f"🎯 Complexity: {analysis.complexity.value[1]}")
        print(f"📏 Lines of Code: {analysis.lines_of_code}")
        print(f"🔄 Cyclomatic Complexity: {analysis.cyclomatic_complexity}")
        print(f"🛠️  Maintainability Index: {analysis.maintainability_index:.1f}/100")
        print(f"🔒 Security Score: {analysis.security_analysis.risk_level}")
        print(f"📚 Documentation Score: {analysis.documentation_score:.1f}/100")
        
        if analysis.suggestions:
            print(f"\n💡 Suggestions:")
            for suggestion in analysis.suggestions[:5]:
                print(f"  • {suggestion}")
        
        if analysis.code_smells:
            print(f"\n👃 Code Smells:")
            for smell in analysis.code_smells[:3]:
                print(f"  • {smell}")
        
        if analysis.refactoring_opportunities:
            print(f"\n🔧 Refactoring Opportunities:")
            for opportunity in analysis.refactoring_opportunities[:3]:
                print(f"  • {opportunity}")
    
    def _update_learning_data(self, description: str, code: str, 
                            execution_result: Dict, analysis: CodeAnalysis):
        """Update learning data based on successful execution"""
        # This would implement machine learning updates
        # For now, just log the success
        complexity_score = analysis.complexity.value[0]
        execution_time = execution_result.get('performance_metrics', PerformanceMetrics()).execution_time
        
        self.db.record_execution(code, True, execution_time, complexity_score)
        
        if self.learning_enabled:
            logger.info(f"Updated learning data for task: {description[:50]}...")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get agent statistics and performance metrics"""
        with sqlite3.connect(self.db.db_path) as conn:
            cursor = conn.cursor()
            
            # Execution statistics
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_executions,
                    AVG(success_rate) as avg_success_rate,
                    AVG(avg_execution_time) as avg_time,
                    AVG(complexity_score) as avg_complexity
                FROM code_executions
            """)
            
            stats = cursor.fetchone()
            
            return {
                'total_executions': stats[0] if stats[0] else 0,
                'average_success_rate': stats[1] if stats[1] else 0.0,
                'average_execution_time': stats[2] if stats[2] else 0.0,
                'average_complexity': stats[3] if stats[3] else 0.0,
                'learning_enabled': self.learning_enabled,
                'auto_optimization': self.auto_optimization,
                'database_path': self.db.db_path
            }

# Main execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Python Code Agent v2.0')
    parser.add_argument('--interactive', action='store_true', help='Start interactive mode')
    parser.add_argument('--task', type=str, help='Single task to execute')
    parser.add_argument('--batch', type=str, help='File with tasks to batch process')
    parser.add_argument('--stats', action='store_true', help='Show statistics')
    
    args = parser.parse_args()
    
    # Initialize agent
    agent = PythonCodeAgent()
    
    if args.stats:
        stats = agent.get_statistics()
        print("📊 Agent Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    elif args.interactive:
        agent.interactive_mode()
    
    elif args.task:
        result = agent.generate_and_execute(args.task)
        agent._display_results(result)
    
    elif args.batch:
        try:
            with open(args.batch, 'r') as f:
                tasks = [line.strip() for line in f if line.strip()]
            
            results = agent.batch_process(tasks)
            print(f"✅ Processed {len(results)} tasks")
            
            # Summary statistics
            successful = sum(1 for r in results if r.get('execution_result', {}).get('success'))
            print(f"📊 Success rate: {successful}/{len(results)} ({successful/len(results)*100:.1f}%)")
            
        except FileNotFoundError:
            print(f"❌ Batch file not found: {args.batch}")
    
    else:
        print("🐍 Enhanced Python Code Agent v2.0")
        print("Use --help for usage information")
        print("Use --interactive to start interactive mode")