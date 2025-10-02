# -*- coding: utf-8 -*-
"""
AI Model Management for NeuraSynth Integrated System
Supports comprehensive AI model lifecycle management
"""

from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
import uuid
import json
from enum import Enum

db = SQLAlchemy()

class ModelType(Enum):
    """AI model type enumeration"""
    LANGUAGE_MODEL = "language_model"
    COMPUTER_VISION = "computer_vision"
    SPEECH_RECOGNITION = "speech_recognition"
    TEXT_TO_SPEECH = "text_to_speech"
    RECOMMENDATION = "recommendation"
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    GENERATIVE = "generative"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    CUSTOM = "custom"

class ModelStatus(Enum):
    """AI model status enumeration"""
    PLANNING = "planning"
    DEVELOPMENT = "development"
    TRAINING = "training"
    VALIDATION = "validation"
    TESTING = "testing"
    DEPLOYED = "deployed"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    FAILED = "failed"
    ARCHIVED = "archived"

class TrainingStatus(Enum):
    """Training status enumeration"""
    NOT_STARTED = "not_started"
    PREPARING = "preparing"
    TRAINING = "training"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"

class AIModel(db.Model):
    """Comprehensive AI model management"""
    __tablename__ = 'ai_models'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    organization_id = db.Column(db.String(36), db.ForeignKey('organizations.id'), nullable=False)
    project_id = db.Column(db.String(36), db.ForeignKey('projects.id'), nullable=False)
    
    # Model identification
    name = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text)
    version = db.Column(db.String(50), default='1.0.0')
    model_code = db.Column(db.String(100), unique=True)  # Unique model identifier
    
    # Model classification
    type = db.Column(db.Enum(ModelType), default=ModelType.CUSTOM)
    status = db.Column(db.Enum(ModelStatus), default=ModelStatus.PLANNING)
    framework = db.Column(db.String(100))  # tensorflow, pytorch, scikit-learn, etc.
    
    # Model architecture and configuration
    architecture = db.Column(db.Text)  # JSON with model architecture details
    hyperparameters = db.Column(db.Text, default='{}')  # JSON with hyperparameters
    configuration = db.Column(db.Text, default='{}')  # JSON with model configuration
    
    # Training information
    training_status = db.Column(db.Enum(TrainingStatus), default=TrainingStatus.NOT_STARTED)
    training_data_path = db.Column(db.String(500))
    validation_data_path = db.Column(db.String(500))
    test_data_path = db.Column(db.String(500))
    
    # Training metrics
    training_metrics = db.Column(db.Text, default='{}')  # JSON with training metrics
    validation_metrics = db.Column(db.Text, default='{}')  # JSON with validation metrics
    test_metrics = db.Column(db.Text, default='{}')  # JSON with test metrics
    
    # Model performance
    accuracy = db.Column(db.Numeric(5, 4))  # e.g., 0.9500 for 95%
    precision = db.Column(db.Numeric(5, 4))
    recall = db.Column(db.Numeric(5, 4))
    f1_score = db.Column(db.Numeric(5, 4))
    
    # Training timeline
    training_started_at = db.Column(db.DateTime)
    training_completed_at = db.Column(db.DateTime)
    training_duration_minutes = db.Column(db.Integer)
    
    # Model files and artifacts
    model_file_path = db.Column(db.String(500))
    weights_file_path = db.Column(db.String(500))
    config_file_path = db.Column(db.String(500))
    artifacts = db.Column(db.Text, default='[]')  # JSON array of artifact paths
    
    # Deployment information
    deployment_config = db.Column(db.Text, default='{}')  # JSON with deployment config
    endpoint_url = db.Column(db.String(500))
    api_key = db.Column(db.String(255))
    deployment_environment = db.Column(db.String(100))  # development, staging, production
    
    # Resource requirements
    cpu_requirements = db.Column(db.String(100))
    memory_requirements = db.Column(db.String(100))
    gpu_requirements = db.Column(db.String(100))
    storage_requirements = db.Column(db.String(100))
    
    # Usage and monitoring
    usage_count = db.Column(db.Integer, default=0)
    last_used_at = db.Column(db.DateTime)
    monitoring_config = db.Column(db.Text, default='{}')  # JSON with monitoring setup
    
    # Versioning and lineage
    parent_model_id = db.Column(db.String(36), db.ForeignKey('ai_models.id'))
    is_baseline = db.Column(db.Boolean, default=False)
    
    # Metadata and tags
    metadata = db.Column(db.Text, default='{}')
    tags = db.Column(db.Text, default='[]')
    
    # Team and ownership
    created_by = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=False)
    trained_by = db.Column(db.String(36), db.ForeignKey('users.id'))
    deployed_by = db.Column(db.String(36), db.ForeignKey('users.id'))
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    deployed_at = db.Column(db.DateTime)
    
    # Relationships
    creator = db.relationship('User', foreign_keys=[created_by], backref='created_models')
    trainer = db.relationship('User', foreign_keys=[trained_by], backref='trained_models')
    deployer = db.relationship('User', foreign_keys=[deployed_by], backref='deployed_models')
    parent_model = db.relationship('AIModel', remote_side=[id], backref='child_models')
    
    def __init__(self, name, project_id, created_by, organization_id, **kwargs):
        self.name = name
        self.project_id = project_id
        self.created_by = created_by
        self.organization_id = organization_id
        
        # Generate unique model code if not provided
        if 'model_code' not in kwargs:
            self.model_code = self.generate_model_code()
        
        # Set optional fields
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def generate_model_code(self):
        """Generate unique model code"""
        import random
        import string
        
        # Format: AI-YYYY-XXXX (e.g., AI-2024-A1B2)
        year = datetime.now().year
        random_part = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
        return f"AI-{year}-{random_part}"
    
    def get_architecture(self):
        """Get architecture as dictionary"""
        try:
            return json.loads(self.architecture or '{}')
        except json.JSONDecodeError:
            return {}
    
    def set_architecture(self, architecture_dict):
        """Set architecture from dictionary"""
        self.architecture = json.dumps(architecture_dict)
    
    def get_hyperparameters(self):
        """Get hyperparameters as dictionary"""
        try:
            return json.loads(self.hyperparameters or '{}')
        except json.JSONDecodeError:
            return {}
    
    def set_hyperparameters(self, hyperparams_dict):
        """Set hyperparameters from dictionary"""
        self.hyperparameters = json.dumps(hyperparams_dict)
    
    def get_configuration(self):
        """Get configuration as dictionary"""
        try:
            return json.loads(self.configuration or '{}')
        except json.JSONDecodeError:
            return {}
    
    def set_configuration(self, config_dict):
        """Set configuration from dictionary"""
        self.configuration = json.dumps(config_dict)
    
    def get_training_metrics(self):
        """Get training metrics as dictionary"""
        try:
            return json.loads(self.training_metrics or '{}')
        except json.JSONDecodeError:
            return {}
    
    def set_training_metrics(self, metrics_dict):
        """Set training metrics from dictionary"""
        self.training_metrics = json.dumps(metrics_dict)
    
    def get_validation_metrics(self):
        """Get validation metrics as dictionary"""
        try:
            return json.loads(self.validation_metrics or '{}')
        except json.JSONDecodeError:
            return {}
    
    def set_validation_metrics(self, metrics_dict):
        """Set validation metrics from dictionary"""
        self.validation_metrics = json.dumps(metrics_dict)
    
    def get_test_metrics(self):
        """Get test metrics as dictionary"""
        try:
            return json.loads(self.test_metrics or '{}')
        except json.JSONDecodeError:
            return {}
    
    def set_test_metrics(self, metrics_dict):
        """Set test metrics from dictionary"""
        self.test_metrics = json.dumps(metrics_dict)
    
    def get_artifacts(self):
        """Get artifacts as list"""
        try:
            return json.loads(self.artifacts or '[]')
        except json.JSONDecodeError:
            return []
    
    def set_artifacts(self, artifacts_list):
        """Set artifacts from list"""
        self.artifacts = json.dumps(artifacts_list)
    
    def add_artifact(self, artifact_path):
        """Add an artifact"""
        artifacts = self.get_artifacts()
        if artifact_path not in artifacts:
            artifacts.append(artifact_path)
            self.set_artifacts(artifacts)
    
    def get_deployment_config(self):
        """Get deployment config as dictionary"""
        try:
            return json.loads(self.deployment_config or '{}')
        except json.JSONDecodeError:
            return {}
    
    def set_deployment_config(self, config_dict):
        """Set deployment config from dictionary"""
        self.deployment_config = json.dumps(config_dict)
    
    def get_monitoring_config(self):
        """Get monitoring config as dictionary"""
        try:
            return json.loads(self.monitoring_config or '{}')
        except json.JSONDecodeError:
            return {}
    
    def set_monitoring_config(self, config_dict):
        """Set monitoring config from dictionary"""
        self.monitoring_config = json.dumps(config_dict)
    
    def get_metadata(self):
        """Get metadata as dictionary"""
        try:
            return json.loads(self.metadata or '{}')
        except json.JSONDecodeError:
            return {}
    
    def set_metadata(self, metadata_dict):
        """Set metadata from dictionary"""
        self.metadata = json.dumps(metadata_dict)
    
    def get_tags(self):
        """Get tags as list"""
        try:
            return json.loads(self.tags or '[]')
        except json.JSONDecodeError:
            return []
    
    def set_tags(self, tags_list):
        """Set tags from list"""
        self.tags = json.dumps(tags_list)
    
    def add_tag(self, tag):
        """Add a tag"""
        tags = self.get_tags()
        if tag not in tags:
            tags.append(tag)
            self.set_tags(tags)
    
    def start_training(self, trained_by_user_id):
        """Start model training"""
        self.training_status = TrainingStatus.TRAINING
        self.training_started_at = datetime.utcnow()
        self.trained_by = trained_by_user_id
        self.status = ModelStatus.TRAINING
    
    def complete_training(self, metrics=None):
        """Complete model training"""
        self.training_status = TrainingStatus.COMPLETED
        self.training_completed_at = datetime.utcnow()
        self.status = ModelStatus.VALIDATION
        
        if self.training_started_at:
            duration = self.training_completed_at - self.training_started_at
            self.training_duration_minutes = int(duration.total_seconds() / 60)
        
        if metrics:
            self.set_training_metrics(metrics)
    
    def fail_training(self, error_message=None):
        """Mark training as failed"""
        self.training_status = TrainingStatus.FAILED
        self.status = ModelStatus.FAILED
        
        if error_message:
            metadata = self.get_metadata()
            metadata['training_error'] = error_message
            metadata['failed_at'] = datetime.utcnow().isoformat()
            self.set_metadata(metadata)
    
    def deploy(self, deployed_by_user_id, endpoint_url=None, environment='production'):
        """Deploy the model"""
        self.status = ModelStatus.DEPLOYED
        self.deployed_by = deployed_by_user_id
        self.deployed_at = datetime.utcnow()
        self.deployment_environment = environment
        
        if endpoint_url:
            self.endpoint_url = endpoint_url
    
    def promote_to_production(self):
        """Promote model to production"""
        self.status = ModelStatus.PRODUCTION
        self.deployment_environment = 'production'
    
    def deprecate(self, reason=None):
        """Deprecate the model"""
        self.status = ModelStatus.DEPRECATED
        
        if reason:
            metadata = self.get_metadata()
            metadata['deprecation_reason'] = reason
            metadata['deprecated_at'] = datetime.utcnow().isoformat()
            self.set_metadata(metadata)
    
    def record_usage(self):
        """Record model usage"""
        self.usage_count += 1
        self.last_used_at = datetime.utcnow()
    
    def create_child_version(self, name, created_by, **kwargs):
        """Create a child version of this model"""
        # Extract version number and increment
        current_version = self.version.split('.')
        major, minor, patch = int(current_version[0]), int(current_version[1]), int(current_version[2])
        new_version = f"{major}.{minor}.{patch + 1}"
        
        child_model = AIModel(
            name=name,
            project_id=self.project_id,
            created_by=created_by,
            organization_id=self.organization_id,
            parent_model_id=self.id,
            version=new_version,
            type=self.type,
            framework=self.framework,
            **kwargs
        )
        
        return child_model
    
    def is_deployed(self):
        """Check if model is deployed"""
        return self.status in [ModelStatus.DEPLOYED, ModelStatus.PRODUCTION]
    
    def is_training(self):
        """Check if model is currently training"""
        return self.training_status == TrainingStatus.TRAINING
    
    def get_performance_summary(self):
        """Get performance metrics summary"""
        return {
            'accuracy': float(self.accuracy) if self.accuracy else None,
            'precision': float(self.precision) if self.precision else None,
            'recall': float(self.recall) if self.recall else None,
            'f1_score': float(self.f1_score) if self.f1_score else None
        }
    
    def to_dict(self, include_details=False):
        """Convert model to dictionary"""
        result = {
            'id': self.id,
            'organization_id': self.organization_id,
            'project_id': self.project_id,
            'name': self.name,
            'description': self.description,
            'version': self.version,
            'model_code': self.model_code,
            'type': self.type.value if self.type else None,
            'status': self.status.value if self.status else None,
            'framework': self.framework,
            'training_status': self.training_status.value if self.training_status else None,
            'performance': self.get_performance_summary(),
            'training_started_at': self.training_started_at.isoformat() if self.training_started_at else None,
            'training_completed_at': self.training_completed_at.isoformat() if self.training_completed_at else None,
            'training_duration_minutes': self.training_duration_minutes,
            'endpoint_url': self.endpoint_url,
            'deployment_environment': self.deployment_environment,
            'usage_count': self.usage_count,
            'last_used_at': self.last_used_at.isoformat() if self.last_used_at else None,
            'parent_model_id': self.parent_model_id,
            'is_baseline': self.is_baseline,
            'tags': self.get_tags(),
            'created_by': self.created_by,
            'trained_by': self.trained_by,
            'deployed_by': self.deployed_by,
            'is_deployed': self.is_deployed(),
            'is_training': self.is_training(),
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'deployed_at': self.deployed_at.isoformat() if self.deployed_at else None
        }
        
        if include_details:
            result.update({
                'architecture': self.get_architecture(),
                'hyperparameters': self.get_hyperparameters(),
                'configuration': self.get_configuration(),
                'training_metrics': self.get_training_metrics(),
                'validation_metrics': self.get_validation_metrics(),
                'test_metrics': self.get_test_metrics(),
                'artifacts': self.get_artifacts(),
                'deployment_config': self.get_deployment_config(),
                'monitoring_config': self.get_monitoring_config(),
                'metadata': self.get_metadata(),
                'creator': self.creator.to_dict() if self.creator else None,
                'trainer': self.trainer.to_dict() if self.trainer else None,
                'deployer': self.deployer.to_dict() if self.deployer else None,
                'parent_model': self.parent_model.to_dict() if self.parent_model else None,
                'child_models': [child.to_dict() for child in self.child_models]
            })
        
        return result
    
    def __repr__(self):
        return f'<AIModel {self.name} v{self.version} ({self.model_code})>'

class ModelExperiment(db.Model):
    """Model experiment tracking"""
    __tablename__ = 'model_experiments'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    model_id = db.Column(db.String(36), db.ForeignKey('ai_models.id'), nullable=False)
    
    # Experiment details
    name = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text)
    experiment_type = db.Column(db.String(100))  # hyperparameter_tuning, architecture_search, etc.
    
    # Experiment configuration
    parameters = db.Column(db.Text, default='{}')  # JSON with experiment parameters
    results = db.Column(db.Text, default='{}')  # JSON with experiment results
    
    # Status and timeline
    status = db.Column(db.String(50), default='running')  # running, completed, failed, cancelled
    started_at = db.Column(db.DateTime, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime)
    
    # Created by
    created_by = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=False)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    model = db.relationship('AIModel', backref='experiments')
    creator = db.relationship('User', backref='model_experiments')
    
    def __init__(self, name, model_id, created_by, **kwargs):
        self.name = name
        self.model_id = model_id
        self.created_by = created_by
        
        # Set optional fields
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def get_parameters(self):
        """Get parameters as dictionary"""
        try:
            return json.loads(self.parameters or '{}')
        except json.JSONDecodeError:
            return {}
    
    def set_parameters(self, params_dict):
        """Set parameters from dictionary"""
        self.parameters = json.dumps(params_dict)
    
    def get_results(self):
        """Get results as dictionary"""
        try:
            return json.loads(self.results or '{}')
        except json.JSONDecodeError:
            return {}
    
    def set_results(self, results_dict):
        """Set results from dictionary"""
        self.results = json.dumps(results_dict)
    
    def complete(self, results=None):
        """Complete the experiment"""
        self.status = 'completed'
        self.completed_at = datetime.utcnow()
        
        if results:
            self.set_results(results)
    
    def fail(self, error_message=None):
        """Mark experiment as failed"""
        self.status = 'failed'
        self.completed_at = datetime.utcnow()
        
        if error_message:
            results = self.get_results()
            results['error'] = error_message
            self.set_results(results)
    
    def to_dict(self):
        """Convert experiment to dictionary"""
        return {
            'id': self.id,
            'model_id': self.model_id,
            'name': self.name,
            'description': self.description,
            'experiment_type': self.experiment_type,
            'parameters': self.get_parameters(),
            'results': self.get_results(),
            'status': self.status,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'created_by': self.created_by,
            'creator': self.creator.to_dict() if self.creator else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
    
    def __repr__(self):
        return f'<ModelExperiment {self.name} for {self.model_id}>'

