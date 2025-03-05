import pytest
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from src.train import ModelTrainer
import joblib
import tempfile
import os

@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    X = pd.DataFrame({
        'variance': [1.0, 2.0, 3.0, 4.0, 5.0],
        'skewness': [1.0, 2.0, 3.0, 4.0, 5.0],
        'curtosis': [1.0, 2.0, 3.0, 4.0, 5.0],
        'entropy': [1.0, 2.0, 3.0, 4.0, 5.0]
    })
    y = pd.Series([0, 1, 0, 1, 0])
    return X, y

def test_model_trainer_initialization():
    """Test ModelTrainer initialization"""
    trainer = ModelTrainer()
    assert trainer.model is None

def test_model_training(sample_data):
    """Test model training functionality"""
    X_train, y_train = sample_data
    trainer = ModelTrainer()
    
    # Train model
    model = trainer.train(X_train, y_train)
    
    # Check if model is trained
    assert isinstance(model, DecisionTreeClassifier)
    assert trainer.model is not None
    
    # Check if model can make predictions
    predictions = model.predict(X_train)
    assert len(predictions) == len(y_train)
    assert all(isinstance(pred, (int, np.integer)) for pred in predictions)

def test_model_save_load(tmp_path):
    """Test model saving and loading"""
    # Create sample data
    X = pd.DataFrame({
        'variance': [1.0, 2.0],
        'skewness': [1.0, 2.0],
        'curtosis': [1.0, 2.0],
        'entropy': [1.0, 2.0]
    })
    y = pd.Series([0, 1])
    
    # Train model
    trainer = ModelTrainer()
    model = trainer.train(X, y)
    
    # Save model
    model_path = tmp_path / "test_model.joblib"
    trainer.save_model(str(model_path))
    
    # Check if file exists
    assert os.path.exists(model_path)
    
    # Load model and check if it works
    loaded_model = joblib.load(model_path)
    assert isinstance(loaded_model, DecisionTreeClassifier)
    
    # Check if loaded model makes same predictions
    original_predictions = model.predict(X)
    loaded_predictions = loaded_model.predict(X)
    np.testing.assert_array_equal(original_predictions, loaded_predictions)

def test_export_tree(tmp_path):
    """Test tree structure export"""
    # Create sample data
    X = pd.DataFrame({
        'variance': [1.0, 2.0],
        'skewness': [1.0, 2.0],
        'curtosis': [1.0, 2.0],
        'entropy': [1.0, 2.0]
    })
    y = pd.Series([0, 1])
    
    # Train model
    trainer = ModelTrainer()
    trainer.train(X, y)
    
    # Export tree
    output_file = tmp_path / "test_tree.dot"
    trainer.export_tree(
        feature_names=['variance', 'skewness', 'curtosis', 'entropy'],
        class_names_list=['0', '1'],
        output_file=str(output_file)
    )
    
    # Check if file exists and has content
    assert os.path.exists(output_file)
    with open(output_file, 'r') as f:
        content = f.read()
        assert len(content) > 0
        assert 'digraph' in content  # DOT file should contain this 