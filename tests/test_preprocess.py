import pytest
import pandas as pd
import numpy as np
from src.preprocess import Preprocessor, split_data

@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    return pd.DataFrame({
        'variance': [1.0, 2.0, 3.0, 4.0, 5.0],
        'skewness': [1.0, 2.0, 3.0, 4.0, 5.0],
        'curtosis': [1.0, 2.0, 3.0, 4.0, 5.0],
        'entropy': [1.0, 2.0, 3.0, 4.0, 5.0],
        'class': [0, 1, 0, 1, 0]
    })

def test_split_data(sample_data):
    """Test data splitting functionality"""
    test_size = 0.2
    random_state = 42
    
    X_train, X_test, y_train, y_test = split_data(
        sample_data,
        test_size=test_size,
        random_state=random_state
    )
    
    # Check shapes
    assert len(X_train) == 4  # 80% of 5 samples
    assert len(X_test) == 1   # 20% of 5 samples
    assert len(y_train) == 4
    assert len(y_test) == 1
    
    # Check feature names
    assert list(X_train.columns) == ['variance', 'skewness', 'curtosis', 'entropy']
    
    # Check data types
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_test, pd.Series)

def test_preprocessor_initialization(tmp_path):
    """Test Preprocessor initialization"""
    # Create a temporary CSV file
    data = pd.DataFrame({
        'variance': [1.0, 2.0],
        'skewness': [1.0, 2.0],
        'curtosis': [1.0, 2.0],
        'entropy': [1.0, 2.0],
        'class': [0, 1]
    })
    test_file = tmp_path / "test_data.csv"
    data.to_csv(test_file, index=False)
    
    # Initialize preprocessor
    preprocessor = Preprocessor(str(test_file))
    assert preprocessor.data_path == str(test_file)

def test_preprocessor_load_data(tmp_path):
    """Test data loading functionality"""
    # Create a temporary CSV file
    data = pd.DataFrame({
        'variance': [1.0, 2.0],
        'skewness': [1.0, 2.0],
        'curtosis': [1.0, 2.0],
        'entropy': [1.0, 2.0],
        'class': [0, 1]
    })
    test_file = tmp_path / "test_data.csv"
    data.to_csv(test_file, index=False)
    
    # Load data
    preprocessor = Preprocessor(str(test_file))
    loaded_data = preprocessor.load_data()
    
    # Check if loaded data matches original
    pd.testing.assert_frame_equal(loaded_data, data)

def test_preprocessor_invalid_file():
    """Test handling of invalid file path"""
    with pytest.raises(Exception):
        preprocessor = Preprocessor("nonexistent_file.csv")
        preprocessor.load_data() 