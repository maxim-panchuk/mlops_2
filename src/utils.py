from sklearn.metrics import classification_report
import configparser
import pandas as pd
from typing import Dict, Any
from src.logger import Logger

config = configparser.ConfigParser()
config.read('config.ini')

output_file = config['train']['output_file']

def evaluate(model, X_test, y_test):
    """
    Output classification_report.
    """
    pred = model.predict(X_test)
    print(classification_report(y_test, pred))


class ModelEvaluator:
    def __init__(self, logger: Logger):
        """
        Initialize ModelEvaluator
        
        Args:
            logger: Logger instance
        """
        self.logger = logger

    def evaluate(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Evaluate model performance and generate classification report
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary containing evaluation metrics
        """
        self.logger.info("Evaluating model performance")
        predictions = model.predict(X_test)
        report = classification_report(y_test, predictions, output_dict=True)
        
        self.logger.info(f"Model evaluation completed. Accuracy: {report['accuracy']:.4f}")
        return report