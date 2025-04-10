from src.preprocess import Preprocessor, split_data
from src.train import ModelTrainer
from src.utils import ModelUtils
from src.config import Config
from src.logger import setup_logger
import logging

# Initialize config
config = Config()

# Get config
preprocess_config = config.get_preprocess_config()
train_config = config.get_train_config()
logging_config = config.get_logging_config()

# Setup logger
logger = setup_logger(
    name="banknote_classifier",
    log_file=logging_config["log_file"],
    level=getattr(logging, logging_config["level"])
)

logger.info("Starting banknote classification pipeline")

# 1. Load and split data
logger.info("Loading and preprocessing data")
preprocessor = Preprocessor(preprocess_config["data_path"])
data = preprocessor.load_data()
logger.info(f"Loaded {len(data)} samples")

X_train, X_test, y_train, y_test = split_data(
    data,
    test_size=preprocess_config["test_size"],
    random_state=preprocess_config["random_state"]
)
logger.info(f"Split data into train ({len(X_train)} samples) and test ({len(X_test)} samples) sets")

# 2. Training
logger.info("Starting model training")
trainer = ModelTrainer()
model = trainer.train(X_train, y_train)
logger.info("Model training completed")

# 3. Export tree struct
logger.info("Exporting decision tree structure")
class_names_list = [str(cls) for cls in data['class'].unique().tolist()]
trainer.export_tree(
    feature_names=['variance', 'skewness', 'curtosis', 'entropy'],
    class_names_list=class_names_list,
    output_file=train_config["output_file"]
)
logger.info(f"Tree structure exported to {train_config['output_file']}")

# 4. Evaluation
logger.info("Evaluating model performance")
model_utils = ModelUtils()
model_utils.evaluate(model, X_test, y_test)

# 5. Save model
logger.info(f"Saving model to {train_config['model_path']}")
trainer.save_model(train_config["model_path"])
logger.info("Pipeline completed successfully")