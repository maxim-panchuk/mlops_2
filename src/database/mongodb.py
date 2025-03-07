import os
from typing import Dict, List, Any
from pymongo import MongoClient
from dotenv import load_dotenv
# Loading environment variables

class MongoDB:
    def __init__(self):
        load_dotenv(override=True)
  
        self.client = MongoClient(
            os.getenv('MONGODB_URI'),
            username=os.getenv('MONGODB_USER'),
            password=os.getenv('MONGODB_PASSWORD'),
            authSource=os.getenv('MONGODB_AUTH_SOURCE')
        )
        self.db = self.client[os.getenv('MONGODB_DB_NAME')]
        
    def save_model_result(self, collection_name, prediction):
        """Saving model execution result"""
        collection = self.db[collection_name]
        collection.insert_one({
            "prediction": prediction
        })
    
    def close(self):
        """Closing database connection"""
        self.client.close() 