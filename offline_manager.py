import os
import json
import pickle
from datetime import datetime
import numpy as np

class OfflineManager:
    def __init__(self):
        self.cache_dir = 'cache'
        self.predictions_file = os.path.join(self.cache_dir, 'offline_predictions.json')
        self.model_cache = os.path.join(self.cache_dir, 'model_cache.pkl')
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            
        # Initialize offline predictions storage
        if not os.path.exists(self.predictions_file):
            self._save_predictions([])
    
    def cache_model(self, model):
        """Cache the model for offline use"""
        try:
            with open(self.model_cache, 'wb') as f:
                pickle.dump(model, f)
            return True
        except Exception as e:
            print(f"Error caching model: {str(e)}")
            return False
    
    def load_cached_model(self):
        """Load model from cache"""
        try:
            if os.path.exists(self.model_cache):
                with open(self.model_cache, 'rb') as f:
                    return pickle.load(f)
            return None
        except Exception as e:
            print(f"Error loading cached model: {str(e)}")
            return None
    
    def save_offline_prediction(self, image_path, prediction, confidence_scores):
        """Save prediction for offline storage"""
        predictions = self._load_predictions()
        
        prediction_entry = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'image_path': image_path,
            'prediction': prediction,
            'confidence_scores': confidence_scores
        }
        
        predictions.append(prediction_entry)
        self._save_predictions(predictions)
        
        return True
    
    def get_offline_predictions(self):
        """Get all stored offline predictions"""
        return self._load_predictions()
    
    def sync_predictions(self, online_manager):
        """Sync offline predictions with online storage when connection is available"""
        offline_predictions = self._load_predictions()
        
        for pred in offline_predictions:
            try:
                # Attempt to sync with online storage
                online_manager.save_prediction(
                    pred['image_path'],
                    pred['prediction'],
                    pred['confidence_scores']
                )
            except Exception as e:
                print(f"Error syncing prediction: {str(e)}")
                continue
        
        # Clear offline predictions after successful sync
        self._save_predictions([])
        
        return True
    
    def _load_predictions(self):
        """Load predictions from file"""
        try:
            with open(self.predictions_file, 'r') as f:
                return json.load(f)
        except Exception:
            return []
    
    def _save_predictions(self, predictions):
        """Save predictions to file"""
        try:
            with open(self.predictions_file, 'w') as f:
                json.dump(predictions, f)
            return True
        except Exception as e:
            print(f"Error saving predictions: {str(e)}")
            return False 