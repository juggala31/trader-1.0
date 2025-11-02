# Real-time Probability Calibrator for XGBoost Model
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
import joblib
import logging
from collections import deque
import threading
import time

class ProbabilityCalibrator:
    def __init__(self, model_path, calibration_window=500):
        self.model = joblib.load(model_path) if model_path else None
        self.calibrator = None
        self.calibration_data = deque(maxlen=calibration_window)
        self.calibration_window = calibration_window
        self.last_calibration_time = 0
        self.calibration_interval = 3600  # Recalibrate every hour
        
        # Thread lock for thread-safe operations
        self.lock = threading.Lock()
        
        # Start background calibration thread
        self.calibration_thread = threading.Thread(target=self._background_calibration, daemon=True)
        self.calibration_thread.start()
        
        logging.info("Probability Calibrator initialized")
        
    def _background_calibration(self):
        """Background thread for periodic calibration"""
        while True:
            try:
                current_time = time.time()
                if (current_time - self.last_calibration_time > self.calibration_interval and 
                    len(self.calibration_data) >= 100):
                    self._update_calibrator()
                    self.last_calibration_time = current_time
                time.sleep(300)  # Check every 5 minutes
            except Exception as e:
                logging.error(f"Background calibration error: {e}")
                time.sleep(600)
                
    def calibrate_probabilities(self, features, raw_predictions, actual_outcome=None):
        """Calibrate model probabilities in real-time"""
        with self.lock:
            # Store calibration data if outcome is available
            if actual_outcome is not None:
                self.calibration_data.append({
                    'features': features,
                    'predictions': raw_predictions,
                    'actual_outcome': actual_outcome,
                    'timestamp': pd.Timestamp.now()
                })
                
            # Apply current calibration
            if self.calibrator is not None and len(self.calibration_data) >= 50:
                try:
                    calibrated_probs = self.calibrator.predict_proba([raw_predictions])[0]
                    confidence = max(calibrated_probs)
                    return calibrated_probs, confidence
                except Exception as e:
                    logging.warning(f"Calibration failed: {e}")
                    
            # Return raw predictions if calibration not available
            confidence = max(raw_predictions)
            return raw_predictions, confidence
            
    def _update_calibrator(self):
        """Update probability calibration based on recent performance"""
        if len(self.calibration_data) < 50:
            return
            
        try:
            # Prepare calibration data
            recent_data = list(self.calibration_data)[-200:]  # Use last 200 points
            
            X_cal = np.array([item['features'] for item in recent_data])
            y_cal = np.array([item['actual_outcome'] for item in recent_data if item['actual_outcome'] is not None])
            
            if len(y_cal) >= 50 and len(np.unique(y_cal)) > 1:
                self.calibrator = CalibratedClassifierCV(self.model, method='isotonic', cv=3)
                self.calibrator.fit(X_cal, y_cal)
                logging.info("Probability calibrator updated successfully")
                
        except Exception as e:
            logging.error(f"Calibrator update failed: {e}")
            
    def get_calibration_status(self):
        """Return calibration system status"""
        return {
            'calibration_data_points': len(self.calibration_data),
            'last_calibration_time': self.last_calibration_time,
            'calibrator_available': self.calibrator is not None
        }
