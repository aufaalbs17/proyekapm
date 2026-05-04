import joblib
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, '../models')

svm_model = joblib.load(os.path.join(MODEL_DIR, 'svm_model.pkl'))
print(f"coef shape: {svm_model.coef_.shape}")
print(f"intercept shape: {svm_model.intercept_.shape}")
