import joblib

svm_model = joblib.load('svm_model.pkl')
print(f"coef shape: {svm_model.coef_.shape}")
print(f"intercept shape: {svm_model.intercept_.shape}")
