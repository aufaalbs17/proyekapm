import joblib
import numpy as np

tfidf = joblib.load('tfidf_model.pkl')
svm = joblib.load('svm_model.pkl')

feature_names = tfidf.get_feature_names_out()
coefs = svm.coef_  # This is the 1D array

top_fake_indices = np.argsort(coefs)[:30]
top_real_indices = np.argsort(coefs)[-30:]

print("\n--- TOP 30 CIRI-CIRI FAKE (CG) ---")
for idx in top_fake_indices:
    name = feature_names[idx].split('__')[-1] if '__' in feature_names[idx] else feature_names[idx]
    print(f"{name}: {coefs[idx]:.4f}")

print("\n--- TOP 30 CIRI-CIRI ASLI (OR) ---")
for idx in reversed(top_real_indices):
    name = feature_names[idx].split('__')[-1] if '__' in feature_names[idx] else feature_names[idx]
    print(f"{name}: {coefs[idx]:.4f}")
