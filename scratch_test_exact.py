import joblib
import server  # for preprocess

tfidf = joblib.load('tfidf_model.pkl')
svm = joblib.load('svm_model.pkl')

text = "I would highly recommend this item. We love this product. It is exactly what you would expect."
final_text, _ = server.preprocess(text, translate=False)
vec = tfidf.transform([final_text])

label = svm.predict(vec)[0]
score = svm.decision_function(vec)[0]
print(f"Text: {text}")
print(f"Label: {label}")
print(f"Decision Function Score (Distance to hyperplane): {score}")
print(f"Classes: {svm.classes_}")

# Let's see the contribution of each word
indices = vec.indices
values = vec.data
coef = svm.coef_[0] if len(svm.coef_.shape) == 2 else svm.coef_
feature_names = tfidf.get_feature_names_out()

print("\nFeature contributions:")
for idx, val in zip(indices, values):
    s = val * coef[idx]
    name = feature_names[idx].split('__')[-1] if '__' in feature_names[idx] else feature_names[idx]
    print(f"  {name}: {s:.4f}")

