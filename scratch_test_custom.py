import joblib
import server  # to use preprocess function

tfidf = joblib.load('tfidf_model.pkl')
svm = joblib.load('svm_model.pkl')

texts = [
    "I would highly recommend this item. We love this blanket.",
    "These are so flimsy! They are not the quality you would expect from a piece of furniture."
]

for t in texts:
    final_text, _ = server.preprocess(t, translate=False)
    vec = tfidf.transform([final_text])
    p = svm.predict(vec)[0]
    print(f"[{p}] - {t}")
