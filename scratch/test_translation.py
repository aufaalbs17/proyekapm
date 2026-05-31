import requests

url = 'http://127.0.0.1:5000/predict'
text = "Barangnya sudah sampai dengan selamat, packing rapih, kualitas sesuai harga. Terima kasih seller."

for translate_flag in [True, False]:
    payload = {'text': text, 'translate': translate_flag}
    res = requests.post(url, json=payload).json()
    print(f"Translate={translate_flag}: Prediksi={res.get('result', 'error').upper()}")
    # print steps to see if it translated
    if 'steps' in res:
        print([s['step'] for s in res['steps']])
