from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Memuat model
model = tf.keras.models.load_model('')

# Endpoint untuk prediksi

@app.route('/prediksi', methods=['POST'])
def prediksi_rating():
    # Mendapatkan data JSON dari permintaan
    data = request.json

    # Mengekstrak user_feature dari data permintaan
    user_feature = np.array(data['user_feature'])

    # Mengekstrak tempat_feature dari data permintaan
    tempat_feature = np.array(data['tempat_feature'])

    # Menduplikasi user_feature untuk setiap tempat
    user_feature = np.repeat(user_feature, tempat_feature.shape[0], axis=0)

    # Menggabungkan user_feature dan tempat_feature
    features = np.concatenate((np.expand_dims(
        user_feature, axis=0), np.expand_dims(tempat_feature, axis=0)), axis=0)

    # Melakukan prediksi menggunakan model yang dimuat
    ratings = model.predict(features)

    # Mengurutkan rating secara menurun dan mendapatkan indeksnya
    sorted_indices = np.argsort(ratings)[::-1]

    # Mengembalikan indeks yang sudah diurutkan sebagai hasil prediksi
    predictions = sorted_indices.tolist()
    return jsonify(predictions)


if __name__ == '__main__':
    app.run()
