from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import joblib


app = Flask(__name__)

model_greenify = "deliverable/model.h5"
scalerTourism = joblib.load('deliverable/scalerTourism.save')
scalerUser = joblib.load('deliverable/scalerUser.save')
scalerTarget = joblib.load('deliverable/scalerTarget.save')

try:
    # Membaca model dari file .h5
    model = tf.keras.models.load_model(model_greenify)
    print("Model berhasil dimuat.")
except (OSError, IOError) as e:
    print("Gagal memuat model:", str(e))


@app.route('/prediksi', methods=['POST'])
def prediksi_rating():

    # Mendapatkan data JSON dari permintaan
    data = request.json

    # Mengekstrak user_feature dari data permintaan
    user_feature = np.array(data['user_feature'])
    user_feature = scalerUser.transform(user_feature)

    # Mengekstrak tempat_feature dari data permintaan
    tempat_feature = np.array(data['tempat_feature'])
    tempat_feature = scalerTourism.transform(tempat_feature)

    # Menduplikasi user_feature untuk setiap tempat
    user_feature = np.repeat(user_feature, tempat_feature.shape[0], axis=0)

    # Menggabungkan user_feature dan tempat_feature
    features = [user_feature, tempat_feature]

    # Melakukan prediksi menggunakan model yang dimuat
    ratings = model.predict(features)
    ratings = scalerTarget.inverse_transform(ratings)

    # Sorting berdasarkan index
    sorted_index = np.argsort(-ratings, axis=0).reshape(-1).tolist()

    return jsonify(sorted_index)


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
