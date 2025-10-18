import os, io, numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf

# ========= Konfigurasi dasar =========
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # Redam log TF
IMG_SIZE = 224
MODEL_PATH = os.getenv("MODEL_PATH", "tomato_best.keras")
THRESHOLD_UNKNOWN = float(os.getenv("THRESHOLD_UNKNOWN", "0.55"))  # 0.5–0.7 umumnya bagus

# Label asli (URUTAN HARUS SAMA dengan saat training)
LABEL_EN = [
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy',
    'Unknown',  # kelas tambahan
]

# Terjemahan label ke Indonesia
LABEL_ID = {
    'Tomato___Bacterial_spot': 'Bercak Bakteri',
    'Tomato___Early_blight': 'Hawar Daun Dini',
    'Tomato___Late_blight': 'Hawar Daun (Late blight)',
    'Tomato___Leaf_Mold': 'Jamur Daun (Leaf Mold)',
    'Tomato___Septoria_leaf_spot': 'Bercak Daun Septoria',
    'Tomato___Spider_mites Two-spotted_spider_mite': 'Tungau Laba-laba Dua Bintik',
    'Tomato___Target_Spot': 'Bercak Target',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'Virus Kuning Keriting Daun Tomat (TYLCV)',
    'Tomato___Tomato_mosaic_virus': 'Virus Mosaik Tomat (ToMV)',
    'Tomato___healthy': 'Sehat',
    'Unknown': 'Bukan daun tomat / Tidak dikenali',
}

# ========= App & CORS =========
app = Flask(__name__)
# Izinkan semua origin (kalau mau lebih ketat, ganti origins dengan domain Anda)
CORS(app, resources={r"/predict": {"origins": "*"}, r"/health": {"origins": "*"}})

# ========= Muat model =========
model = tf.keras.models.load_model(MODEL_PATH)

# ========= Util =========
def preprocess(pil_img: Image.Image):
    pil_img = pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    x = np.array(pil_img) / 255.0
    return np.expand_dims(x, 0)

# ========= Routes =========
@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok", "model": MODEL_PATH, "classes": len(LABEL_EN)}, 200

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "Harap kirim gambar pada field form-data bernama 'file'."}), 400

    try:
        img_bytes = request.files["file"].read()
        pil_img = Image.open(io.BytesIO(img_bytes))
    except Exception as e:
        return jsonify({"error": f"Gagal membaca gambar: {e}"}), 400

    x = preprocess(pil_img)
    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    label_en = LABEL_EN[idx]
    label_id = LABEL_ID.get(label_en, label_en)
    konf = float(probs[idx])

    # top-3
    urut = np.argsort(probs)[::-1][:3]
    top3 = [
        {
            "label_en": LABEL_EN[i],
            "label_id": LABEL_ID.get(LABEL_EN[i], LABEL_EN[i]),
            "confidence": float(probs[i])
        }
        for i in urut
    ]

    # Logika Unknown / tolak jika di bawah threshold
    is_unknown = False

    # Kalau model benar-benar prediksi Unknown
    if label_en == "Unknown":
       is_unknown = True
    # Kalau model yakin < threshold → tandai unknown
    elif konf < THRESHOLD_UNKNOWN:
       is_unknown = True


    if is_unknown:
        return jsonify({
            "label_id": LABEL_ID['Unknown'],
            "label_en": "Unknown",
            "confidence": konf,
            "top3": top3,
            "is_unknown": True,
            "threshold_used": THRESHOLD_UNKNOWN
        })

    return jsonify({
        "label_id": label_id,
        "label_en": label_en,
        "confidence": konf,
        "top3": top3,
        "is_unknown": False,
        "threshold_used": THRESHOLD_UNKNOWN
    })

if __name__ == "__main__":
    # Jalankan:  python app.py
    # Ubah port jika perlu. n8n/ngrok biasa memakai 8000.
    app.run(host="0.0.0.0", port=8000)
