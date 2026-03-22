from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = FastAPI(title="Leaf Detection API")

# ===== DEBUG STARTUP =====
print("🚀 Starting FastAPI server...")
print("📂 Current directory:", os.getcwd())
print("📂 Files in directory:", os.listdir())

# ===== LOAD MODEL SAFELY =====
MODEL_PATH = "healthy_unhealthy_model.h5"

try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ Model file not found: {MODEL_PATH}")

    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully")

except Exception as e:
    print("❌ Error loading model:", str(e))
    raise e  # This will show error in Render logs

# ===== PREDICTION FUNCTION =====
def predict_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    img = np.array(image) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)[0][0]

    if prediction >= 0.5:
        return "Unhealthy", float(prediction * 100)
    else:
        return "Healthy", float((1 - prediction) * 100)

# ===== ROOT ROUTE (for testing) =====
@app.get("/")
def home():
    return {"message": "Leaf Detection API is running 🚀"}

# ===== API ENDPOINT =====
@app.post("/api/leaf")
async def detect_leaf(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        label, confidence = predict_image(image)

        return {
            "label": label,
            "confidence": confidence
        }

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
