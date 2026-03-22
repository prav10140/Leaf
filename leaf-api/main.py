from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI(title="Leaf Detection API")

# Load model
model = tf.keras.models.load_model("healthy_unhealthy_model.h5")

# Prediction function
def predict_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    img = np.array(image) / 255.0
    img = np.expand_dims(img, axis=0)

    p = model.predict(img)[0][0]

    if p >= 0.5:
        return "Unhealthy", float(p * 100)
    else:
        return "Healthy", float((1 - p) * 100)

# API endpoint
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
