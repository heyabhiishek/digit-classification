from pathlib import Path
import io

import numpy as np
from PIL import Image

import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model" / "mnist_model.keras"
STATIC_DIR = BASE_DIR / "static"


# --------------------------------------------------
# FASTAPI
# --------------------------------------------------

app = FastAPI(title="Handwritten Digit Classifier")

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.")


app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static"
)


# --------------------------------------------------
# HOME
# --------------------------------------------------

@app.get("/")
async def home():
    return FileResponse(STATIC_DIR / "index.html")


# --------------------------------------------------
# PREPROCESS IMAGE
# --------------------------------------------------

def preprocess_image(image_bytes: bytes):

    # Load image
    image = Image.open(io.BytesIO(image_bytes)).convert("L")

    image_array = np.array(image, dtype=np.uint8)

    # --------------------------------------------------
    # 1. Ensure black background / white digit
    # --------------------------------------------------

    # Canvas should already be black background.
    # Only invert if the image is actually mostly white.
    if image_array.mean() > 127:
        image_array = 255 - image_array

    # --------------------------------------------------
    # 2. Remove very faint pixels
    # --------------------------------------------------

    image_array[image_array < 30] = 0

    # --------------------------------------------------
    # 3. Find digit
    # --------------------------------------------------

    coords = np.argwhere(image_array > 0)

    if coords.size == 0:
        return np.zeros((1, 28, 28), dtype=np.float32)

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    cropped = image_array[
        y_min:y_max + 1,
        x_min:x_max + 1
    ]

    # --------------------------------------------------
    # 4. Resize digit
    # --------------------------------------------------

    cropped_image = Image.fromarray(cropped)

    width, height = cropped_image.size

    target_size = 20

    scale = target_size / max(width, height)

    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))

    cropped_image = cropped_image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS
    )

    resized = np.array(
        cropped_image,
        dtype=np.float32
    )

    # --------------------------------------------------
    # 5. Create 28×28 image
    # --------------------------------------------------

    final = np.zeros((28, 28), dtype=np.float32)

    # Initial geometric placement
    x = (28 - new_width) // 2
    y = (28 - new_height) // 2

    final[
        y:y + new_height,
        x:x + new_width
    ] = resized

    # --------------------------------------------------
    # 6. Center based on center of mass
    # --------------------------------------------------

    total = final.sum()

    if total > 0:

        yy, xx = np.indices((28, 28))

        center_x = (xx * final).sum() / total
        center_y = (yy * final).sum() / total

        target_center = 13.5

        shift_x = int(round(target_center - center_x))
        shift_y = int(round(target_center - center_y))

        centered = np.zeros_like(final)

        src_x1 = max(0, -shift_x)
        src_x2 = min(28, 28 - shift_x)

        src_y1 = max(0, -shift_y)
        src_y2 = min(28, 28 - shift_y)

        dst_x1 = max(0, shift_x)
        dst_x2 = dst_x1 + (src_x2 - src_x1)

        dst_y1 = max(0, shift_y)
        dst_y2 = dst_y1 + (src_y2 - src_y1)

        centered[
            dst_y1:dst_y2,
            dst_x1:dst_x2
        ] = final[
            src_y1:src_y2,
            src_x1:src_x2
        ]

        final = centered

    # --------------------------------------------------
    # 7. Normalize exactly like MNIST training
    # --------------------------------------------------

    final /= 255.0

    # --------------------------------------------------
    # 8. Add batch dimension
    # --------------------------------------------------

    return final.reshape(1, 28, 28).astype(np.float32)

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:

        # Read uploaded image
        image_bytes = await file.read()

        # Preprocess
        image_array = preprocess_image(image_bytes)

        debug_image = (
            image_array[0] * 255
        ).astype(np.uint8)

        Image.fromarray(debug_image).resize(
            (280, 280),
            Image.Resampling.NEAREST
        ).save(
            BASE_DIR / "debug.png"
        )

        # Prediction
        probabilities = model.predict(
            image_array,
            verbose=0
        )[0]

        # Predicted digit
        digit = int(np.argmax(probabilities))

        # Confidence
        confidence = float(
            probabilities[digit]
        )

        # All probabilities
        all_predictions = [
            {
                "digit": i,
                "probability": round(
                    float(probabilities[i]) * 100,
                    2
                )
            }
            for i in range(10)
        ]

        return {
            "digit": digit,
            "confidence": round(
                confidence * 100,
                2
            ),
            "probabilities": all_predictions
        }

    except Exception as e:

        return {
            "error": str(e)
        }