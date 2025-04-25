from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from app.predict import predict_image

app = FastAPI()

# CORS (adjust origin for your Expo app if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Save uploaded image temporarily
    image_path = f"temp_{file.filename}"
    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run prediction
    try:
        predictions = predict_image(image_path)
    finally:
        os.remove(image_path)

    return {"predictions": predictions}
if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
