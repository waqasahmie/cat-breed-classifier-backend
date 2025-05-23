from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from app.predict import predict_image
import uvicorn  # Import uvicorn to run the app

app = FastAPI()

# CORS (adjust origin for your Expo app if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Health check or friendly root message
@app.get("/")
async def root():
    return {"message": "Cat Breed Classifier API is running :)"}

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

    # Check if predictions match the known classes
    known_classes = ["Abyssinian", "Bengal", "Birman", "Bombay", "BritishShorthair", 
                     "EgyptianMau", "MaineCoon", "Persian", "Ragdoll", "RussianBlue", 
                     "Siamese", "Sphynx"]

    return {"predictions": predictions}

# If run directly, use uvicorn to start the app
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # Get PORT from env variable, fallback to 8000
    uvicorn.run(app, host="0.0.0.0", port=port)  # Run the app with Uvicorn
