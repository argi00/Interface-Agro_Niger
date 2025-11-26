"""
FastAPI application for tomato disease classification.

This API provides endpoints for classifying tomato leaf diseases using a PyTorch model.
"""

import io
import os
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import transforms

# Tomato disease classes
CLASSES = [
    "Bacterial_spot",
    "Early_blight",
    "Late_blight",
    "Leaf_Mold",
    "Septoria_leaf_spot",
    "Spider_mites",
    "Target_Spot",
    "Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato_mosaic_virus",
    "healthy",
]

# Image preprocessing transform
IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# Model path - can be configured via environment variable
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    str(Path(__file__).parent / "model.pt")
)

# Global model variable
model = None


def load_model():
    """Load the PyTorch model from disk."""
    global model
    if model is not None:
        return model

    if not os.path.exists(MODEL_PATH):
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.eval()
    return model


def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """Preprocess an image for model inference."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = IMAGE_TRANSFORM(image)
    return tensor.unsqueeze(0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events."""
    # Load the model on startup if available
    load_model()
    yield


app = FastAPI(
    title="Agro Niger - Tomato Disease Classifier",
    description="API for classifying tomato leaf diseases",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for the mobile app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Welcome to Agro Niger - Tomato Disease Classifier API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Upload an image to classify tomato disease",
            "/health": "GET - Check API health status",
            "/classes": "GET - List all disease classes",
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    model_loaded = model is not None
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "model_path": MODEL_PATH,
    }


@app.get("/classes")
async def get_classes():
    """Get the list of disease classes."""
    return {"classes": CLASSES}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict the disease class of a tomato leaf image.

    Args:
        file: Image file upload (JPEG, PNG, etc.)

    Returns:
        JSON with prediction result and confidence scores
    """
    # Validate file type
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )

    # Read and validate image
    try:
        image_bytes = await file.read()
        input_tensor = preprocess_image(image_bytes)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image file: {str(e)}"
        )

    # Check if model is loaded
    current_model = load_model()
    if current_model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not available. Please ensure model file exists at: {MODEL_PATH}"
        )

    # Run inference
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_tensor = input_tensor.to(device)

        with torch.no_grad():
            outputs = current_model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)

        predicted_class = CLASSES[predicted_idx.item()]
        confidence_score = confidence.item()

        # Get top 3 predictions
        top_k = min(3, len(CLASSES))
        top_probs, top_indices = torch.topk(probabilities, top_k)
        top_predictions = [
            {"class": CLASSES[idx.item()], "confidence": prob.item()}
            for prob, idx in zip(top_probs[0], top_indices[0])
        ]

        return {
            "result": {
                "pred": predicted_class,
                "confidence": confidence_score,
            },
            "prediction": predicted_class,
            "confidence": confidence_score,
            "top_predictions": top_predictions,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error during prediction: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
