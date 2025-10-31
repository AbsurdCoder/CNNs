"""
REST API for CNN Model Serving
Deploy any trained model as a web service

Usage:
    uvicorn serving_api:app --host 0.0.0.0 --port 8000 --reload

Test:
    curl -X POST "http://localhost:8000/predict" \
         -H "Content-Type: multipart/form-data" \
         -F "file=@image.jpg" \
         -F "model_name=resnet50"
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import time
import json
import os

# Import model factories
from lenet import create_lenet
from alexnet import create_alexnet
from vggnet import create_vgg16
from googlenet import create_googlenet
from resnet import create_resnet50, create_resnet18
from mobilenet import create_mobilenet
from densenet import create_densenet121
from efficientnet import create_efficientnet_b0
from inceptionv3 import create_inceptionv3
from vit import create_vit_base


# Initialize FastAPI app
app = FastAPI(
    title="CNN Models API",
    description="REST API for CNN model inference and comparison",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global model cache
MODEL_CACHE = {}
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Pydantic models
class PredictionResponse(BaseModel):
    model_name: str
    predicted_class: int
    predicted_label: str
    confidence: float
    probabilities: Dict[str, float]
    inference_time_ms: float


class ModelInfo(BaseModel):
    name: str
    parameters: int
    description: str
    input_size: List[int]
    available: bool


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_time_ms: float


# Image preprocessing
def get_transform(img_size: int = 224):
    """Get image transformation pipeline"""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


# Model loading
def load_model(model_name: str, num_classes: int, checkpoint_path: Optional[str] = None) -> nn.Module:
    """Load model from cache or create new"""
    cache_key = f"{model_name}_{num_classes}"
    
    if cache_key in MODEL_CACHE:
        return MODEL_CACHE[cache_key]
    
    # Create model
    models = {
        'lenet5': lambda: create_lenet(num_classes, input_channels=3),
        'alexnet': lambda: create_alexnet(num_classes),
        'vgg16': lambda: create_vgg16(num_classes),
        'googlenet': lambda: create_googlenet(num_classes),
        'resnet50': lambda: create_resnet50(num_classes),
        'resnet18': lambda: create_resnet18(num_classes),
        'mobilenet': lambda: create_mobilenet(num_classes),
        'densenet121': lambda: create_densenet121(num_classes),
        'efficientnet_b0': lambda: create_efficientnet_b0(num_classes),
        'inceptionv3': lambda: create_inceptionv3(num_classes),
        'vit_base': lambda: create_vit_base(num_classes=num_classes),
    }
    
    if model_name not in models:
        raise ValueError(f"Model {model_name} not supported")
    
    model = models[model_name]()
    
    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model.to(DEVICE)
    model.eval()
    
    # Cache the model
    MODEL_CACHE[cache_key] = model
    
    return model


def predict_image(model: nn.Module, image: Image.Image, class_names: List[str]) -> Dict:
    """Predict single image"""
    # Preprocess
    transform = get_transform()
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    # Inference
    start_time = time.time()
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    
    inference_time = (time.time() - start_time) * 1000  # ms
    
    # Get predictions
    confidence, predicted = torch.max(probabilities, 1)
    predicted_class = predicted.item()
    predicted_label = class_names[predicted_class] if class_names else str(predicted_class)
    
    # Get top-5 probabilities
    top5_prob, top5_indices = torch.topk(probabilities[0], min(5, len(probabilities[0])))
    top5_dict = {
        (class_names[idx.item()] if class_names else str(idx.item())): prob.item()
        for idx, prob in zip(top5_indices, top5_prob)
    }
    
    return {
        'predicted_class': predicted_class,
        'predicted_label': predicted_label,
        'confidence': confidence.item(),
        'probabilities': top5_dict,
        'inference_time_ms': inference_time
    }


# API Endpoints

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "CNN Models API",
        "version": "1.0.0",
        "endpoints": {
            "/models": "List available models",
            "/predict": "Predict single image",
            "/predict/batch": "Predict multiple images",
            "/compare": "Compare models on single image",
            "/health": "Health check"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "device": str(DEVICE),
        "models_cached": len(MODEL_CACHE)
    }


@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List all available models"""
    models_info = [
        {"name": "lenet5", "parameters": 60000, "description": "LeNet-5 (1998)", "input_size": [3, 32, 32], "available": True},
        {"name": "alexnet", "parameters": 60000000, "description": "AlexNet (2012)", "input_size": [3, 224, 224], "available": True},
        {"name": "vgg16", "parameters": 138000000, "description": "VGG-16 (2014)", "input_size": [3, 224, 224], "available": True},
        {"name": "googlenet", "parameters": 6000000, "description": "GoogLeNet (2014)", "input_size": [3, 224, 224], "available": True},
        {"name": "resnet50", "parameters": 25600000, "description": "ResNet-50 (2015)", "input_size": [3, 224, 224], "available": True},
        {"name": "resnet18", "parameters": 11700000, "description": "ResNet-18 (2015)", "input_size": [3, 224, 224], "available": True},
        {"name": "mobilenet", "parameters": 4200000, "description": "MobileNet (2017)", "input_size": [3, 224, 224], "available": True},
        {"name": "densenet121", "parameters": 8000000, "description": "DenseNet-121 (2017)", "input_size": [3, 224, 224], "available": True},
        {"name": "efficientnet_b0", "parameters": 5300000, "description": "EfficientNet-B0 (2019)", "input_size": [3, 224, 224], "available": True},
        {"name": "inceptionv3", "parameters": 24000000, "description": "Inception-v3 (2015)", "input_size": [3, 299, 299], "available": True},
        {"name": "vit_base", "parameters": 86000000, "description": "ViT-Base (2020)", "input_size": [3, 224, 224], "available": True},
    ]
    return models_info


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    model_name: str = Form("resnet50"),
    num_classes: int = Form(1000),
    checkpoint_path: Optional[str] = Form(None),
    class_names: Optional[str] = Form(None)
):
    """
    Predict single image
    
    Args:
        file: Image file
        model_name: Model architecture name
        num_classes: Number of output classes
        checkpoint_path: Optional path to model checkpoint
        class_names: Optional JSON string of class names
    """
    try:
        # Load image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Parse class names
        classes = json.loads(class_names) if class_names else None
        
        # Load model
        model = load_model(model_name, num_classes, checkpoint_path)
        
        # Predict
        result = predict_image(model, image, classes)
        
        return PredictionResponse(
            model_name=model_name,
            **result
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    files: List[UploadFile] = File(...),
    model_name: str = Form("resnet50"),
    num_classes: int = Form(1000),
    class_names: Optional[str] = Form(None)
):
    """Predict multiple images"""
    try:
        start_time = time.time()
        
        # Parse class names
        classes = json.loads(class_names) if class_names else None
        
        # Load model once
        model = load_model(model_name, num_classes)
        
        # Process all images
        predictions = []
        for file in files:
            image_bytes = await file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            result = predict_image(model, image, classes)
            predictions.append(PredictionResponse(model_name=model_name, **result))
        
        total_time = (time.time() - start_time) * 1000
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_time_ms=total_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/compare")
async def compare_models(
    file: UploadFile = File(...),
    model_names: str = Form("resnet50,mobilenet,efficientnet_b0"),
    num_classes: int = Form(1000),
    class_names: Optional[str] = Form(None)
):
    """
    Compare multiple models on single image
    
    Args:
        file: Image file
        model_names: Comma-separated model names
        num_classes: Number of output classes
        class_names: Optional JSON string of class names
    """
    try:
        # Load image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Parse inputs
        models_list = [m.strip() for m in model_names.split(',')]
        classes = json.loads(class_names) if class_names else None
        
        # Compare models
        results = {}
        for model_name in models_list:
            try:
                model = load_model(model_name, num_classes)
                result = predict_image(model, image, classes)
                results[model_name] = result
            except Exception as e:
                results[model_name] = {"error": str(e)}
        
        # Add rankings
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        
        if valid_results:
            # Fastest model
            fastest = min(valid_results.items(), key=lambda x: x[1]['inference_time_ms'])
            # Most confident
            most_confident = max(valid_results.items(), key=lambda x: x[1]['confidence'])
            
            results['rankings'] = {
                'fastest': {'model': fastest[0], 'time_ms': fastest[1]['inference_time_ms']},
                'most_confident': {'model': most_confident[0], 'confidence': most_confident[1]['confidence']}
            }
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/cache")
async def clear_cache():
    """Clear model cache"""
    global MODEL_CACHE
    MODEL_CACHE.clear()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return {"message": "Cache cleared", "device": str(DEVICE)}


@app.get("/cache/info")
async def cache_info():
    """Get cache information"""
    return {
        "cached_models": list(MODEL_CACHE.keys()),
        "cache_size": len(MODEL_CACHE),
        "device": str(DEVICE)
    }


# Example client code
EXAMPLE_CLIENT_CODE = """
# Python client example
import requests

# Single prediction
with open('image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/predict',
        files={'file': f},
        data={
            'model_name': 'resnet50',
            'num_classes': 1000
        }
    )
print(response.json())

# Compare models
with open('image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/compare',
        files={'file': f},
        data={
            'model_names': 'resnet50,mobilenet,efficientnet_b0',
            'num_classes': 1000
        }
    )
print(response.json())

# JavaScript client example
const formData = new FormData();
formData.append('file', imageFile);
formData.append('model_name', 'resnet50');

fetch('http://localhost:8000/predict', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => console.log(data));
"""


if __name__ == "__main__":
    import uvicorn
    print("Starting CNN Models API...")
    print(f"Device: {DEVICE}")
    print("\nExample usage:")
    print(EXAMPLE_CLIENT_CODE)
    uvicorn.run(app, host="0.0.0.0", port=8000)