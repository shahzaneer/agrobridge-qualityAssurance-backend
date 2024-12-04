from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
import tensorflow as tf
import numpy as np
from PIL import Image, UnidentifiedImageError
import io
from keras.layers import Dropout
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define your custom FixedDropout layer
class FixedDropout(Dropout):
    def __init__(self, rate, **kwargs):
        super(FixedDropout, self).__init__(rate, **kwargs)

# Initialize FastAPI app
app = FastAPI()

# Add middleware for CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models with custom_objects for FixedDropout
try:
    models = {
        "apple": tf.keras.models.load_model(
            "models/appleGradingModel.h5", 
            custom_objects={'FixedDropout': FixedDropout}
        ),
        "banana": tf.keras.models.load_model(
            "models/gradingbanana.h5", 
            custom_objects={'FixedDropout': FixedDropout}
        ),
        "orange": tf.keras.models.load_model(
            "models/gradingorange.h5", 
            custom_objects={'FixedDropout': FixedDropout}
        )
    }
    logging.info("Models loaded successfully.")
except Exception as e:
    logging.error(f"Error loading models: {e}")
    raise RuntimeError("Failed to load models. Check model paths or configurations.")

# Prepare your image for prediction
def prepare_image(image: Image.Image) -> np.ndarray:
    image = image.resize((100, 100))  # Resize to your model's expected input size
    image_array = np.array(image) / 255.0  # Normalize the image
    return np.expand_dims(image_array, axis=0)  # Add batch dimension

# Class mappings for each fruit
class_mappings = {
    "apple": {
        0: "Yellow Apples",
        1: "Red Apples Grade 1",
        2: "Red Apples Grade 2",
        3: "Green Apples"
    },
    "banana": {
        0: "Rotten Bananas",
        1: "Grade 2 Bananas (unripe)",
        2: "Grade 1 Bananas"
    },
    "orange": {
        0: "Rotten Oranges",
        1: "Grade 2 Oranges",
        2: "Grade 1 Oranges"
    }
}

# Common prediction handler
async def predict_fruit(fruit_type: str, file: UploadFile):
    try:
        # Ensure the uploaded file is an image
        if file.content_type not in ["image/jpeg", "image/png"]:
            raise HTTPException(status_code=400, detail="Invalid file type. Only JPEG and PNG are supported.")

        image = Image.open(io.BytesIO(await file.read()))
        img_array = prepare_image(image)
        
        # Get the model and class mapping for the specific fruit
        model = models[fruit_type]
        classes = class_mappings[fruit_type]
        
        # Make prediction
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)
        predicted_probability = predictions[0][predicted_class]  # Get the probability of the predicted class
        
        # Define a confidence threshold
        threshold = 0.6  # You can adjust this value as needed
        
        # Check if the predicted probability meets the threshold
        if predicted_probability < threshold:
            return JSONResponse(content={
                "fruit": fruit_type,
                "category": f"Not a confidently recognized {fruit_type}",
                "prediction_values": predictions[0].tolist()
            }, status_code=400)
        
        # Prepare response with confident prediction
        response = {
            "fruit": fruit_type,
            "category": classes.get(predicted_class, f"{fruit_type}"),
            "prediction_values": predictions[0].tolist()
        }
        logging.info(f"Prediction successful for {fruit_type}: {response}")
        return JSONResponse(content=response)
    
    except UnidentifiedImageError:
        logging.error("Uploaded file is not a valid image.")
        raise HTTPException(status_code=400, detail="Invalid image file.")
    except Exception as e:
        logging.error(f"Unexpected error during prediction: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing the prediction.")

# Define routes for different fruits
@app.post("/predict/apples/")
async def predict_apples(file: UploadFile = File(...)):
    return await predict_fruit("apple", file)

@app.post("/predict/bananas/")
async def predict_bananas(file: UploadFile = File(...)):
    return await predict_fruit("banana", file)

@app.post("/predict/oranges/")
async def predict_oranges(file: UploadFile = File(...)):
    return await predict_fruit("orange", file)

# Add global exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logging.warning(f"HTTP error occurred: {exc.detail}")
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logging.warning(f"Validation error: {exc.errors()}")
    return JSONResponse(status_code=422, content={"error": "Invalid request parameters."})

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logging.error(f"Unhandled error: {exc}")
    return JSONResponse(status_code=500, content={"error": "An internal server error occurred."})
