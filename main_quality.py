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


def generate_quality_statement(fruit_type: str, predicted_class: int, predicted_probability: float):
    if fruit_type == "apple":
        # Quality statement for apple
        if predicted_class == 0:  # Yellow Apple
            if predicted_probability > 0.8:
                return f"The apple is of Yellow quality, with a very high confidence of {predicted_probability*100:.2f}%. It is fresh and ripe."
            elif predicted_probability > 0.65:
                return f"The apple is likely Yellow, with a moderate confidence of {predicted_probability*100:.2f}%. It may still be good but might not be at its peak."
            else:
                return f"The apple might be Yellow, but the confidence is low at {predicted_probability*100:.2f}%. It may not be very fresh."
        
        elif predicted_class == 1:  # Red Apple (Grade 1)
            if predicted_probability >= 0.8:
                return f"The apple is of Grade 1 Red quality, with a very high confidence of {predicted_probability*100:.2f}%. It's fresh, ripe, and highly recommended."
            elif predicted_probability >= 0.65:
                return f"The apple is likely Grade 1 Red, with a moderate confidence of {predicted_probability*100:.2f}%. It may be ripe, but further inspection is suggested."
            else:
                return f"The apple might be Grade 1 Red, but the confidence is low at {predicted_probability*100:.2f}%. Proceed with caution, as it may not be of the highest quality."
        
        elif predicted_class == 2:  # Grade 2 Red Apple
            if predicted_probability >= 0.8:
                return f"The apple is of Grade 2 Red quality, with a very high confidence of {predicted_probability*100:.2f}%. It is still of decent quality, but may not be as fresh."
            elif predicted_probability >= 0.65:
                return f"The apple is likely Grade 2 Red, with a moderate confidence of {predicted_probability*100:.2f}%. It may not be very fresh, but still usable."
            else:
                return f"The apple is likely Grade 2 Red, but the confidence is low at {predicted_probability*100:.2f}%. It's likely not fresh."
        
        elif predicted_class == 3:  # Green Apple
            if predicted_probability >= 0.8:
                return f"The apple is of Green quality, with a very high confidence of {predicted_probability*100:.2f}%. It may be under-ripe and firm, but very fresh."
            elif predicted_probability >= 0.65:
                return f"The apple is likely Green, with a moderate confidence of {predicted_probability*100:.2f}%. It is probably firm and not ripe yet."
            else:
                return f"The apple might be Green, with a low confidence of {predicted_probability*100:.2f}%. It may still be too firm and unripe."

    elif fruit_type == "banana":
        # Quality statement for banana
        if predicted_class == 0:  # Rotten Banana
            if predicted_probability >= 0.8:
                return f"The banana is rotten, with a very high confidence of {predicted_probability*100:.2f}%. It's past the point of consumption."
            elif predicted_probability >= 0.65:
                return f"The banana is likely rotten, with a moderate confidence of {predicted_probability*100:.2f}%. It may not be safe to eat."
            else:
                return f"The banana might be rotten, with a low confidence of {predicted_probability*100:.2f}%. It's likely not fit for consumption."
        
        elif predicted_class == 1:  # Grade 2 (Unripe Banana)
            if predicted_probability >= 0.8:
                return f"The banana is of Grade 2 quality (unripe), with a very high confidence of {predicted_probability*100:.2f}%. It's firm and not yet ready for consumption."
            elif predicted_probability >= 0.65:
                return f"The banana is likely Grade 2 (unripe), with a moderate confidence of {predicted_probability*100:.2f}%. It may need a few more days to ripen."
            else:
                return f"The banana might be Grade 2 (unripe), with a low confidence of {predicted_probability*100:.2f}%. It is not ripe enough yet."
        
        elif predicted_class == 2:  # Grade 1 (Ripe Banana)
            if predicted_probability >= 0.8:
                return f"The banana is of Grade 1 quality, with a very high confidence of {predicted_probability*100:.2f}%. It's ripe and perfect for consumption."
            elif predicted_probability >= 0.65:
                return f"The banana is likely Grade 1 quality, with a moderate confidence of {predicted_probability*100:.2f}%. It seems ripe, but may require inspection."
            else:
                return f"The banana might be Grade 1, but the confidence is low at {predicted_probability*100:.2f}%. It's likely ripe but not guaranteed."

    elif fruit_type == "orange":
        # Quality statement for orange
        if predicted_class == 0:  # Rotten Orange
            if predicted_probability >= 0.8:
                return f"The orange is rotten, with a very high confidence of {predicted_probability*100:.2f}%. It should not be eaten."
            elif predicted_probability >= 0.65:
                return f"The orange is likely rotten, with a moderate confidence of {predicted_probability*100:.2f}%. It may not be safe to consume."
            else:
                return f"The orange might be rotten, with a low confidence of {predicted_probability*100:.2f}%. It is probably not fit for consumption."
        
        elif predicted_class == 1:  # Grade 2 Orange
            if predicted_probability >= 0.8:
                return f"The orange is of Grade 2 quality, with a very high confidence of {predicted_probability*100:.2f}%. It may not be very fresh but still edible."
            elif predicted_probability >= 0.65:
                return f"The orange is likely Grade 2, with a moderate confidence of {predicted_probability*100:.2f}%. It may be overripe or not very fresh."
            else:
                return f"The orange might be Grade 2, with a low confidence of {predicted_probability*100:.2f}%. It's probably not fresh."
        
        elif predicted_class == 2:  # Grade 1 Orange
            if predicted_probability >= 0.8:
                return f"The orange is of Grade 1 quality, with a very high confidence of {predicted_probability*100:.2f}%. It is fresh, juicy, and highly recommended."
            elif predicted_probability >= 0.65:
                return f"The orange is likely Grade 1, with a moderate confidence of {predicted_probability*100:.2f}%. It seems fresh, but you may want to inspect it."
            else:
                return f"The orange might be Grade 1, but the confidence is low at {predicted_probability*100:.2f}%. It may not be of the highest quality."

    return "Unable to determine fruit quality based on the prediction values."


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
        threshold = 0.6  # 60 match then predict else out
        
        # Check if the predicted probability meets the threshold
        if predicted_probability < threshold:
            return JSONResponse(content={
                "fruit": fruit_type,
                "category": f"Not a confidently recognized {fruit_type}",
                "prediction_values": predictions[0].tolist(),
                "quality_description": generate_quality_statement(fruit_type, predicted_class, predicted_probability)
            }, status_code=400)
        
        # Prepare response with confident prediction
        response = {
            "fruit": fruit_type,
            "category": classes.get(predicted_class, f"{fruit_type}"),
            "prediction_values": predictions[0].tolist(),
            "quality_description": generate_quality_statement(fruit_type, predicted_class, predicted_probability)
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
