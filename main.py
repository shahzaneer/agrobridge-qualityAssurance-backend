from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from keras.layers import Dropout

# Define your custom FixedDropout layer
class FixedDropout(Dropout):
    def __init__(self, rate, **kwargs):
        super(FixedDropout, self).__init__(rate, **kwargs)
    # You can add any other custom functionality here if needed

app = FastAPI()

# Load models with custom_objects for FixedDropout
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
        
        return JSONResponse(content=response)
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

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

