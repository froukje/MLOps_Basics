from fastapi import FastAPI
from inference_onnx import ColaONNXPredictor
app = FastAPI(title="MLOps Basics App")

# load the model
predictor = ColaONNXPredictor("./models/model.onnx")


@app.get("/")
async def home_page():
    return "<h2>Sample prediction API</h2>"


@app.get("/predict")
async def home(text: str):
    result = predictor.predict(text)
    return result
