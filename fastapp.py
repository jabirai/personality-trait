import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

dataset = pd.read_excel('combined_hvp_numeric.xlsx')
Traits = dataset.drop(columns=['Message', 'ID'])
y = Traits.values
scaler = MinMaxScaler()
y = scaler.fit_transform(y)
pca = PCA(n_components=20)
pca.fit(y)


pipeline = joblib.load('my_pipeline.pkl')


app = FastAPI()

class PredictionRequest(BaseModel):
    input_text: str

def scale_back(scaler: MinMaxScaler, vector):
    return scaler.inverse_transform(vector.reshape(1, -1))


def predict_traits(input_text):
    y_pred_reduced = pipeline.predict([input_text])
    y_pred_full = pca.inverse_transform(y_pred_reduced)[0]
    y_pred_full = np.array([float(score) for score in y_pred_full])
    trait_names = list(Traits.columns)
    scaled_y_pred = [round(float(trait), 4) for trait in scale_back(scaler, y_pred_full)[0]]
    trait_dict = dict(zip(trait_names, scaled_y_pred))
    return trait_dict, scaled_y_pred

@app.post("/predict")
def predict(request: PredictionRequest):
    input_text = request.input_text
    if not input_text:
        raise HTTPException(status_code=400, detail="Input text cannot be empty")
    trait_dict, y_pred_full = predict_traits(input_text)
    return {"traits": trait_dict, "raw_output": y_pred_full}
