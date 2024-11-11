from flask import Flask, request, jsonify
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from openpyxl import load_workbook
from aws_lambda_wsgi import response, request as lambda_request

# Initialize the Flask app
app = Flask(__name__)

# Load your model, dataset, etc.
working_dir = os.getcwd()
model_path = os.path.join(working_dir, 'model', 'my_pipeline.pkl')
dataset_path = os.path.join(working_dir, 'dataset', 'combined_hvp_numeric.xlsx')

pipeline = joblib.load(model_path)
dataset = pd.read_excel(dataset_path)

Traits = dataset.drop(columns=['Message', 'ID'])
trait_names = list(Traits.columns)

# Initialize MinMaxScaler and PCA
scaler = MinMaxScaler()
y = Traits.values
y = scaler.fit_transform(y)

pca = PCA(n_components=20)
pca.fit(y)

def scale_back(scaler: MinMaxScaler, vector):
    return scaler.inverse_transform(vector.reshape(1, -1))

@app.route('/')
def home():
    return "Welcome to the Flask API!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_text = data['text']
    
    y_pred_reduced = pipeline.predict([input_text])
    y_pred_full = pca.inverse_transform(y_pred_reduced)[0]
    y_pred_full = np.array([float(score) for score in y_pred_full])
    y_pred_full_scaled = scale_back(scaler, y_pred_full)[0]
    
    trait_dict = dict(zip(trait_names, [round(trait, 4) for trait in y_pred_full_scaled]))
    
    return jsonify({'predictions': trait_dict})

# Lambda entry point using aws-lambda-wsgi
def lambda_handler(event, context):
    return lambda_request(app, event, context)
