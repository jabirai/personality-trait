from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from openpyxl import load_workbook
import os

app = Flask(__name__)

# Get the current working directory
working_dir = os.getcwd()

model_path = os.path.join(working_dir, 'model', 'my_pipeline.pkl')
dataset_path = os.path.join(working_dir, 'dataset', 'combined_hvp_numeric.xlsx')


# Load the pre-trained model pipeline (which includes the vectorizer, scaler, and model)
pipeline = joblib.load(model_path)  # Adjust this path if needed

# Load dataset for column names (traits)
dataset = pd.read_excel(dataset_path)  # Adjust path if needed


Traits = dataset.drop(columns=['Message', 'ID'])
trait_names = list(Traits.columns)

# Initialize MinMaxScaler and PCA as in your `app.py`
# scaler = MinMaxScaler()
# pca = PCA(n_components=20)
y = Traits.values
scaler = MinMaxScaler()

# Fit the scaler to the target data and transform the target values (normalization). Scaling the targets to a range [0, 1]
y = scaler.fit_transform(y)

# Perform PCA (Principal Component Analysis) to reduce the dimensionality of the target space 'n_components=20' means we reduce the target to 20 principal components (important features)
pca = PCA(n_components=20)
pca.fit(y)

# Function to scale back the predictions to the original range (inverse scaling)
def scale_back(scaler: MinMaxScaler, vector):
    return scaler.inverse_transform(vector.reshape(1, -1))

# Route for the /predict endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request (expected to be text)
    data = request.get_json()
    input_text = data['text']  # Ensure the client sends 'text' as input
    
    # Use the pipeline to predict the traits for the input text
    y_pred_reduced = pipeline.predict([input_text])
    
    # Apply PCA inverse transformation to recover the full traits from the reduced ones
    y_pred_full = pca.inverse_transform(y_pred_reduced)[0]
    
    # Scale back to original range
    y_pred_full = np.array([float(score) for score in y_pred_full])
    y_pred_full_scaled = scale_back(scaler, y_pred_full)[0]
    
    # Create a dictionary of predicted trait values
    trait_dict = dict(zip(trait_names, [round(trait, 4) for trait in y_pred_full_scaled]))
    
    # Return the prediction as a JSON response
    return jsonify({'predictions': trait_dict})

# Home route (basic check)
@app.route('/')
def home():
    return "Welcome to the Flask API!"

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)