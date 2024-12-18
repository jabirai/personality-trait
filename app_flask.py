from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import utils.utility as utls
import json
import os

app = Flask(__name__)

# Get the current working directory
working_dir = os.getcwd()

# Dictionary to map model names to file paths
models = {
    'hvp': {
        'model_path': os.path.join(working_dir, 'model', 'hvp_pipeline.pkl'),
        'dataset_path': os.path.join(working_dir, 'dataset', 'combined_hvp_numeric.xlsx'),
        'n_components': 20
    },
    'svs': {
        'model_path': os.path.join(working_dir, 'model', 'svs_pipeline.pkl'),
        'dataset_path': os.path.join(working_dir, 'dataset', 'combined_svs.xlsx'),
        'n_components': 14
    }
    ,'ocean': {
        'model_path': os.path.join(working_dir, 'model', 'ocean_pipeline.pkl'),
        'dataset_path': os.path.join(working_dir, 'dataset', 'ocean_prepared_dataset.xlsx'),
        'n_components': 3
    }
}

# Helper function to load a model and dataset
def load_model_and_data(model_name):
    model_info = models.get(model_name)

    if not model_info:
        raise ValueError(f"Model '{model_name}' not found!")

    # Load the pre-trained model pipeline
    pipeline = joblib.load(model_info['model_path'])

    # Load the dataset for column names (traits)
    dataset = pd.read_excel(model_info['dataset_path'])

    # n_components for PCA
    n_components = model_info['n_components']

    if model_name == 'svs':
        dataset = dataset[~dataset["Conformity"].isnull()]
    
    # Extract trait names (assuming dataset has 'Message' and 'ID' columns to drop)
    traits = dataset.drop(columns=['Message', 'ID'])
    trait_names = list(traits.columns)

    return pipeline, traits, trait_names, n_components

# Function to scale back the predictions to the original range (inverse scaling)
def scale_back(scaler: MinMaxScaler, vector):
    return scaler.inverse_transform(vector.reshape(1, -1))

# Route for the /predict endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request (expected to be text and model name)
    data = request.get_json()
    input_text = data['text']  # Ensure the client sends 'text' as input
    model_name = data.get('model', 'hvp')  # Default to 'hvp' if no model is specified
    user_id = data.get('id')
    
    # Load the model and dataset dynamically based on the model_name
    pipeline, traits, trait_names, n_components = load_model_and_data(model_name)

    # Initialize MinMaxScaler and PCA as in your `app.py`
    scaler = MinMaxScaler()

    # Fit the scaler to the target data and transform the target values (normalization)
    y = traits.values
    y_scaled = scaler.fit_transform(y)

    # Perform PCA (Principal Component Analysis) to reduce the dimensionality of the target space
    pca = PCA(n_components)
    pca.fit(y_scaled)

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
    # if model_name == 'svs':
    #     return jsonify({'predictions': trait_dict,'actual':svs_original_scores[str(user_id)] if svs_original_scores[str(user_id)] else "Previous record not found."})
    if model_name == 'ocean':
        return jsonify({'predictions': utls.scale_to_range(trait_dict)})
    return jsonify({'predictions': trait_dict})

# Home route (basic check)
@app.route('/')
def home():
    return "Welcome to the Personality Trait Analysis!"

# Run the Flask app
if __name__ == '__main__':
    with open(os.path.join(working_dir, 'dataset', '../dataset/original_svs_scores.json'), 'r') as file:
        svs_original_scores = json.load(file)
        file.close()
    app.run(host='0.0.0.0', port=5000,debug=True)
    