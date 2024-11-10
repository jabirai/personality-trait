import json
import joblib
import numpy as np
import pandas as pd
import boto3
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib  # joblib for model loading
import os


# Fetch environment variables
MODEL_BUCKET_NAME = os.getenv('MODEL_BUCKET_NAME')
MODEL_FILE_KEY = os.getenv('MODEL_FILE_KEY')
DATASET_FILE_KEY = os.getenv('DATASET_FILE_KEY')


# AWS S3 client to fetch model and dataset files from S3
s3_client = boto3.client('s3')

# Function to download files from S3
def download_from_s3(bucket_name, file_key, download_path):
    try:
        s3_client.download_file(bucket_name, file_key, download_path)
        print(f"Downloaded {file_key} to {download_path}")
    except Exception as e:
        print(f"Error downloading {file_key}: {str(e)}")
        raise

# Load the model and dataset from S3
def load_model_and_data():
    model_path = '/tmp/my_pipeline.pkl'
    download_from_s3(MODEL_BUCKET_NAME, MODEL_FILE_KEY, model_path)
    dataset_path = '/tmp/combined_hvp_numeric.xlsx'
    download_from_s3(MODEL_BUCKET_NAME, DATASET_FILE_KEY, dataset_path)
    dataset = pd.read_excel(dataset_path)

    Traits = dataset.drop(columns=['Message', 'ID'])
    y = Traits.values
    scaler = MinMaxScaler()
    y = scaler.fit_transform(y)
    pipeline = joblib.load(model_path)
    pca = PCA(n_components=20)
    pca.fit(y)
    
    return pipeline, Traits, scaler, pca

# Inverse scaling function
def scale_back(scaler, vector):
    return scaler.inverse_transform(vector.reshape(1, -1))

# Function to make predictions based on input text
def predict_traits(input_text, pipeline, pca, scaler, Traits):
    # Predict the traits using the pipeline
    y_pred_reduced = pipeline.predict([input_text])

    # Inverse PCA to get the full trait predictions
    y_pred_full = pca.inverse_transform(y_pred_reduced)[0]
    y_pred_full = np.array([float(score) for score in y_pred_full])
    
    # Get the trait names from the dataset columns
    trait_names = list(Traits.columns)
    
    # Create a dictionary of predicted trait values
    trait_dict = dict(zip(trait_names, [round(float(trait), 4) for trait in scale_back(scaler, y_pred_full)[0]]))
    
    return trait_dict, y_pred_full

# Lambda handler function to be triggered by API Gateway
def lambda_handler(event, context):
    # Load model and data (load from S3 each time the function runs)
    pipeline, Traits, scaler, pca = load_model_and_data()

    # Extract the user input from the API Gateway event (POST request body)
    input_text = json.loads(event['body'])['text']
    
    # Predict the traits
    y, y_out = predict_traits(input_text, pipeline, pca, scaler, Traits)
    
    # Return the prediction in the response
    response = {
        "statusCode": 200,
        "body": json.dumps({
            "predictions": y,
            "predicted_scores": y_out.tolist()
        })
    }
    
    return response
