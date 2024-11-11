# Import necessary libraries
import streamlit as st  
import plotly.graph_objs as go
import joblib  
from sklearn.decomposition import PCA  
import numpy as np  
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error  
from xgboost import XGBRegressor 
from sklearn.preprocessing import MinMaxScaler 

# Load the dataset from an Excel file
dataset = pd.read_excel('./dataset/combined_hvp_numeric.xlsx')

# Extract the traits from the dataset (dropping the 'Message' and 'ID' columns)
Traits = dataset.drop(columns=['Message', 'ID'])

# The target variable 'y' is the traits we are predicting
y = Traits.values

# Initialize the MinMaxScaler for scaling the target values
scaler = MinMaxScaler()

# Fit the scaler to the target data and transform the target values (normalization). Scaling the targets to a range [0, 1]
y = scaler.fit_transform(y)

# Perform PCA (Principal Component Analysis) to reduce the dimensionality of the target space. 'n_components=20' means we reduce the target to 20 principal components (important features)
pca = PCA(n_components=20)
pca.fit(y)

# Initialize the TF-IDF Vectorizer, which is used to convert text input into numerical features
vectorizer = TfidfVectorizer(max_features=512)  # 'max_features=512' limits the number of features

# Define the hyperparameters for the XGBoost regressor
tuned_params = {
    'objective': 'reg:squarederror',  
    'eta': 0.1, 
    'max_depth': 6,
    'subsample': 0.8, 
    'colsample_bytree': 0.8,
    'n_estimators': 100 
}

# Initialize the model as a MultiOutputRegressor with XGBoost as the base model. MultiOutputRegressor allows us to predict multiple outputs (traits) at once
xgb_model = MultiOutputRegressor(XGBRegressor(**tuned_params))

# Function to scale back the predictions to the original range (inverse scaling). This is necessary because the model was trained on scaled data, but we want the predictions in their original scale
def scale_back(scaler: MinMaxScaler, vector):
    return scaler.inverse_transform(vector.reshape(1, -1))

# Function to make predictions based on the user input text
def predict_traits(input_text):
    # Use the pre-trained model (pipeline) to predict the traits for the input text
    y_pred_reduced = pipeline.predict([input_text])

    # Apply PCA inverse transformation to recover the full traits from the reduced ones
    y_pred_full = pca.inverse_transform(y_pred_reduced)[0]

    # Convert the predictions into float values
    y_pred_full = np.array([float(score) for score in y_pred_full])

    # Get the names of the traits (column names of the Traits dataframe)
    trait_names = list(Traits.columns)

    # Create a dictionary of predicted trait values, scaled back to their original range
    trait_dict = dict(zip(trait_names, [round(float(trait), 4) for trait in scale_back(scaler, y_pred_full)[0]]))
    
    # Return both the trait dictionary and the scaled predictions
    return trait_dict, y_pred_full

# Main function to run the Streamlit app
if __name__ == "__main__":

    pipeline = joblib.load('./model/my_pipeline.pkl')

    st.title("HVP | Personality Trait Prediction")

    user_input = st.text_input("Enter a text here:")

    if user_input:
        y, y_out = predict_traits(user_input)

        st.write("User input: \n\n", f"{user_input}")

        # Create a Plotly chart to visualize the predicted personality traits
        fig = go.Figure(data=go.Scatter(
            x=list(Traits.columns), 
            y=scale_back(scaler, y_out)[0], 
            mode='lines+markers', 
            line=dict(color='yellow', width=2),  
            marker=dict(color='yellow', size=8)  
        ))

        # Display the Plotly chart in the Streamlit app
        st.plotly_chart(fig)

        # Display the predicted personality traits in the app
        st.write("HVP Scores\n:", y)
