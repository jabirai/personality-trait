# Import necessary libraries
import streamlit as st  # For building the web interface
import plotly.graph_objs as go  # For creating the interactive plot
import joblib  # For loading the pre-trained model

# For performing Principal Component Analysis (PCA)
from sklearn.decomposition import PCA
import numpy as np  # For handling numerical operations
import pandas as pd  # For data manipulation

from utils import utility as utls

# For scaling the target values (normalization)
from sklearn.preprocessing import MinMaxScaler
from openpyxl import load_workbook  # For loading excel files
import json
import os


working_dir = os.getcwd()

# Dictionary to map model names to file paths
models = {
    "hvp": {
        "model_path": "./model/hvp_pipeline.pkl",
        "dataset_path": "./dataset/combined_hvp_numeric.xlsx",
        "n_components": 20,  # Default PCA components for hvp
    },
    "svs": {
        "model_path": "./model/svs_pipeline.pkl",
        "dataset_path": "./dataset/combined_svs.xlsx",
        "n_components": 14,  # Reduced PCA components for svs
    },
    "ocean": {
        "model_path": os.path.join(working_dir, "model", "ocean_pipeline.pkl"),
        "dataset_path": os.path.join(
            working_dir, "dataset", "ocean_prepared_dataset.xlsx"
        ),
        "n_components": 3,
    },
}


# Helper function to load a model and dataset dynamically
def load_model_and_data(model_name):
    model_info = models.get(model_name)

    if not model_info:
        raise ValueError(f"Model '{model_name}' not found!")

    # Load the pre-trained model pipeline
    pipeline = joblib.load(model_info["model_path"])

    # Load the dataset for column names (traits)
    dataset = pd.read_excel(model_info["dataset_path"])

    # Adjust the number of PCA components if needed
    n_components = model_info["n_components"]

    if model_name == "svs":
        dataset = dataset[~dataset["Conformity"].isnull()]

    # Extract trait names (assuming dataset has 'Message' and 'ID' columns to drop)
    traits = dataset.drop(columns=["Message", "ID"])
    trait_names = list(traits.columns)

    return pipeline, traits, trait_names, n_components


# Function to scale back the predictions to the original range (inverse scaling)


def scale_back(scaler: MinMaxScaler, vector):
    return scaler.inverse_transform(vector.reshape(1, -1))


# Main function to run the Streamlit app


def run_streamlit_app():
    # Set the title of the Streamlit app
    st.title("Personality Trait Prediction")

    # Create a dropdown for selecting the model (hvp or svs)
    model_name = st.selectbox("Select Model", list(models.keys()))

    if model_name == "svs":
        user_id = st.selectbox("Select ID", list(svs_original_scores.keys()))

    # Load the model and dataset dynamically based on the selected model
    pipeline, traits, trait_names, n_components = load_model_and_data(model_name)

    # Initialize the MinMaxScaler for scaling the target values
    scaler = MinMaxScaler()

    # Fit the scaler to the target data and transform the target values (normalization)
    y = traits.values
    y_scaled = scaler.fit_transform(y)

    # Perform PCA (Principal Component Analysis) to reduce the dimensionality of the target space
    pca = PCA(n_components)
    pca.fit(y_scaled)

    # Create a text input field for the user to enter some text
    user_input = st.text_input("Enter a text here:")

    # If the user has entered text, proceed with prediction
    if user_input:
        # Use the pre-trained model (pipeline) to predict the traits for the input text
        y_pred_reduced = pipeline.predict([user_input])

        # Apply PCA inverse transformation to recover the full traits from the reduced ones
        y_pred_full = pca.inverse_transform(y_pred_reduced)[0]

        # Convert the predictions into float values
        y_pred_full = np.array([float(score) for score in y_pred_full])

        # Create a dictionary of predicted trait values, scaled back to their original range
        trait_dict = dict(
            zip(
                trait_names,
                [
                    round(float(trait), 4)
                    for trait in scale_back(scaler, y_pred_full)[0]
                ],
            )
        )

        if model_name == "ocean":
            trait_dict = utls.scale_to_range(trait_dict)

        # Display the user input text in the Streamlit app
        st.write("User input: \n\n", f"{user_input}")

        # Create a Plotly chart to visualize the predicted personality traits
        fig = go.Figure()
        if model_name == "ocean":
            fig.add_trace(
                go.Scatterpolar(
                    r=list(trait_dict.values()),  # Y-axis values for radar chart
                    theta=trait_names,  # Corresponding trait names
                    mode="lines+markers",  # Display lines and markers
                    name="Predicted",
                    line=dict(color="grey", width=2),  # Line color and width
                    marker=dict(color="blue", size=8),  # Marker color and size
                )
            )

        else:
            fig.add_trace(
                go.Scatter(
                    x=list(trait_names),  # X-axis represents the trait names
                    # Y-axis represents the predicted trait scores, scaled back
                    y=scale_back(scaler, y_pred_full)[0],
                    mode="lines+markers",  # Display as both line and markers
                    name="Predicted",
                    line=dict(color="yellow", width=2),  # Line color and width
                    marker=dict(color="yellow", size=8),  # Marker color and size
                )
            )
            if model_name == "svs":
                fig.add_trace(
                    go.Scatter(
                        x=list(trait_names),  # X-axis represents the trait names
                        # Y-axis represents the predicted trait scores, scaled back
                        y=list(svs_original_scores[str(user_id)].values()),
                        mode="lines+markers",  # Display as both line and markers
                        name="Actual",
                        line=dict(color="grey", width=2),  # Line color and width
                        marker=dict(color="grey", size=8),  # Marker color and size
                    )
                )

        # Display the Plotly chart in the Streamlit app
        st.plotly_chart(fig)

        # Display the predicted personality traits in the app
        st.write(
            "Predicted Traits for {} Model:\n".format(model_name.upper()), trait_dict
        )
        if model_name == "svs":
            st.write(
                "Actual Traits for {} Model:\n".format(model_name.upper()),
                svs_original_scores[str(user_id)],
            )


# Run the Streamlit app
if __name__ == "__main__":
    with open(
        os.path.join(working_dir, "dataset", "original_svs_scores.json"), "r"
    ) as file:
        svs_original_scores = json.load(file)
        file.close()
    run_streamlit_app()
