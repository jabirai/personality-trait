# Project Overview

This repository contains the code and resources for **Personality Trait Analysis** using the **HVP**, **SVS**, and **Ocean** models. It includes instructions for running the application using either **Lambda** or **Docker containers**. 

- The **`main`** branch is designed to run the application using **Streamlit**, providing an interactive web interface.
- The **`deploy`** branch contains the necessary Docker container and its relevant dependencies to run the application.

The project includes two models:
- **HVP** (High-Value Prediction)
- **SVS** (Sustainable Value Scoring)

## Project Structure

- **`codes/`**: Contains the code for the HVP, SVS, and Ocean models.
- **`dataset/`**: Contains the data files used for the HVP and SVS models:
  - **`combined_hvp_numeric.xlsx`**: The dataset for the HVP model.
  - **`combined_svs.xlsx`**: The dataset for the SVS model.
- **`docs/`**: Documentation folder with guides on how to run the models using Lambda or Docker containers.
- **`model/`**: Contains the trained `.pkl` model files:
  - **`hvp_pipeline.pkl`**: The trained pipeline for the HVP model.
  - **`svs_pipeline.pkl`**: The trained pipeline for the SVS model.
- **`app.py`**: A Streamlit application that provides an interactive web interface for running the models. This can be launched using Streamlit.
- **`app_flask.py`**: A Flask-based API for serving the model predictions, supporting both HVP and SVS models.
  
## Installation and Setup

### Prerequisites

- **Create a virtual environment**:
   ```bash
   python -m venv venv
   ./venv/scripts/activate  # On Windows
   source venv/bin/activate  # On Mac/Linux
   ```

- **Install the required dependencies for Streamlit app**:
   ```bash
   pip install -r requirements.txt
   ```

- **Install the required dependencies for Flask app** (if you're running `app_flask.py`):
   ```bash
   pip install -r requirements_for_flask.txt
   ```

### Running the Application

#### Option 1: Run with Streamlit (Main Branch)

1. Ensure you have installed the necessary dependencies as mentioned above.
2. Run the Streamlit application by using the following command:
   ```bash
   streamlit run app.py
   ```
3. The application will be available in your browser at `http://localhost:8501`.

#### Option 2: Run with Flask (Deploy Branch)

If you want to run the application using Flask (in the `deploy` branch):
1. Make sure Flask and other dependencies are installed using the `requirements_for_flask.txt`.
2. Run the Flask app using:
   ```bash
   python app_flask.py
   ```
3. The Flask app will be available at `http://localhost:5000`.

### Model Details

- **HVP Model**:
  - **Model file**: `hvp_pipeline.pkl`
  - **Dataset file**: `combined_hvp_numeric.xlsx`
  - **PCA components**: 20
  - This model predicts high-value personality traits from input text.
  
- **SVS Model**:
  - **Model file**: `svs_pipeline.pkl`
  - **Dataset file**: `combined_svs.xlsx`
  - **PCA components**: 14
  - This model predicts sustainable value scores for personality traits.

### Using the Models in Streamlit

The Streamlit application (`app.py`) allows users to select between the **HVP** and **SVS** models. Depending on the model selected, the appropriate model and dataset will be loaded dynamically. Here's how it works:

1. When you open the app, you will see a dropdown menu where you can choose between the `HVP` and `SVS` models.
2. After selecting a model, you can input text into the text box, and the app will predict the personality traits using the selected model.
3. The predicted traits will be displayed numerically and visualized on an interactive plot.

### Using the Models in Flask

The Flask app (`app_flask.py`) exposes an API endpoint (`/predict`) that accepts a POST request with a `text` field for the input text and an optional `model` field to specify whether to use `hvp` or `svs`. The app will return the predicted traits in JSON format.

- Example request to the `/predict` endpoint:
  ```bash
  curl -X POST -H "Content-Type: application/json" -d '{"text": "Some input text", "model": "hvp"}' http://localhost:5000/predict
  ```
  This will return the predicted traits for the **HVP** model. If no model is specified, it defaults to `hvp`.
  Update:
    - we have added another variable as user_id that tells us the difference between actual and predicted score. This is only done for svs model.
---

## Troubleshooting

1. **Issue with missing libraries**: If you encounter an error regarding missing libraries, make sure to activate the virtual environment and install all the dependencies via `pip install -r requirements.txt` or `pip install -r requirements_for_flask.txt` as per the application you're running.
2. **Model Not Found**: Ensure that the correct model and dataset files (`hvp_pipeline.pkl`, `svs_pipeline.pkl`, `combined_hvp_numeric.xlsx`, `combined_svs.xlsx`) are located in the respective folders (`model/` and `dataset/`).

---

## Conclusion

This repository provides two methods of interacting with the personality trait prediction models: through a Streamlit web application (`app.py`) or via a Flask API (`app_flask.py`). It supports both HVP and SVS models for personality analysis, with easy instructions for setting up and running in different environments like Lambda or Docker.
