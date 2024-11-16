# Project Overview

This repository contains the code and resources for Personality Trait Analysis.
It comprises of three analysis namely: HVP, SVS, and Ocean models along with instructions for running the application using either Lambda or Docker containers.
The main branch is using streamlit to run the application whereas deploy branch uses docker container and its relevant dependencies to run the application

## Project Structure

- **`codes/`**: Contains all the code for HVP (High-Value Prediction), SVS (Sustainable Value Scoring), and Ocean models.
- **`dataset/`**: Contains the data used for HVP and SVS models.
- **`docs/`**: Documentation folder with guides on how to run the models using Lambda or Docker containers.
- **`model/`**: Contains the trained `.pkl` model files for the HVP model.
- **`app.py`**: Runs a Streamlit application that provides an interactive web interface for running the models. This can be launched using Streamlit.

## Installation and Setup

### Prerequisites

- Create a venv by runnning 
   ```bash
   python -m virtualenv venv
   ./venv/scripts/activate
   ```

- Install the required dependencies for streamlit app:
   ```bash
   pip install -r requirements.txt
   ```

- Install the required dependencies for flask app:
   ```bash
   pip install -r requirements_for_flask.txt
   ```

### Running the Application

#### Option 1: Run with Streamlit

- Run the application using the following command for this main branch
   ```bash
   streamlit run app.py
   ```