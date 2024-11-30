---

# Personality Trait Analysis

This repository provides resources for **Personality Trait Analysis** using the **HVP**, **SVS**, and **Ocean** models. It includes instructions for running the application via **Streamlit** or **Docker containers**.

### Project Structure

- **`codes/`**: Code for HVP, SVS, and Ocean models.
- **`dataset/`**: Datasets for the models.
- **`docs/`**: Guides for running models with Lambda or Docker.
- **`model/`**: Trained model files (`hvp_pipeline.pkl`, `svs_pipeline.pkl`, `ocean_pipeline.pkl`).
- **`app.py`**: Streamlit app for interactive model use.
- **`app_flask.py`**: Flask-based API for serving model predictions.

---

## Installation & Setup

### Prerequisites

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Mac/Linux
   ./venv/scripts/activate  # Windows
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Running the Application

### Option 1: Streamlit (Main Branch)

1. Install dependencies.
2. Run the app:
   ```bash
   streamlit run app.py
   ```
3. Access it at `http://localhost:8501`.

### Option 2: Flask (Deploy Branch)

1. Install Flask dependencies (`requirements_for_flask.txt`).
2. Run the Flask app:
   ```bash
   python app_flask.py
   ```
3. Access it at `http://localhost:5000`.

---

## Docker Setup

To build and run the application with Docker:

1. **Build the Docker image**:
   ```bash
   docker build -t jabirai096/personality_trait .
   ```

2. **Run the Docker container**:
   ```bash
   sudo docker run -d --network host --name personality_trait jabirai096/personality_trait
   ```

3. **Login to Docker Hub**:
   ```bash
   docker login
   ```

---

## Model Details

- **HVP Model**: Predicts high-value personality traits.  
  File: `hvp_pipeline.pkl`, Dataset: `combined_hvp_numeric.xlsx`

- **SVS Model**: Predicts sustainable value scores.  
  File: `svs_pipeline.pkl`, Dataset: `combined_svs.xlsx`

- **Ocean Model**: Predicts big five personality model
  File: `ocean_pipeline.pkl`, Dataset: `ocean_prepared_dataset.xlsx`  

---

## Troubleshooting

1. **Missing Libraries**: Ensure dependencies are installed via the requirements files.
2. **Model Not Found**: Check if model and dataset files are in the correct folders (`model/` and `dataset/`).

---

## Conclusion

This repository provides two methods to interact with the personality trait models: a Streamlit web app (`app.py`) and a Flask API (`app_flask.py`), supporting both HVP and SVS models. You can also use Docker for easier setup and deployment.

---
