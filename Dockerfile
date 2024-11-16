FROM python:3.12-slim

# Set working directory inside the container
WORKDIR /app

# Install necessary system dependencies (to build packages like TensorFlow, Spacy)
RUN apt-get update && apt-get install -y \
    gcc \
    libatlas-base-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy all files into the container
COPY dataset/combined_hvp_numeric.xlsx /app/dataset/combined_hvp_numeric.xlsx
COPY codes/lambda_handler.py /app/codes/lambda_handler.py
COPY model/my_pipeline.pkl /app/model/my_pipeline.pkl
COPY app.py /app/
COPY app_flask.py /app/
COPY requirements.txt .

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements_for_flask.txt

# Expose port for Streamlit
# EXPOSE 8501
EXPOSE 5000

# CMD ["streamlit", "run", "app.py"]
# CMD ["python", "app_flask.py"]
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app_flask:app"]
