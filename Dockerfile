# Stage 1: Build the image for installing dependencies and obfuscating code
FROM python:3.12-slim AS builder

# Set working directory inside the container
WORKDIR /app

# Install necessary system dependencies for building packages like TensorFlow, Spacy, and others
RUN apt-get update && apt-get install -y \
    gcc \
    libatlas-base-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy all the necessary files into the container
COPY dataset/combined_hvp_numeric.xlsx /app/dataset/combined_hvp_numeric.xlsx
COPY model/hvp_pipeline.pkl /app/model/hvp_pipeline.pkl
COPY dataset/combined_svs.xlsx /app/dataset/combined_svs.xlsx
COPY model/svs_pipeline.pkl /app/model/svs_pipeline.pkl
COPY app_flask.py /app/
COPY requirements_for_flask.txt /app/

# Stage 2: Build the final image for runtime
FROM python:3.12-slim AS runtime

# Set working directory for the runtime container
WORKDIR /app

# Install runtime dependencies (in case they're different from the build dependencies)
RUN apt-get update && apt-get install -y \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the obfuscated files from the builder stage
COPY --from=builder /app /app

# Upgrade pip and install dependencies from requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements_for_flask.txt

# Now, obfuscate the Python code using PyArmor
RUN pyarmor gen /app/app_flask.py

# After obfuscation, the obfuscated file will be placed in a folder named `dist`. Let's rename it back to app_flask.py and remove original files
RUN mv /app/app_flask.py /app/app_flask.py.orig && \
    mv /app/dist/app_flask.py /app/app_flask.py && \
    mv /app/dist/* /app/ && \
    rm -rf /app/app_flask.py.orig /app/dist

# Expose the port Flask is running on
EXPOSE 5000

# Set the command to run the Flask app using Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app_flask:app"]