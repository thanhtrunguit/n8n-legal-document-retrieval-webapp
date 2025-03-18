# Use an official Python runtime as the base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies (e.g., Java for vncorenlp)
RUN apt-get update && apt-get install -y \
    openjdk-11-jre \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

# Expose the port the app runs on (Hugging Face Spaces default is 7860)
EXPOSE 7860


CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
