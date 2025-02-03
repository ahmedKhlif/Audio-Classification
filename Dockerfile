# Use a base image with Python 3.8
FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Install system dependencies for librosa and soundfile
RUN apt-get update && \
    apt-get install -y \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements.txt file
COPY requirements.txt requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Ensure the wavfiles directory is copied
COPY wavfiles /app/wavfiles

# Expose the port for your application (if needed)
# EXPOSE 8080  # Uncomment if your application listens on a specific port

# Command to run when the container starts
CMD ["sh", "-c", "python eda.py && python model.py && python predict.py"]