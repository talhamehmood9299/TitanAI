# Use the official Python base image
FROM python:3.11-slim

# Set environment variables to prevent Python from writing pyc files to disc
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1


# Set the working directory in the container
WORKDIR /app

# Create the tmp folder
RUN mkdir -p tmp

# Install ffmpeg
RUN apt-get update && apt-get install -y ffmpeg

# Copy the requirements file into the container
COPY requirements.txt /app/

# Install the dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the rest of the application code into the container
COPY . /app/

# Expose the new port for the FastAPI app
EXPOSE 5000

# Define the command to run the FastAPI app with Uvicorn on the new port
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]
