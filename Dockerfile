# Use the official Python image from Docker Hub
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file to the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the container
COPY . .

# Expose any ports if necessary (adjust based on your application's needs)
EXPOSE 8000

# Command to run your application (adjust based on your main file)
CMD ["python", "main.py"]