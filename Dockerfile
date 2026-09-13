# Use official Python 3.14 slim image
FROM python:3.14-slim

# Prevent Python from writing .pyc files & buffer output
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Run Streamlit on container launch
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]