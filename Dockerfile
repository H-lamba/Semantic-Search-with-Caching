FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data (needed by clustering service at import time)
RUN python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"

# Copy application code and pre-built models
COPY . .

# Expose port 8000 (Task 95)
EXPOSE 8000

# Entry point: start uvicorn server (Task 95)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
