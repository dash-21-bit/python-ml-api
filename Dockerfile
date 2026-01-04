# 1) Use a small official Python base image
FROM python:3.12-slim

# 2) Make python output logs immediately (no buffering)
ENV PYTHONUNBUFFERED=1

# 3) Set working directory inside container
WORKDIR /app

# 4) Install system deps (optional but helpful for matplotlib + SSL)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 5) Copy only requirements first (enables Docker layer caching)
COPY requirements.txt /app/requirements.txt

# 6) Install Python dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# 7) Copy the full project code into container
COPY . /app

# 8) Install your package (so "ml_api" imports work cleanly)
RUN pip install --no-cache-dir -e .

# 9) Create dataset + train model during image build
#    This ensures the container starts with a trained model available.
RUN python scripts/make_dataset.py && python scripts/train_model.py && python scripts/evaluate_model.py

# 10) Expose port used by uvicorn
EXPOSE 8000

# 11) Start the API server
CMD ["uvicorn", "ml_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
