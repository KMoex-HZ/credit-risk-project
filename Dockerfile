FROM python:3.11-slim

WORKDIR /app

# Copy the requirements file first so Docker can cache this layer.
# This avoids reinstalling dependencies every time the source code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source code and model files required by the API
COPY api/ ./api/
COPY src/ ./src/
COPY models/ ./models/

EXPOSE 8000

# Start the FastAPI application using Uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]