# FastAPI Prediction API — Dockerfile:
FROM python:3.9-slim

WORKDIR /app

# Install dependencies:
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy API source:
COPY api/ ./api/

# Copy trained models and reports that the API needs at runtime:
COPY models/  ./models/
COPY reports/ ./reports/
COPY data/    ./data/

EXPOSE 8000

# Run FastAPI with uvicorn:
# workers=1 keeps it simple for a demo; increase for production:
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
