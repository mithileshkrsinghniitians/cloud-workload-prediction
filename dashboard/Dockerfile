# Streamlit Dashboard — Dockerfile:
FROM python:3.9-slim

WORKDIR /app

# Install dependencies:
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy dashboard source:
COPY dashboard/ ./dashboard/

# Copy shared project files that dashboard reads:
COPY data/      ./data/
COPY models/    ./models/
COPY reports/   ./reports/

EXPOSE 8501

# Run Streamlit on all interfaces (required inside Docker):
CMD ["streamlit", "run", "dashboard/streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
