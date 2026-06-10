FROM python:3.11-slim

WORKDIR /app

# scikit-surprise requires a C compiler to build from source; curl for healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir -e .

# 8000 = FastAPI backend · 8501 = Streamlit app. docker-compose runs one per service.
EXPOSE 8000 8501

# Default command = backend API. The Streamlit service overrides `command` in compose.
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
