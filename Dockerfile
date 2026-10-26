FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required for XGBoost OpenMP and psycopg2
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python libraries
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files into container
COPY src/ ./src/
COPY app/ ./app/
COPY data/credit_risk_dataset.csv ./data/credit_risk_dataset.csv
COPY start.sh .

# Create directories and grant full permissions for Hugging Face non-root user (UID 1000)
RUN mkdir -p data models reports && \
    python src/1_data_prep.py && \
    python src/2_train_models.py && \
    python src/3_drift_monitor.py && \
    sed -i 's/\r$//' start.sh && \
    chmod +x start.sh && \
    chmod -R 777 /app

EXPOSE 8000 7860

CMD ["./start.sh"]