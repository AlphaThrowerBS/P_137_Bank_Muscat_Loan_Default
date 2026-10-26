#!/bin/bash
echo "🚀 Starting Bank Muscat FastAPI Backend on port 8000..."
uvicorn app.main:app --host 0.0.0.0 --port 8000 &

# Wait 3 seconds for FastAPI to load models
sleep 3

echo "🏦 Starting Bank Muscat Streamlit Risk Portal on port 7860..."
streamlit run app/streamlit_app.py --server.port 7860 --server.address 0.0.0.0