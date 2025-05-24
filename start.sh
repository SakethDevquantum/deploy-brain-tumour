#!/bin/bash
uvicorn main:app --host 0.0.0.0 --port 8000 &
streamlit run app.py --server.port 8080 --server.enableCORS false
