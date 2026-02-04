#!/bin/bash
cd "$(dirname "$0")"
python3 -m uvicorn api:app --reload --port 8000
