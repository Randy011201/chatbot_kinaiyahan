#!/usr/bin/env python
"""Simple chatbot startup script"""
import os
import sys

# Change to chatbot directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Import and run
import uvicorn
from main import app

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level="info"
    )
