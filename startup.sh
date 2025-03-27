#!/bin/bash

# Update package lists
apt update

# Install required system libraries
apt install -y libgl1 libglib2.0-0

# Install Python dependencies
pip install --no-cache-dir -r requirements.txt

# Start the Gradio app
python app.py
