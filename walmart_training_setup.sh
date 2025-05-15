#!/bin/bash

# Walmart Sales Model Training App Setup Script

echo "Setting up Walmart Sales Model Training App..."

# Create directory structure
echo "Creating directory structure..."
mkdir -p models/default
mkdir -p utils

# Install requirements
echo "Installing required packages..."
pip install -r requirements.txt

# Generate test data (optional)
read -p "Generate test data for development? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Generating test data..."
    python utils/test_data_generator.py
fi

echo "Setup complete!"
echo
echo "To run the app:"
echo "  streamlit run app.py"
echo
echo "The app will be available at: http://localhost:8501"