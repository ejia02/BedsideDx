#!/bin/bash
# Activation script for BedsideDx virtual environment

cd /Users/ericjia/Downloads/BedsideDx
source venv/bin/activate

echo "✅ Virtual environment activated!"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo ""
echo "Installed packages:"
pip list | grep -E "extractous|requests|beautifulsoup4|pdfplumber|streamlit|openai|pandas"
echo ""
echo "To deactivate, type: deactivate"

