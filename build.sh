#!/usr/bin/env bash
set -o errexit

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Download spaCy English model (required by resume parser)
python -m spacy download en_core_web_sm
