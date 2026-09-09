# Interactive Evolutionary Prompt Optimization for Music Generation

This repository implements a human-in-the-loop interactive evolutionary computation (IEC) framework combined with hill-climbing search in embedding space to optimize text prompts for generative audio models (facebook/musicgen-small).

## Architecture
- Language & Embedding Model: sentence-transformers/all-MiniLM-L6-v2
- Audio Generative Model: facebook/musicgen-small via Hugging Face Transformers
- Optimization Paradigm: IEC-based target preference vector updating and cosine distance navigation
- User Interface: Gradio web interface

## Installation
pip install -r requirements.txt

## Running the System
python app.py
