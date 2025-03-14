# Document Analysis and Script Generator

This application uses Google's Gemini AI to analyze multiple types of documents (marketing, hooks, scripts, and product descriptions) and generate winning scripts with proper reasoning and timestamps.

## Setup

1. Create a `.env` file in the root directory and add your Google API key:
```
GOOGLE_API_KEY=your_api_key_here
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Place your documents in the respective folders:
- `docs/marketing/` - Marketing related documents
- `docs/hooks/` - Hook related documents
- `docs/scripts/` - Existing scripts
- `docs/products/` - Product descriptions

4. Run the application:
```bash
streamlit run src/app.py
```

## Features

- Document upload and management
- Multi-document analysis using Gemini AI
- Script generation with timestamps and reasoning
- Source tracking for generated content
- Interactive Streamlit UI 