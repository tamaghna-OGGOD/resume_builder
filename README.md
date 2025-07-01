# ATS Résumé Builder

This project provides an end-to-end solution for building ATS-optimized résumés from PDF files. Key features include:

1. **OCR Processing:** Extracts text from PDF résumés using Tesseract OCR.
2. **AI-Powered Résumé Generation:** Creates ATS-friendly résumés using either a local Ollama LLM or the OpenAI API.
3. **Web Application:** FastAPI backend with a simple web UI for uploading and generating résumés.
4. **Customization:** Supports multiple résumé styles and model options.

---

## Project Structure

- **config.yml** – Configuration for Ollama, OpenAI, Poppler, and prompt templates.
- **ocr_utils.py** – Handles PDF-to-image conversion and OCR extraction.
- **script.py** – Orchestrates the workflow: loads config, runs OCR, builds prompts, calls the LLM, and parses results.
- **server.py** – FastAPI server that serves the web UI and `/api/build-resume` endpoint.
- **static/** – Contains the single-page HTML and JavaScript frontend.
- **resume_juanjosecarin.pdf** – Example résumé for testing.

---

## Prerequisites

- **Python 3.9+**
- **Required packages:**  
  Install dependencies with:
  ```bash
  pip install -r requirements.txt
  ```
- **Tesseract OCR:**  
  - [Windows installation guide (YouTube)](https://www.youtube.com/watch?v=2kWvk4C1pMo)
- **Poppler (for pdf2image):**  
  - Set `poppler_path` in `config.yml` to your Poppler `bin/` folder or add it to your `PATH`.  
  - [Windows installation guide (YouTube)](https://www.youtube.com/watch?v=IDu46GjahDs)
- **Ollama (optional, for local LLMs):**  
  - [Run a local Ollama server](https://www.youtube.com/watch?v=e3j1a2PKw1k)
- **OpenAI API Key (optional, for OpenAI models):**  
  - Add your `OPENAI_API_KEY` to `config.yml`.

---

## Running the Application

To launch the application using a batch script, update the following in `launch_resume_app.bat`:

```batch
REM Set the path to your application folder - replace with your actual path
set APP_PATH=C:\Users\xxxxxxxxxx\resume_builder

REM Set your conda environment name - replace with your actual environment name
set CONDA_ENV=ENV_NAME
```