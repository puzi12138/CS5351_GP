# Running SnapML

SnapML is a drag-and-drop platform for machine learning based on BlocklyML. This document provides instructions for running the application.

## Requirements

- Python 3.11 (required)
- Flask and other dependencies listed in requirements.txt

## Setup Instructions

### Method 1: Using Python Virtual Environment (Recommended)

1. Create a virtual environment with Python 3.11:
   ```bash
   python3.11 -m venv venv
   ```

2. Activate the virtual environment:
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   python app.py
   ```

5. Open your browser and navigate to http://localhost:5003

### Method 2: Using Docker

1. Build the Docker image:
   ```bash
   docker build -t snapml .
   ```

2. Run the Docker container:
   ```bash
   docker run -p 5000:5000 snapml
   ```

3. Open your browser and navigate to http://localhost:5000

## Troubleshooting

### Port Already in Use

If you see an error like "Address already in use," try changing the port number in `app.py`:

```python
app.run(host='0.0.0.0', port=8080)  # Change 8080 to any available port
```

### Templates Not Found

Make sure you're running the application from the SnapML directory:

```bash
cd path/to/SnapML
python app.py
```

### Python Version Issues

SnapML requires Python 3.11. Check your Python version with:

```bash
python --version
```

If you have multiple Python versions installed, make sure to use the correct one:

```bash
python3.11 app.py
```

### Missing Dependencies

If you encounter errors about missing modules, install them with:

```bash
pip install -r requirements.txt
```

## Simple Test App

If you're having trouble running the full application, you can try the simplified version to verify your setup:

```bash
python simple_app.py
```

This should start a basic web server on port 8080 that you can access at http://localhost:8080 