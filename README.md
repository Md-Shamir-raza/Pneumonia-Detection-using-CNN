# Pneumonia Detection

Pneumonia Detection is a web application that uses a trained VGG19-based Keras model to classify chest X-ray images as `PNEUMONIA` or `NORMAL`.

The app provides a browser interface for uploading X-ray images, sends the image to an Express backend, and runs a Python prediction script against the saved model weights.

## Features

- Upload chest X-ray images from the browser.
- Supports JPG, JPEG, and PNG image files.
- Runs predictions with a TensorFlow/Keras model.
- Displays the predicted label and confidence score.
- Includes a responsive frontend in `public/index.html`.

## Project Structure

```text
.
+-- model_weight/
|   +-- vgg19_model_final.h5
+-- public/
|   +-- index.html
+-- uploads/
+-- hybrid.ipynb
+-- package.json
+-- predict.py
+-- server.js
```

## Requirements

- Node.js
- Python
- A Python virtual environment at `.venv`
- TensorFlow
- OpenCV
- NumPy

The backend currently expects the Python executable at:

```text
.venv/Scripts/python.exe
```

This path is configured in `server.js`.

## Installation

Install Node dependencies:

```bash
npm install
```

Create and activate a Python virtual environment:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

Install Python dependencies:

```bash
pip install tensorflow opencv-python numpy
```

Make sure the model file exists at:

```text
model_weight/vgg19_model_final.h5
```

## Running the App

Start the Express server:

```bash
node server.js
```

Open the app in your browser:

```text
http://localhost:3000
```

Upload a chest X-ray image from the web interface to receive a prediction.

## Prediction Script

You can also run the Python predictor directly:

```bash
python predict.py path/to/xray-image.jpg
```

The script prints a JSON response containing either:

```json
{
  "label": "PNEUMONIA",
  "confidence": 97.5
}
```

or an error message.

## Notes

- This project is intended for educational and experimental use.
- The prediction result should not be used as a final medical diagnosis.
- Uploaded images are stored in the `uploads/` directory by the server.
- If your Python environment is not located at `.venv/Scripts/python.exe`, update the `pythonExecutable` path in `server.js`.
