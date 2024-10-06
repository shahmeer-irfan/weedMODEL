# agri-backend


# YOLOv5 Image Detection API

This FastAPI application uses YOLOv5 to detect objects in images. The application accepts an image as input via an API endpoint, processes it using a YOLOv5 model, and returns the image with detected bounding boxes.

## Features

- **Endpoint**: `/detect/`
  - Accepts an image file (`.jpg`, `.png`, etc.)
  - Runs YOLOv5 object detection
  - Returns the image with bounding boxes drawn around detected objects

## Requirements

- Python 3.8+

## Installation

Follow these steps to set up and run the application:

1. **Clone the repository** (or download the code):

   git clone <your-repo-url>
   cd https://github.com/mohibahmedbleedai/agri-backend.git


  python -m venv env
  
venv/scripts/activate

pip install -r requirements.txt

python app.py

this will start backend server on

http://127.0.0.1:8000

