# Crop and Weed Classification using YOLOv3

This project is used to classify crops and weeds using the YOLOv3 object detection model. It utilizes the [Crop and Weed Detection Dataset](https://www.kaggle.com/ravirajsinh45/crop-and-weed-detection-data-with-bounding-boxes), which includes 1300 images of sesame crops and various weed types, each labeled in YOLO format.

## Setup and Usage

### Prerequisites

Make sure you have the following dependencies installed:

- Python 3.x
- Flask
- opencv-python
- Werkzeug
- darknet-python
- torch
- matplotlib

### Installation and Running

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/jayanthmankavil/Weed-Crop-Detection.git 
   ```
2. Download the Pretrained Weights from:
    ```bash
    https://drive.google.com/file/d/1MbV7YoxC3GQjCzsqe93dK0bvPoz0ldzN/view?usp=drive_link
    ```
2. Navigate to the directory:
    ```bash
    cd Weed-Crop_Detection
    ```
3. Install the required Python packages by running:
    ```bash
    pip install -r requirements.txt
    ```
4. Start the Flask application by running `app.py`
    ```bash
    python app.py
    ```
5. Access the web application by opening your web browser and navigating to ``` http://localhost:5000```

## Configuration
The YOLOv3 model is loaded from the following configuration and weight files:
```crop_weed.cfg```: Configuration file for the YOLO model.  
```crop_weed_final.weights```: Pre-trained weights for the YOLO model.  
```obj.names```: Object class names.  
## Results
After uploading an image, the application will perform crop and weed detection using the YOLOv3 model and display the results.




### Feel free to contribute, report issues, or provide feedback to improve this project!