from flask import Flask, request, render_template, send_from_directory, url_for
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import cv2
from werkzeug.utils import secure_filename
from darknet import Darknet
from utils import *

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'data/images/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the YOLO model
cfg_file = 'data/cfg/crop_weed.cfg'
weight_file = 'data/weights/crop_weed_final.weights'
namesfile = 'data/names/obj.names'
m = Darknet(cfg_file)
m.load_weights(weight_file)
class_names = load_class_names(namesfile)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/detect', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return render_template('index.html', error='No file part')

    file = request.files['image']
    if file.filename == '':
        return render_template('index.html', error='No selected file')

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(img_path)

        try:
            # Perform detection
            detections = detection(img_path, 0.4, 0.6)
            processed_image_filename = 'processed_image.jpg'
            detection_image_url = url_for('send_detection_image', filename=processed_image_filename)
            return render_template('result.html', image_url=detection_image_url)
        except Exception as e:
            return str(e)
    
    return

def detection(path, iou_thresh=0.4, nms_thresh=0.6):
    img = cv2.imread(path)
    original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(original_image, (m.width, m.height))
    boxes = detect_objects(m, resized_image, iou_thresh, nms_thresh)
    print_objects(boxes, class_names)
    plot_boxes(original_image, boxes, class_names, plot_labels=True, color=None)
    result_path = 'static/upload/' + os.path.basename(path)
    cv2.imwrite(result_path, cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))
    return result_path

@app.route('/detection/<filename>')
def send_detection_image(filename):
    return send_from_directory('data/detection', filename)


@app.route('/static/results/<filename>')
def send_file(filename):
    return send_from_directory('static/results', filename)

if __name__ == '__main__':
    app.run(debug=True)
