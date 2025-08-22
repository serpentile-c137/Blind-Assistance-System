# -*- coding: utf-8 -*-
"""
Blind Assistance System - macOS Compatible Version
"""

import os
import cv2
import sys
import tarfile
import numpy as np
import pytesseract
import pyttsx3
import tensorflow as tf
import torch
from torch.nn import functional as F
from torchvision import transforms as trn
from PIL import Image
from utils import label_map_util, visualization_utils as vis_util
import warnings
from torch.serialization import SourceChangeWarning
warnings.filterwarnings("ignore", category=SourceChangeWarning)


os.environ["AVFoundationLoggingLevel"] = "off"


# Tesseract path for macOS
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

# Initialize text-to-speech
engine = pyttsx3.init()

# Constants
MODEL_NAME = 'ssd_inception_v2_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
PATH_TO_CKPT = os.path.join(MODEL_NAME, 'frozen_inference_graph.pb')
# PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
PATH_TO_LABELS = os.path.join('models', 'research', 'object_detection', 'data', 'mscoco_label_map.pbtxt')
# models/research/object_detection/data/mscoco_label_map.pbtxt
NUM_CLASSES = 90

# Download model if not found
if not os.path.exists(PATH_TO_CKPT):
    print("Downloading model...")
    os.system(f'curl -O {DOWNLOAD_BASE + MODEL_FILE}')
    with tarfile.open(MODEL_FILE) as tar:
        tar.extractall()
    print("Model extracted.")

# Load detection graph
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as f:
        od_graph_def.ParseFromString(f.read())
        tf.import_graph_def(od_graph_def, name='')

# Load label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load PyTorch Places365 model
arch = 'resnet18'
model_path = f'whole_{arch}_places365_python36.pth.tar'
assert os.path.exists(model_path), f"Scene model {model_path} not found."

# model = torch.load(model_path, map_location=torch.device('cpu'))
model = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)

model.eval()

centre_crop = trn.Compose([
    trn.Resize((256, 256)),
    trn.CenterCrop(224),
    trn.ToTensor(),
    trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load scene categories
# with open('categories_places365.txt') as f:
#     classes = [line.strip().split(' ')[0][3:] for line in f]
file_path = os.path.join(os.path.dirname(__file__), 'categories_places365.txt')
with open(file_path) as f:
    classes = [line.strip().split(' ')[0][3:] for line in f]
classes = tuple(classes)

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Webcam not found.")
    sys.exit()

with detection_graph.as_default():
    with tf.compat.v1.Session(graph=detection_graph) as sess:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            key = cv2.waitKey(1) & 0xFF

            # Scene recognition
            if key == ord('b'):
                cv2.imwrite('scene.jpg', frame)
                img = Image.open('scene.jpg')
                input_img = centre_crop(img).unsqueeze(0)
                with torch.no_grad():
                    logit = model(input_img)
                    h_x = F.softmax(logit, 1).squeeze()
                    probs, idx = h_x.sort(0, True)

                    print("POSSIBLE SCENES:")
                    engine.say("Possible Scene may be")
                    for i in range(5):
                        scene = classes[idx[i]]
                        print(scene)
                        engine.say(scene)
                    engine.runAndWait()

            # Object detection
            image_np_expanded = np.expand_dims(frame, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes_tensor = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            (boxes, scores, classes_detected, _) = sess.run(
                [boxes, scores, classes_tensor, num_detections],
                feed_dict={image_tensor: image_np_expanded}
            )

            # Show boxes when 'a' is pressed
            if key == ord('a'):
                vis_util.visualize_boxes_and_labels_on_image_array(
                    frame,
                    np.squeeze(boxes),
                    np.squeeze(classes_detected).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8
                )

            # OCR when 'r' is pressed
            if key == ord('r'):
                text = pytesseract.image_to_string(frame)
                print(text.strip())
                engine.say(text)
                engine.runAndWait()

            # Proximity warning
            for i, box in enumerate(boxes[0]):
                if scores[0][i] < 0.5:
                    continue

                class_id = int(classes_detected[0][i])
                mid_x = (box[1] + box[3]) / 2
                apx_distance = round((1 - (box[3] - box[1])) ** 4, 1)
                if class_id in [1, 3, 6, 8, 44] and apx_distance <= 0.5 and 0.3 < mid_x < 0.7:
                    object_name = category_index[class_id]['name']
                    cv2.putText(frame, 'WARNING!!!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    warning_msg = f"Warning - {object_name} very close to the frame"
                    print(warning_msg)
                    engine.say(warning_msg)
                    engine.runAndWait()

            # Display frame
            cv2.imshow('Blind Assistance Feed', cv2.resize(frame, (1024, 768)))

            # Exit on 't'
            if key == ord('t'):
                break

cap.release()
cv2.destroyAllWindows()
