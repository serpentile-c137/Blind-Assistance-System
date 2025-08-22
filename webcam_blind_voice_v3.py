# -*- coding: utf-8 -*-
"""
Blind Assistance System - macOS + iPhone Camera + No Repeated TTS Alerts
"""

import os
import sys
import cv2
import tarfile
import numpy as np
import pytesseract
import pyttsx3
import tensorflow as tf
import torch
from torch.nn import functional as F
from torchvision import transforms as trn
from PIL import Image
from collections import deque
from utils import label_map_util, visualization_utils as vis_util
import warnings
from torch.serialization import SourceChangeWarning

warnings.filterwarnings("ignore", category=SourceChangeWarning)
os.environ["AVFoundationLoggingLevel"] = "off"

# Setup Tesseract for macOS
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

# Initialize TTS
engine = pyttsx3.init()

# Constants
MODEL_NAME = 'ssd_inception_v2_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
PATH_TO_CKPT = os.path.join(MODEL_NAME, 'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join('models', 'research', 'object_detection', 'data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90

# Download model if not available
if not os.path.exists(PATH_TO_CKPT):
    print("Downloading model...")
    os.system(f'curl -O {DOWNLOAD_BASE + MODEL_FILE}')
    with tarfile.open(MODEL_FILE) as tar:
        tar.extractall()
    print("Model extracted.")

# Load TensorFlow detection graph
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

# Load Places365 model for scene recognition
arch = 'resnet18'
model_path = f'whole_{arch}_places365_python36.pth.tar'
assert os.path.exists(model_path), f"Scene model {model_path} not found."

model = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
model.eval()

centre_crop = trn.Compose([
    trn.Resize((256, 256)),
    trn.CenterCrop(224),
    trn.ToTensor(),
    trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

file_path = os.path.join(os.path.dirname(__file__), 'categories_places365.txt')
with open(file_path) as f:
    classes = [line.strip().split(' ')[0][3:] for line in f]
classes = tuple(classes)

# === iPhone Camera Detection ===
def detect_iphone_camera():
    print("Detecting available cameras...")
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame.shape[1] >= 640:
                print(f"Using camera index {i}")
                return cap
            cap.release()
    print("No suitable camera found. Please ensure Continuity Camera is enabled.")
    sys.exit()

cap = detect_iphone_camera()

# History buffer to prevent repeated TTS
tts_history = deque(maxlen=20)

with detection_graph.as_default():
    with tf.compat.v1.Session(graph=detection_graph) as sess:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            key = cv2.waitKey(1) & 0xFF

            # Scene detection on 'b'
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
            boxes_tensor = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores_tensor = detection_graph.get_tensor_by_name('detection_scores:0')
            classes_tensor = detection_graph.get_tensor_by_name('detection_classes:0')
            num_tensor = detection_graph.get_tensor_by_name('num_detections:0')

            (boxes, scores, classes_detected, _) = sess.run(
                [boxes_tensor, scores_tensor, classes_tensor, num_tensor],
                feed_dict={image_tensor: image_np_expanded}
            )

            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            classes_detected = np.squeeze(classes_detected).astype(np.int32)
            height, width, _ = frame.shape

            tts_queue = []

            for i in range(min(10, boxes.shape[0])):
                if scores[i] < 0.5:
                    continue

                class_id = classes_detected[i]
                box = boxes[i]
                y1, x1, y2, x2 = box
                x1_pixel, y1_pixel = int(x1 * width), int(y1 * height)
                x2_pixel, y2_pixel = int(x2 * width), int(y2 * height)

                object_name = category_index[class_id]['name']
                apx_distance_cm = round(((1 - (x2 - x1)) ** 4) * 100, 2)

                label = f"{object_name}: {int(scores[i]*100)}% - {apx_distance_cm:.2f} cm"
                print(label)
                cv2.rectangle(frame, (x1_pixel, y1_pixel), (x2_pixel, y2_pixel), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1_pixel, max(y1_pixel - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                if scores[i] >= 0.7 and object_name not in tts_history:
                    tts_queue.append(f"{object_name} at {apx_distance_cm:.2f} centimeters")
                    tts_history.append(object_name)

                mid_x = (x1 + x2) / 2
                if class_id in [1, 3, 6, 8, 44] and apx_distance_cm <= 50 and 0.3 < mid_x < 0.7:
                    warning_msg = f"Warning - {object_name} very close to the frame"
                    print(warning_msg)
                    engine.say(warning_msg)

            if tts_queue:
                sentence = "Detected objects: " + ", ".join(tts_queue)
                engine.say(sentence)
                engine.runAndWait()

            if key == ord('r'):
                text = pytesseract.image_to_string(frame)
                print("OCR Text:\n", text.strip())
                engine.say(text)
                engine.runAndWait()

            cv2.imshow('Blind Assistance Feed', cv2.resize(frame, (1024, 768)))

            if key == ord('t'):
                break

cap.release()
cv2.destroyAllWindows()
