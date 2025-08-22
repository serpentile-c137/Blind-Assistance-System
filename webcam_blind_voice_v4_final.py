# -*- coding: utf-8 -*-
"""
Blind Assistance System - macOS + iPhone/Android Camera + TTS every 15 Frames
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
import argparse
from utils import label_map_util, visualization_utils as vis_util
import warnings
from torch.serialization import SourceChangeWarning

warnings.filterwarnings("ignore", category=SourceChangeWarning)
os.environ["AVFoundationLoggingLevel"] = "off"

# === Argument Parsing ===
parser = argparse.ArgumentParser(description="Blind Assistance System")
parser.add_argument('--android-url', type=str, help="Android IP Webcam stream URL (e.g., http://192.168.1.100:8080/video)")
args = parser.parse_args()

# === Tesseract for macOS (adjust if using Linux/Windows) ===
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

# === TTS Engine ===
engine = pyttsx3.init()

# === TensorFlow Object Detection Config ===
MODEL_NAME = 'ssd_inception_v2_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
PATH_TO_CKPT = os.path.join(MODEL_NAME, 'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join('models', 'research', 'object_detection', 'data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90

if not os.path.exists(PATH_TO_CKPT):
    print("Downloading detection model...")
    os.system(f'curl -O {DOWNLOAD_BASE + MODEL_FILE}')
    with tarfile.open(MODEL_FILE) as tar:
        tar.extractall()
    print("Model extracted.")

# === Load Object Detection Model ===
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as f:
        od_graph_def.ParseFromString(f.read())
        tf.import_graph_def(od_graph_def, name='')

# === Load Labels ===
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# === Load Places365 Scene Classifier ===
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

with open(os.path.join(os.path.dirname(__file__), 'categories_places365.txt')) as f:
    classes = [line.strip().split(' ')[0][3:] for line in f]
classes = tuple(classes)

# === Camera Detection Function ===
def detect_camera_source():
    if args.android_url:
        print(f"Using Android IP camera stream: {args.android_url}")
        return cv2.VideoCapture(args.android_url)

    print("Scanning for local cameras (macOS Continuity Camera or USB)...")
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame.shape[1] >= 640:
                print(f"Camera index {i} selected.")
                return cap
            cap.release()
    print("No valid camera found.")
    sys.exit(1)

cap = detect_camera_source()

# === Tracking TTS for objects ===
object_last_spoken_frame = {}
frame_counter = 0

# === Main Loop ===
with detection_graph.as_default():
    with tf.compat.v1.Session(graph=detection_graph) as sess:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera feed lost.")
                break

            key = cv2.waitKey(1) & 0xFF

            # === Scene Recognition ===
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

            # === Object Detection ===
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
                object_name = category_index[class_id]['name']
                box = boxes[i]
                y1, x1, y2, x2 = box
                x1_pixel, y1_pixel = int(x1 * width), int(y1 * height)
                x2_pixel, y2_pixel = int(x2 * width), int(y2 * height)

                distance_cm = round(((1 - (x2 - x1)) ** 4) * 100, 2)
                label = f"{object_name}: {int(scores[i]*100)}% - {distance_cm:.2f} cm"
                print(label)
                cv2.rectangle(frame, (x1_pixel, y1_pixel), (x2_pixel, y2_pixel), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1_pixel, max(y1_pixel - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                if scores[i] >= 0.6:
                    last_spoken = object_last_spoken_frame.get(object_name, -1000)
                    if frame_counter - last_spoken >= 15:
                        tts_queue.append(f"{object_name} at {distance_cm:.2f} centimeters")
                        object_last_spoken_frame[object_name] = frame_counter

                mid_x = (x1 + x2) / 2
                if class_id in [1, 3, 6, 8, 44] and distance_cm <= 50 and 0.3 < mid_x < 0.7:
                    warning = f"Warning - {object_name} very close to the frame"
                    print(warning)
                    engine.say(warning)

            if tts_queue:
                engine.say("Detected objects: " + ", ".join(tts_queue))
                engine.runAndWait()

            if key == ord('r'):
                text = pytesseract.image_to_string(frame)
                print("OCR Text:\n", text.strip())
                engine.say(text)
                engine.runAndWait()

            cv2.imshow('Blind Assistance Feed', cv2.resize(frame, (1024, 768)))

            if key == ord('t'):
                break

            frame_counter += 1

cap.rele
