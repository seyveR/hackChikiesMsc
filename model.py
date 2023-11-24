from PIL import Image
from ultralytics import YOLO
import cv2
import numpy as np
from io import BytesIO
import base64

def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def image_to_base64_2(image):
    _, buffer = cv2.imencode('.png', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return base64.b64encode(buffer).decode("utf-8")

def paint_boxes(image_array, predictions_list):
    image_with_boxes = np.copy(image_array)
    for predictions in predictions_list:
        for box in predictions.boxes.xyxy:
            xB = int(box[2])
            xA = int(box[0])
            yB = int(box[3])
            yA = int(box[1])
            cv2.rectangle(image_with_boxes, (xA, yA), (xB, yB), (0, 0, 255), 5)
    return image_with_boxes

def load_model(model_path, task='detect'):
    return YOLO(model_path, task=task)
    