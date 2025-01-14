from ultralytics import YOLO
import tkinter as tk 
from tkinter import filedialog 

yoloModel = "yolo11n.pt" # Dit model herkent deuren helaas niet

# Kan later eventueel uitgebreid worden om live detection te doen met een camera:
# https://docs.ultralytics.com/modes/predict/#introduction
class ObjectDetection:
    def __init__(self):
        self.model = self.load_model()

    def load_model(self):
        model = YOLO(yoloModel)       
        model.fuse() #Performance verbetering

        return model

    def predict(self, imgPath):
        results = self.model(imgPath, device='cpu', save=True, show=True)

        return results

    def plot_bboxes(self, results, imgPath):
        xyxys = []
        confidences = []
        class_ids = []

        for result in results:
            boxes = result.boxes.cpu().numpy()
            # Coördinaten van de bounding boxes
            xyxy = boxes.xyxy

            print(boxes)

        return imgPath

root = tk.Tk() 
root.withdraw() 
file_path = filedialog.askopenfilename() 

door_detection = ObjectDetection()
door_detection.load_model()
door_detection_results = door_detection.predict(file_path)

door_detection.plot_bboxes(door_detection_results, file_path) # Coördinaten bounding boxes
door_detection_results







# Load a model
#model = YOLO("yolo11n.yaml")  # build a new model from YAML
#model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
#model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # build from YAML and transfer weights

# Train the model
#results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
