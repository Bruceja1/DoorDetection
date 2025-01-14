'''
Stap 1: python -m venv .venv
Stap 2: ./.venv/Scripts/activate
Stap 3: pip install -r requirements.txt
Stap 4: Run het bestand
'''

# Werkt zowel met losse afbeeldingen als met video's
# Kan later eventueel uitgebreid worden om live detection te doen met een camera:
# https://docs.ultralytics.com/modes/predict/#introduction
from ultralytics import YOLO
import tkinter as tk 
from tkinter import filedialog 

yoloModel = "yolo11n.pt" # Dit model herkent deuren helaas niet


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

# Window pop-up om een afbeelding te selecteren
root = tk.Tk() 
root.withdraw() 
file_path = filedialog.askopenfilename(initialdir='./data/') 

door_detection = ObjectDetection()
door_detection.load_model()
door_detection_results = door_detection.predict(file_path)

door_detection.plot_bboxes(door_detection_results, file_path) # Coördinaten bounding boxes
door_detection_results

print("\n Open './runs/' om het resultaat te bekijken! \n")





# Load a model
#model = YOLO("yolo11n.yaml")  # build a new model from YAML
#model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
#model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # build from YAML and transfer weights

# Train the model
#results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
