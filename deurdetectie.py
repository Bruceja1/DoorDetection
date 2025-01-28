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

yoloModel = "DoorModel-init.pt"

class ObjectDetection:
    def __init__(self):
        self.model = self.load_model()

    def load_model(self):
        model = YOLO(yoloModel)       
        model.fuse() #Performance verbetering

        return model

    def predict(self, imgPath):
        results = self.model(imgPath, device='cpu', save=True, show=False)

        return results

    def plot_bboxes(self, results, imgPath):
        xyxys = []
        door_confidence = 0
        class_ids = []
        door_detected = False

        for result in results:
            boxes = result.boxes.cpu().numpy()
            # Coördinaten van de bounding boxes
            for box in boxes:
                xyxy = boxes.xyxy
                confidence = box.conf 
                class_id = box.cls 

                xyxys.append(xyxy)
                #confidences.append(confidence)
                class_ids.append(class_id)

                if class_id == 0: 
                    door_detected = True
                    door_confidence = confidence

        return door_detected, door_confidence

def process_image():
    door_detection = ObjectDetection()
    door_detection.load_model()
    
    while True:
        # Window pop-up om een afbeelding te selecteren
        root = tk.Tk() 
        root.withdraw() 
        file_path = filedialog.askopenfilename(initialdir='./data/') 
        door_detection_results = door_detection.predict(file_path)
        is_valid = door_detection.plot_bboxes(door_detection_results, file_path)
        if is_valid[0]:
            print(f'\nDeur succesvol gedetecteerd met confidence {is_valid[1]}\n')
            break
        else:
            print('\nGeen deur gedetecteerd, maak een nieuwe foto!\n')
        
process_image()

# Load a model
#model = YOLO("yolo11n.yaml")  # build a new model from YAML
#model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
#model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # build from YAML and transfer weights

# Train the model
#results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
