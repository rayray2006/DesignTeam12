# Yolo trained on DocCheck (Rona) dataset

from ultralytics import YOLO
import torch

# load yolov5 from online
# TODO: change to local
model = torch.hub.load('ultralytics/yolov5', 'custom', path='./models/instrument_detector_model.pt', force_reload=False)  # load a custom model

# Predict with the model
results = model("./eval-images/cropped.png")  # predict on an image
