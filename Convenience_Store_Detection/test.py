import torch
import pymysql
import pandas as pd
import cv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mini_coffee_tea_model = torch.hub.load('ultralytics/yolov5', 'custom', path="./models/mini_coffee_tea.pt")

mini_coffee_tea_model.conf = 0.5
mini_coffee_tea_model.iou = 0.35

img = cv2.imread("./images/1677152696223872.jpg")

detail_detact = mini_coffee_tea_model(img, size=640)
result_pd = detail_detact.pandas().xyxy[0]
print(result_pd)


# 1677152696223872

