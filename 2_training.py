#訓練模型
import os
import cv2
import torch
from ultralytics import YOLO
from roboflow import Roboflow


if __name__ == '__main__':
    #使用cpu,請自行下載cuda和cudnn
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO("yolov8s.pt").to(device)
    model.train(data="photo-label--1/data.yaml", epochs=70, imgsz=1280, device=device)