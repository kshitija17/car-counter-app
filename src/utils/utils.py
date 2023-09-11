from ultralytics import YOLO
import cv2
from sort import *  
import numpy as np

import var

class Utils:
    def __init__(self):
        pass

    def load_model(self,yolo_model):
        var.model =  YOLO(yolo_model)

    def read_video(self,video_path):
        print(video_path)
        var.cap = cv2.VideoCapture(video_path)

    def read_mask(self,mask_path):
        var.mask  = cv2.imread(mask_path)

    def load_tracker(self,max_age=20, min_hits=2, iou_threshold=0.3):
        var.tracker = Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)
    
    def set_line_coordinates(self,coordinates):
        var.line_coordinates = coordinates

    def get_frame_size(self):
        # Open the video file
        video_path = 'static/cars2-small.mp4'  # Replace with your video file path
        cap = cv2.VideoCapture(video_path)

        # Check if the video file was opened successfully
        if not cap.isOpened():
            print("Error opening video file")
            exit()

        # Read the first frame to get video dimensions
        ret, frame = cap.read()

        # Check if the frame was read successfully
        if not ret:
            print("Error reading video frame")
            exit()

        # Get the height and width of the frame
        height = frame.shape[0]
        width = frame.shape[1]
        return height, width
