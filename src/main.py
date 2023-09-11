from ultralytics import YOLO
import cv2
from sort import *  
import numpy as np

model =  YOLO('yolov8n.pt')
# results = model('images/human-faces3.jpg', show=True)
# cv2.waitKey(0)


# for webcam
# cap = cv2.VideoCapture(0)
# cap.set(3,1280)
# cap.set(4,720)

# for video
cap = cv2.VideoCapture('videos/cars2-small.mp4')
# cap.set(3,640)
# cap.set(4,480)

# # Get video frame size
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))



classNames =['person','bicycle','car','motorbike','aeroplane','bus','train','truck','boat',
             'traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep',
             'cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase',
             'frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard',
             'surfboard','tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana',
             'apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','sofa',
             'pottdplant','bed','diningtable','toilet','tvmonitor','laptop','mouse','remote','keyboard',
             'cell phone','microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors',
             'teddy bear','hair drier','toothbrush'
             ]

# import mask
mask  = cv2.imread('videos/cars2-mask.png')

# Tracking
tracker = Sort(max_age=20, min_hits=2, iou_threshold=0.3)

# limits for line
limits = [227,300,900,300]
totalCarCount = []

# image by image object detection with yolo
while True:
    
    # Read image
    success, img = cap.read()
    if not success:
        break
    
    # overaly image with mask
    overlayImg = cv2.bitwise_and(img,mask)
    # Get the width and height of the image
    height, width, _ = img.shape

    # obtain yolo detectios on overlay image
    results = model(overlayImg,stream=True)

    detections = np.empty((0,5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            print(x1,y1,x2,y2)
            
            # Display confidence score on top of the bounding box
            conf = box.conf[0]
            conf_text = f"Confidence: {conf:.2f}"
            text_size, _ =  cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            
            # Display class name on top of the bounding box
            cls = int(box.cls[0])
            detectedClasses = ['car','motorbike','bus','truck']
            className = classNames[cls]

            if className in detectedClasses and conf > 0.3:
                class_text = f"Class: {classNames[cls]} "
                text = class_text+conf_text
                # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
                # cv2.putText(img, text, (x1, y1 - text_size[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
                
                currentArray = np.array([x1,y1,x2,y2,conf])
                detections = np.vstack((detections, currentArray))



# to run tracker
    resultTracker = tracker.update(detections)
    cv2.line(img,(limits[0],limits[1]),(limits[2],limits[3]),(0,0,255),3)

    for result in resultTracker:
        x1,y1,x2,y2, id = result  
        x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)   
        print(result) 
        cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),3)
        cv2.putText(img, f"{int(id)} ", (x1, y1 - text_size[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)

        # find centers for each vehicle
        # check this - it is not working
        cx,cy =  int(x1+(x2-x1)/2), int(y1+(y2-y1)/2)
        print(cx,cy)
        cv2.circle(img,(cx,cy),3,(255,0,255), cv2.FILLED)

        if limits[0]< cx <limits[2] and  limits[1]-5 < cy < limits[1]+5:
            if totalCount.count(id) == 0: 
                totalCount.append(id)
                cv2.line(img,(limits[0],limits[1]),(limits[2],limits[3]),(0,255,0),3)

    cv2.putText(img, f"Count: {len(totalCount)} ", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
    

    cv2.imshow("Image",img)
    cv2.imwrite('output_image.jpg', img)
    # cv2.imshow("OverlayImage",overlayImg)
    cv2.waitKey(1)