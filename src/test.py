from ultralytics import YOLO
import cv2

model =  YOLO('yolov5n.pt')
# results = model('images/human-faces3.jpg', show=True)
# cv2.waitKey(0)


# for webcam
# cap = cv2.VideoCapture(0)
# cap.set(3,1280)
# cap.set(4,720)

# for video
cap = cv2.VideoCapture('videos/cars1-small.mp4')
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

while True:
    
    success, img = cap.read()
    if not success:
        break

    # Resize image to fit screen
    # img = cv2.resize(img, (screen_width, screen_height))

    # Get the width and height of the image
    height, width, _ = img.shape

    # Print the width and height
    # print("Image Width:", width)
    # print("Image Height:", height)
    # img = cv2.resize(img, (3000, 2000))

    results = model(img,stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            print(x1,y1,x2,y2)
            cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            # Display confidence score on top of the bounding box
            conf = box.conf[0]
            conf_text = f"Confidence: {conf:.2f}"
            text_size, _ =  cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            
            # Display class name on top of the bounding box
            cls = int(box.cls[0])
            class_text = f"Class: {classNames[cls]} "
            text = class_text+conf_text
            cv2.putText(img, text, (x1, y1 - text_size[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
            # cv2.putText(img, , (x1, y1 - text_size[1] - 5), 0.6, (0, 255, 0), 1, cv2.LINE_AA)
            


    cv2.imshow("Image",img)
    cv2.waitKey(1)