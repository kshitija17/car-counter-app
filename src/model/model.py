import var
import cv2
import numpy as np
from utils.stream import Stream
from utils.utils import Utils

class Model:
    def __init__(self):
        pass

    def __call__(self):
        # Output video filename
        output_filename = "output_video.mp4"
        utils = Utils()
        height,width = utils.get_frame_size()

        # Define the codec and create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # You can change the codec as per your requirements
        video_writer = cv2.VideoWriter(output_filename, fourcc, 30.0, (width, height))


        # image by image object detection with yolo
        while True:
            
            # Read image
            success, img = var.cap.read()
            if not success:
                break
            
            # overaly image with mask
            overlayImg = cv2.bitwise_and(img,var.mask)
            # Get the width and height of the image
            height, width, _ = img.shape

            # obtain yolo detectios on overlay image
            results = var.model(overlayImg,stream=True)

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
                    className = var.classNames[cls]

                    if className in detectedClasses and conf > 0.3:
                        class_text = f"Class: {var.classNames[cls]} "
                        text = class_text+conf_text
                        # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
                        # cv2.putText(img, text, (x1, y1 - text_size[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
                        
                        currentArray = np.array([x1,y1,x2,y2,conf])
                        detections = np.vstack((detections, currentArray))



        # run tracker
            resultTracker = var.tracker.update(detections)
            cv2.line(img,(var.line_coordinates[0],var.line_coordinates[1]),(var.line_coordinates[2],var.line_coordinates[3]),(0,0,255),3)

            for result in resultTracker:
                x1,y1,x2,y2, id = result  
                x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)   
                # print(result) 
                # draw bounding box
                cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),3)
                cv2.putText(img, f"{int(id)} ", (x1, y1 - text_size[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)

                # find centers for each vehicle
                # check this - it is not working
                cx,cy =  int(x1+(x2-x1)/2), int(y1+(y2-y1)/2)
                print(cx,cy)
                cv2.circle(img,(cx,cy),3,(255,0,255), cv2.FILLED)

                if var.line_coordinates[0]< cx <var.line_coordinates[2] and  var.line_coordinates[1]-5 < cy < var.line_coordinates[1]+5:
                    if var.totalCarCount.count(id) == 0: 
                        var.totalCarCount.append(id)
                        cv2.line(img,(var.line_coordinates[0],var.line_coordinates[1]),(var.line_coordinates[2],var.line_coordinates[3]),(0,255,0),3)

            cv2.putText(img, f"Count: {len(var.totalCarCount)} ", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
            
            video_writer.write(img)
        # Release the VideoWriter object
        video_writer.release()    
        
        

            # cv2.imshow("Image",img)
            # cv2.imwrite('output_image.jpg', img)
            # # cv2.imshow("OverlayImage",overlayImg)
            # cv2.waitKey(1)

            