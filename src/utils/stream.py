import cv2

class Stream:

    def __init__(self,img):
        # Convert the frame to JPEG format
        ret, buffer = cv2.imencode('.jpg', img)
        img = buffer.tobytes()

        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')
        
    