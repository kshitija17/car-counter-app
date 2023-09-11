import numpy as np
from utils.utils import Utils
from model.model import Model
import var

def run():
    utils = Utils()
    model=Model()
    # load model
    utils.load_model('yolov8n.pt')

    # read video
    utils.read_video('static/cars2-small.mp4')

    # read mask
    utils.read_mask('static/cars2-mask.png')

    # load tracker
    utils.load_tracker(max_age=20, min_hits=2, iou_threshold=0.3)

    # set line_coordinates
    utils.set_line_coordinates(coordinates=[227,300,900,300])

    stream_img = model()

    return stream_img





if __name__=="__main__":
    run()