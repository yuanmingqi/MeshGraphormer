import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

from realtime import build_model, inference, parse_args
from filter import MovingAverage

if __name__ == '__main__':
    # arguments
    args = parse_args()
    # create model, renderer and mesh sampler
    _model, mano_model, renderer, mesh_sampler = build_model(args)
    # open camera and start real-time inference
    cap = cv2.VideoCapture(4)

    # Set the maximum size of the queue
    ma = MovingAverage()
    max_length = 100
    if_tip_y = deque(maxlen=max_length)
        
    while True:
        ret, frame = cap.read()
        if ret == False:
            print("Error in reading video stream or file")
            break

        # real-time inference
        visual_imgs, joints_xyz = inference(frame, _model, mano_model, renderer, mesh_sampler, display=True)

        # # get angle between index and middle finger
        index_vec = joints_xyz[8] - joints_xyz[5]
        ref_vec = joint_ref - joints_xyz[5]
        middle_vec = joints_xyz[12] - joints_xyz[9]
        # get degree between index and middle finger
        dot = np.dot(index_vec, middle_vec)
        norm = np.linalg.norm(index_vec) * np.linalg.norm(middle_vec)
        cos = dot / norm
        angle_index = np.degrees(np.arccos(cos))
        print(joints_xyz[8], joints_xyz[5], angle_index)

        plt.show()
        cv2.imshow('frame', visual_imgs)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()