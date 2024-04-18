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
    fig, ax = plt.subplots()
    line, = ax.plot([], [], 'r-', linewidth=2, label='raw')
    line_smoothed, = ax.plot([], [], 'b-', linewidth=2, label='smoothed')
    ax.set_ylim(-0.1, 0.1)
    plt.xlabel('Time')
    plt.ylabel('IF-Y')
    plt.legend()
    # turn on interactive mode
    plt.ion()

    def update_plot():
        line.set_data(range(len(if_tip_y)), if_tip_y)
        line_smoothed.set_data(range(len(if_tip_y)), [ma.update(x) for x in list(if_tip_y)])
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()
        
    while True:
        ret, frame = cap.read()
        if ret == False:
            print("Error in reading video stream or file")
            break

        # real-time inference
        visual_imgs, joints_xyz = inference(frame, _model, mano_model, renderer, mesh_sampler, display=False)

        if_tip_y.append(joints_xyz[8][1])
        update_plot()

        # get angle between index and middle finger
        index_vec = joints_xyz[8] - joints_xyz[5]
        middle_vec = joints_xyz[12] - joints_xyz[9]
        # get degree between index and middle finger
        dot = np.dot(index_vec, middle_vec)
        norm = np.linalg.norm(index_vec) * np.linalg.norm(middle_vec)
        cos = dot / norm
        angle_index_middle = np.degrees(np.arccos(cos))

        plt.show()
        # cv2.imshow('frame', visual_imgs)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()