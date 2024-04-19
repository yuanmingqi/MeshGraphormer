import cv2
import datetime
import numpy as np

from ral2024.realtime import build_model, inference, parse_args

# 0: come here
# 1: stop 
# 2: ring

if __name__ == '__main__':
    file_name = input("File name:")
    # arguments
    args = parse_args()
    # create model, renderer and mesh sampler
    _model, mano_model, renderer, mesh_sampler = build_model(args)
    # open camera and start real-time inference
    cap = cv2.VideoCapture(4)

    all_joints_xyz = []
    count = 0
    while True:
        ret, frame = cap.read()
        if ret == False:
            print("Error in reading video stream or file")
            break

        # real-time inference
        visual_imgs, joints_xyz = inference(frame, _model, mano_model, renderer, mesh_sampler, display=True)
        all_joints_xyz.append(joints_xyz)
        cv2.imshow('frame', visual_imgs)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            np.save(f'./datasets/motion_{file_name}_{datetime.datetime.now().strftime("%Y_%m_%d_%I_%M_%S")}.npy', np.array(all_joints_xyz))
            break
        count += 1
        print(count)

    cv2.destroyAllWindows()
    cap.release()