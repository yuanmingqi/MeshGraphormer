import cv2
import numpy as np
import torch
import torch.nn.functional as F
import multiprocessing as mp

from ral2024.hmm import HandGestureModel, num_joints, num_classes, motion_labels
from ral2024.realtime import build_model, inference, parse_args

def mesh_process(joints_xyz):
    # arguments
    args = parse_args()
    # create model, renderer and mesh sampler
    _model, mano_model, renderer, mesh_sampler = build_model(args)
    # open camera and start real-time inference
    cap = cv2.VideoCapture(4)

    # loop
    while True:
        ret, frame = cap.read()
        if ret == False:
            print("Error in reading video stream or file")
            break
        # real-time inference
        visual_imgs, joints_xyz_data = inference(frame, _model, mano_model, renderer, mesh_sampler, display=True)
        joints_xyz.put(joints_xyz_data)

        cv2.imshow('frame', visual_imgs)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()

def motion_process(joints_xyz):
    # also use this process to send the data
    def real_time_prediction(model, sequence):
        # Assuming 'sequence' is already in the correct shape and preprocessed
        # Convert sequence to tensor
        sequence = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

        # Make prediction
        with torch.no_grad():
            output = model(sequence)
            predicted_class = torch.argmax(output, dim=1).item()

        return predicted_class, F.softmax(output, dim=1).squeeze(0)

    person = 'ymq'
    seq_length = 10
    threshold = 0.8

    # Initialize the model
    model = HandGestureModel(num_joints=num_joints, num_classes=num_classes, hidden_dim=64, num_layers=2)
    # Load trained weights
    model.load_state_dict(torch.load(f'./models/hmm_{person}.pth'))
    model.eval()  # Set the model to evaluation mode

    # placeholder
    seq_placeholder = np.zeros((seq_length, num_joints * 3))
    count = 0

    while True:
        # motion recognition
        seq_placeholder[count % seq_length] = joints_xyz.get().flatten()
        pred_class, probs = real_time_prediction(model, seq_placeholder)
        if probs[pred_class] > threshold:
            print(count, motion_labels[pred_class], probs)
        count += 1

if __name__ == '__main__':
    joints_xyz = mp.Queue(maxsize=1)
    # add processes
    processes = []
    processes.append(mp.Process(target=mesh_process, args=(joints_xyz,)))
    processes.append(mp.Process(target=motion_process, args=(joints_xyz,)))
    # start processes
    for p in processes:
        p.start()
    for p in processes:
        p.join()