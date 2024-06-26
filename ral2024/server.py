import cv2
import numpy as np
import torch
import torch.nn.functional as F
import multiprocessing as mp
import socket

from ral2024.hmm import HandGestureModel, num_joints, num_classes, motion_labels
from ral2024.realtime import build_model, inference, parse_args
from ral2024.filter import IntentionFilter, MovingAverage

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

    # socket
    host = '192.168.1.1'
    port = 5000
    server_socket = socket.socket()
    server_socket.bind((host, port))
    server_socket.listen(2)
    conn, address = server_socket.accept()
    conn.settimeout(60)
    print("Connected to: " + str(address))

    # Initialize the model
    model = HandGestureModel(num_joints=num_joints, num_classes=num_classes, hidden_dim=64, num_layers=2)
    # Load trained weights
    model.load_state_dict(torch.load(f'./models/hmm_{person}.pth'))
    model.eval()  # Set the model to evaluation mode

    # placeholder
    seq_placeholder = np.zeros((seq_length, num_joints * 3))
    count = 0

    # intention filter
    intention_filter = IntentionFilter(maxlen=10)
    # moving average
    angle_ma = MovingAverage(window_size=10)

    while True:
        joints_xyz_data = joints_xyz.get()
        # motion recognition
        seq_placeholder[count % seq_length] = joints_xyz_data.flatten()
        pred_class, probs = real_time_prediction(model, seq_placeholder)
        if probs[pred_class] > threshold:
            print(count, motion_labels[pred_class], probs)
            pred_class = intention_filter.add_prediction(pred_class)
        else:
            pred_class = intention_filter.add_prediction(np.random.randint(num_classes+1, num_classes+100))

        # get angle between index and middle fingers
        index_vec = joints_xyz_data[8] - joints_xyz_data[5]
        middle_vec = joints_xyz_data[12] - joints_xyz_data[9]
        dot = np.dot(index_vec, middle_vec)
        norm = np.linalg.norm(index_vec) * np.linalg.norm(middle_vec)
        cos = dot / norm
        angle_index_middle = np.degrees(np.arccos(cos))
        # apply moving average
        angle_index_middle = angle_ma.update(angle_index_middle)
        # clip the angle
        angle_index_middle = np.clip(angle_index_middle-7, 0, 29)

        # send data
        try:
            array = joints_xyz_data[8].tolist() # index finger tip joint
            array.append(angle_index_middle)
            array.append(pred_class)
            array = np.array(array).astype(np.float32)
            data = array.tobytes()
            conn.send(data)  # 发送序列化的数组
            print(f"Sent array: {array}")
        except socket.timeout:
            pass

        # time step
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