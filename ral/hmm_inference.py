import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ral.hmm import HandGestureModel, num_joints, num_classes
from ral.realtime import build_model, inference, parse_args

def real_time_prediction(model, sequence):
    # Assuming 'sequence' is already in the correct shape and preprocessed
    # Convert sequence to tensor
    sequence = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

    # Make prediction
    with torch.no_grad():
        output = model(sequence)
        predicted_class = torch.argmax(output, dim=1).item()

    return predicted_class, F.softmax(output, dim=1).squeeze(0)

if __name__ == '__main__':
    seq_length = 100
    threshold = 0.8

    # Initialize the model
    model = HandGestureModel(num_joints=num_joints, num_classes=num_classes, hidden_dim=64, num_layers=2)
    # Load trained weights
    model.load_state_dict(torch.load('./models/hmm.pth'))
    model.eval()  # Set the model to evaluation mode

    # tmp_seqs = np.load('./datasets/motion_come_2024_04_18_03_12_10.npy')
    # seq = np.zeros((seq_length, 63))

    # for i in range(tmp_seqs.shape[0]):
    #     seq[i % 100] = tmp_seqs[i].flatten()
    #     pred_class, probs = real_time_prediction(model, seq)
    #     if probs[pred_class] > threshold:
    #         print(i, pred_class, probs)

    # arguments
    args = parse_args()
    # create model, renderer and mesh sampler
    _model, mano_model, renderer, mesh_sampler = build_model(args)
    # open camera and start real-time inference
    cap = cv2.VideoCapture(4)

    # placeholder
    seq_placeholder = np.zeros((seq_length, num_joints * 3))
    count = 0

    while True:
        ret, frame = cap.read()
        if ret == False:
            print("Error in reading video stream or file")
            break
        
        # real-time inference
        visual_imgs, joints_xyz = inference(frame, _model, mano_model, renderer, mesh_sampler, display=True)
        
        # motion recognition
        seq_placeholder[count % 100] = joints_xyz.flatten()
        pred_class, probs = real_time_prediction(model, seq_placeholder)
        if probs[pred_class] > threshold:
            print(count, pred_class, probs)

        cv2.imshow('frame', visual_imgs)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        count += 1

    cv2.destroyAllWindows()
    cap.release()