import array
import socket
import cv2
import numpy as np

from realtime import build_model, inference, parse_args

if __name__ == '__main__':
    # socket
    host = '192.168.1.1'
    port = 5000

    server_socket = socket.socket()
    server_socket.bind((host, port))

    server_socket.listen(2)
    conn, address = server_socket.accept()
    conn.settimeout(60)
    print("Connected to: " + str(address))
    # arguments
    args = parse_args()
    # create model, renderer and mesh sampler
    _model, mano_model, renderer, mesh_sampler = build_model(args)
    # open camera and start real-time inference
    cap = cv2.VideoCapture(4)
    
    while True:
        ret, frame = cap.read()
        if ret == False:
            print("Error in reading video stream or file")
            break

        # real-time inference
        visual_imgs, joints_xyz, angle_index_middle = inference(frame, _model, mano_model, renderer, mesh_sampler)
        # send data
        # try:
        #     data = joints_pos.tobytes()
        #     conn.send(data)  # 发送序列化的数组
        #     print(f"Sent array: {joints_pos}")
        # # except socket.error as e:
        #     # print("Client disconnected or error:", e)
        # except socket.timeout:
        #     print("Timeout")
        # finally:
        #     conn.close()

        try:
            array = joints_xyz[8].tolist() # index finger tip joint
            array.append(angle_index_middle)
            array = np.array(array).astype(np.float32)
            data = array.tobytes()
            conn.send(data)  # 发送序列化的数组
            print(f"Sent array: {array}")
        except socket.timeout:
            pass

        cv2.imshow('frame', visual_imgs)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()
