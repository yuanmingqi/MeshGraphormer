import socket
import pickle
import numpy as np

def client_program():
    host = socket.gethostname()  # as both code is run on same pc
    port = 5000  # socket server port number

    client_socket = socket.socket()  # instantiate
    client_socket.connect((host, port))  # connect to the server

    # 创建一个NumPy数组
    array = np.array([i for i in range(20)])
    data = pickle.dumps(array)  # 序列化数组
    client_socket.send(data)  # 发送数据

    # 接收服务器响应
    data = client_socket.recv(4096)  # 接收服务器响应
    response_array = pickle.loads(data)
    print('Received from server:', response_array)

    client_socket.close()  # close the connection

if __name__ == '__main__':
    client_program()
