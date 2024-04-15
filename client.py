import socket
import pickle
import numpy as np

def client_program():
    host = socket.gethostname()  # as both code is run on same pc
    port = 5000  # socket server port number

    client_socket = socket.socket()  # instantiate
    client_socket.connect((host, port))  # connect to the server

    # 接收数据
    data = client_socket.recv(4096)  # 可能需要调整缓冲区大小
    array = pickle.loads(data)  # 反序列化数据
    print('Received array:', array)

    client_socket.close()  # close the connection

if __name__ == '__main__':
    client_program()
