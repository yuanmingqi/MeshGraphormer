import socket
import pickle
import numpy as np

def client_program():
    host = '192.168.1.1'
    port = 5000  # socket server port number

    client_socket = socket.socket()  # instantiate
    client_socket.connect((host, port))  # connect to the server

    try:
        while True:
            data = client_socket.recv(4096)  # 接收数据
            if not data:
                break  # 如果没有数据，结束循环
            array = pickle.loads(data)  # 反序列化数据
            print('Received array:', array)
    except socket.error as e:
        print("Error receiving data:", e)
    finally:
        client_socket.close()  # close the connection
        print("Connection closed.")

if __name__ == '__main__':
    client_program()
