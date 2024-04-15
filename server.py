import socket
import pickle
import numpy as np

def server_program():
    host = '192.168.1.1'
    port = 5000

    server_socket = socket.socket()
    server_socket.bind((host, port))

    server_socket.listen(2)
    conn, address = server_socket.accept()
    print("Connection from: " + str(address))

    # 创建一个NumPy数组
    array = np.array([i for i in range(20)])
    data = pickle.dumps(array)  # 序列化数组
    conn.send(data)  # 发送序列化的数组

    conn.close()

if __name__ == '__main__':
    server_program()
