import socket
import pickle
import numpy as np

def server_program():
    host = '192.169.1.100'
    port = 5000

    server_socket = socket.socket()
    server_socket.bind((host, port))

    server_socket.listen(2)
    conn, address = server_socket.accept()
    print("Connection from: " + str(address))
    while True:
        data = conn.recv(4096)  # 增加接收缓冲区大小
        if not data:
            break
        array = pickle.loads(data)  # 反序列化数据
        print("Received array:", array)

        # 如果需要发送回复
        response = np.array([i for i in range(10)])  # 创建一个示例数组
        conn.send(pickle.dumps(response))  # 发送序列化的数组

    conn.close()

if __name__ == '__main__':
    server_program()
