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
    print("Connected to: " + str(address))

    try:
        while True:
            # 创建一个随机大小的NumPy数组
            size = np.random.randint(10, 20)
            array = np.random.randint(0, 100, size)
            data = pickle.dumps(array)  # 序列化数组
            conn.send(data)  # 发送序列化的数组
            print(f"Sent array: {array}")
    except socket.error as e:
        print("Client disconnected or error:", e)
    finally:
        conn.close()

if __name__ == '__main__':
    server_program()
