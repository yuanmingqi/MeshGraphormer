import socket


host = socket.gethostname()
port = 5000

server_socket = socket.socket()
server_socket.bind((host, port))

