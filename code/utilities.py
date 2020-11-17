import socket
import sys
import time
import pickle


class ControllableSocket(socket.socket):
    # TODO if you implement some netem or tcset commands for latency, do them in the send() method

    def __init__(self, latency, bandwidth):
        super(ControllableSocket, self).__init__(socket.AF_INET, socket.SOCK_STREAM)
        self._latency = latency
        self._bandwidth = bandwidth

    def send(self, bytes):
        # How long should it have taken to send how many bytes we've sent with our
        # given bandwidth limitation?
        required_duration = len(bytes) / self._bandwidth
        time.sleep(max(required_duration, self._latency))
        return super(ControllableSocket, self).send(bytes)


def create_socket(socket_type, host, port, params=None):
    if socket_type == 'control':
        try:
            my_socket = ControllableSocket(params[0], params[1])
            my_socket.connect((host, port))
        except socket.error:
            if my_socket:
                my_socket.close()
            return -1
    elif socket_type == 'normal':
        try:
            my_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            my_socket.connect((host, port))
        except socket.error:
            if my_socket:
                my_socket.close()
            return -1
    elif socket_type == 'listen':
        try:
            my_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            my_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            my_socket.bind((host, port))
            my_socket.listen(params)
        except socket.error:
            if my_socket:
                my_socket.close()
            return -1
    else:
        print("Please specify of the predefined types of sockets: 'control', 'normal' or 'listen'")
        sys.exit(-1)
    return my_socket


def receive_sample(accepted_socket, buf_size):
    data = b""
    while True:
        try:
            packet = accepted_socket.recv(buf_size)
        except ConnectionResetError:
            continue
        data += packet
        if len(packet) < buf_size:
            try:
                data = pickle.loads(data, encoding='bytes')
            except (EOFError, pickle.UnpicklingError, KeyError, ValueError, OverflowError):
                data = int(0)
            break
    return data
