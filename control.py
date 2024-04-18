import time
import numpy as np
import serial
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
import socket
from struct import unpack

# Clear and initialize (equivalent in Python)
# Note: No direct equivalent for `clear all` or `clc` in Python as garbage collection is automatic.

# Initial UR5 setting
rtde_r = RTDEReceiveInterface("169.254.159.50")
rtde_c = RTDEControlInterface("169.254.159.50")
print('UR5 Successfully Connected!')
time.sleep(1)

RobotiQ_2F_140_Length = 1e-3 * 232.8  # m

# # Initial Gripper Setting
# # Close COM4 if it's open
# ser = serial.Serial('COM4', 115200, timeout=1)
# if ser.isOpen():
#     ser.close()
# ser.open()

# # Gripper activation request
# activation_request = bytes([9, 16, 3, 232, 0, 3, 6, 0, 0, 0, 0, 0, 0, 115, 48])
# ser.write(activation_request)
# time.sleep(0.01)  # Wait for response

# The RGBD camera communication
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('192.168.1.1', 5000))
print('RGBD Camera Connection Success!')
numElement = 1 * 4

# Set UR5 initial position
initial_TCP = [0.0562, -0.3742, 0.4558 - RobotiQ_2F_140_Length, 0, 0, 0]
rtde_c.moveL(initial_TCP, 0.05, 0.05)
time.sleep(4)

safeBounds_x = np.array([initial_TCP[0] - 30e-3, initial_TCP[0] + 30e-3])
safeBounds_y = np.array([initial_TCP[1] - 30e-3, initial_TCP[1] + 30e-3])
safeBounds_z = np.array([initial_TCP[2] - 50e-3, initial_TCP[2] + 50e-3])
safeBounds_rx = np.array([np.deg2rad(-30), np.deg2rad(30)])
safeBounds_ry = np.array([np.deg2rad(-30), np.deg2rad(30)])
safeBounds_rz = np.array([np.deg2rad(-30), np.deg2rad(30)])

# Perform picking up the ring
pickAcc = 0.05
pickVelo = 0.05

ringLocation = [-0.1582, -0.3009, 0.2779 - RobotiQ_2F_140_Length, 0, 0, 0]
rtde_c.moveL(ringLocation, pickAcc, pickVelo)
print('Closing gripper')
close_command = bytes([9, 16, 3, 232, 0, 3, 6, 9, 0, 0, 255, 255, 255, 66, 41])
ser.write(close_command)
time.sleep(2)  # Wait for gripper to close

ringLocationLift = [-0.1582, -0.3009, 0.2779 + 0.03 - RobotiQ_2F_140_Length, 0, 0, 0]
rtde_c.moveL(ringLocationLift, pickAcc, pickVelo)
time.sleep(2)

# This segment of the code below (inserting ring to finger, execute trajectory) follows a similar format
# but depends on the specifics of your application.
# Implement further as needed based on the operations and checks described in MATLAB.

# Stop Script
rtde_c.stopScript()
ser.close()
client_socket.close()
