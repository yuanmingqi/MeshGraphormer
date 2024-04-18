import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import time

# Set the maximum size of the queue
max_length = 100

# Create a deque to hold the x and y data
xdata = deque(maxlen=max_length)
ydata = deque(maxlen=max_length)

# Initialize figure and line
fig, ax = plt.subplots()
line, = ax.plot([], [], 'r-', linewidth=2)
plt.title('Real-time Height Data of a Hand Joint')
plt.xlabel('Time')
plt.ylabel('Height (cm)')
ax.set_xlim(0, max_length)
ax.set_ylim(-10, 10)

def init():
    line.set_data([], [])
    return line,

def update(frame):
    # Simulate real data; replace this with actual data fetching
    new_data = np.random.randn()
    ydata.append(new_data)
    xdata.append(frame if len(xdata) == 0 else xdata[-1] + 1)
    
    line.set_data(list(xdata), list(ydata))
    ax.set_xlim(xdata[0], xdata[-1] + 1)  # Adjust x-axis to show the latest data points
    return line,

ani = FuncAnimation(fig, update, frames=np.arange(1, 200), init_func=init, blit=True)

plt.show()
