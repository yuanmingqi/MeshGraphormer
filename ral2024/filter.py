import numpy as np
from filterpy.kalman import KalmanFilter

def initialize_kalman_filter():
    # 创建卡尔曼滤波器对象
    kf = KalmanFilter(dim_x=6, dim_z=3)  # 假设状态向量包含位置和速度，观测向量只有位置

    # 初始化状态转移矩阵 (假设匀速模型)
    dt = 0.1  # 时间间隔，这里假设为1秒，可以根据需要调整
    kf.F = np.array([[1, 0, 0, dt, 0, 0],
                     [0, 1, 0, 0, dt, 0],
                     [0, 0, 1, 0, 0, dt],
                     [0, 0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 0, 1]])

    # 初始化测量矩阵 (只有位置)
    kf.H = np.array([[1, 0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0]])

    # 初始化协方差矩阵
    kf.P *= 1000.  # 初始估计的不确定性

    # 初始化观测噪声协方差矩阵
    kf.R = np.array([[5, 0, 0],
                     [0, 5, 0],
                     [0, 0, 5]])  # 调整这些值以反映观测的噪声水平

    # 初始化过程噪声协方差
    kf.Q = np.eye(kf.dim_x) * 0.1

    # 初始状态
    kf.x = np.array([0, 0, 0, 0, 0, 0])

    return kf

def update_kalman_filter(kf, measurement):
    # 更新卡尔曼滤波器
    kf.predict()
    kf.update(measurement)
    return kf.x[:3]  # 返回位置估计

# # 初始化滤波器
# kf = initialize_kalman_filter()

# # 示例观测数据（实际使用中需要从3D建模系统获取）
# measurements = [
#     np.array([0.1, 0.2, 0.3]),
#     np.array([0.4, 0.5, 0.6]),
#     np.array([0.7, 0.8, 0.9])
# ]

# # 处理每个观测
# for measurement in measurements:
#     estimate = update_kalman_filter(kf, measurement)
#     print("Estimate:", estimate)

from collections import deque

class MovingAverage:
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.window = deque(maxlen=window_size)
        self.sum = 0

    def update(self, new_value):
        if len(self.window) == self.window_size:
            self.sum -= self.window.popleft()
        self.window.append(new_value)
        self.sum += new_value
        return self.sum / len(self.window)

from collections import deque

class IntentionFilter:
    def __init__(self, maxlen):
        # 初始化一个双端队列，最大长度为10
        self.predictions = deque(maxlen=maxlen)

    def add_prediction(self, prediction):
        """
        添加一个新的预测到队列中，并检查是否有连续10个相同的预测结果。
        如果有，则返回该结果；如果没有有效结果，返回一个特定的错误识别码。
        """
        # 添加新的预测到队列
        if len(self.predictions) == 0 or self.predictions[-1] != prediction:
            # 如果最新预测与队列中的最后一个不同，重置队列
            self.predictions.clear()
        
        self.predictions.append(prediction)

        # 检查队列是否已满，并且所有元素都相同
        if len(self.predictions) == 10 and all(p == self.predictions[0] for p in self.predictions):
            return prediction
        
        # 如果没有有效的结果，返回错误识别码
        return 7777

# # 示例使用
# predictor = IntentPredictor()
# results = ['intent1', 'intent2', 'intent1', 'intent1', 'intent1', 'intent1', 'intent1', 'intent1', 'intent1', 'intent1']

# # 模拟预测数据流
# for result in results:
#     confirmed_result = predictor.add_prediction(result)
#     if confirmed_result != "NO_VALID_RESULT":
#         print(f"有效的预测结果是: {confirmed_result}")
#     else:
#         print("当前没有有效的预测结果。")
