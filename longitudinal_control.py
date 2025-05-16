import numpy as np
import time
from collections import deque

class LongitudinalControl:
    def __init__(self):
        # Speed limits
        self.max_speed = 105.0
        self.min_speed = 50.0

        # Speed prediction weights
        self.curvature_weight = 3.2

        # PID controller parameters
        self.kp = 1.0
        self.ki = 0.05
        self.kd = 0.03

        # PID internal state
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()
        self.integral_max = 6.0  # Anti-windup limit

        # Acceleration/brake scaling
        self.accel_gain = 0.2
        self.brake_gain = 0.03

        # Steering safety filter
        self.steer_safety_threshold = 0.9
        self.steer_history = deque(maxlen=5)
        self.curvature_history = deque(maxlen=4)


    def predict_target_speed(self, trajectory):
        # cut all points that are behind the car
        closest_index = np.argmin(np.linalg.norm(trajectory - [48, 64], axis=1))
        trajectory = trajectory[closest_index:]

        # get first n points
        n = min(15, len(trajectory))
        trajectory = trajectory[:n]

        # transform points to be relative to car
        trajectory = trajectory - np.array([48, 64])

        # calc target speed based on average x value of trajectory
        avg_x_deviation = np.mean(np.abs(trajectory[:, 0]))
        self.curvature_history.append(avg_x_deviation)
        return np.clip(self.max_speed - np.mean(self.curvature_history)  * self.curvature_weight, self.min_speed, self.max_speed)


    def control(self, current_speed, target_speed, steer):

        # Steering safety - if we steer too strong too long, no gas/brakes
        self.steer_history.append(abs(steer))
        if np.mean(self.steer_history) > self.steer_safety_threshold:
            return 0.0, 0.0

        # PID controller
        speed_error = target_speed - current_speed
        current_time = time.time()
        dt = max(current_time - self.last_time, 0.01)
        self.last_time = current_time

        self.integral += speed_error * dt
        self.integral = np.clip(self.integral, -self.integral_max, self.integral_max)

        derivative = (speed_error - self.prev_error) / dt
        self.prev_error = speed_error

        control_value = self.kp * speed_error + self.ki * self.integral + self.kd * derivative

        # Convert control value to gas/brake output
        if control_value > 0:
            gas = np.clip(control_value * self.accel_gain, 0.0, 1.0)
            brake = 0.0
        else:
            brake = np.clip(-control_value * self.brake_gain, 0.0, 1.0)
            gas = 0.0

        if steer > 0.5:
            gas *= 0.5

        return gas, brake
