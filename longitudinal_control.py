import numpy as np
import time
from collections import deque

class LongitudinalControl:
    def __init__(self):
        # Speed limits
        self.max_speed = 80.0
        self.min_speed = 20.0

        # Speed prediction weights
        self.curvature_weight = 1.9
        self.steer_weight = 0.0

        # PID controller parameters
        self.kp = 0.1
        self.ki = 0.05
        self.kd = 0.03

        # PID internal state
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()
        self.integral_max = 6.0  # Anti-windup limit

        # Acceleration/brake scaling
        self.accel_gain = 0.3
        self.brake_gain = 0.08

        # Steering safety filter
        self.steer_safety_threshold = 0.65
        self.steer_history = deque(maxlen=5)

    def predict_target_speed(self, trajectory,a = None,b = None):
        
        # In the test file we get the path and not the curvature
        if not isinstance(trajectory, float):
            if trajectory is None or len(trajectory) < 3:
                curvature = 0.0           
            else:
                p1, p2, p3 = map(np.array, [trajectory[0], trajectory[len(trajectory) // 2], trajectory[-1]])
                vec1, vec2 = p2 - p1, p3 - p2
                norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)

                if norm1 > 1e-4 and norm2 > 1e-4:
                    dot = np.clip(np.dot(vec1, vec2) / (norm1 * norm2), -1.0, 1.0)
                    curvature = np.arccos(dot)
                else:
                    curvature = 0.0
        else:
            curvature = trajectory

        curvature_factor = 1.0 - self.curvature_weight * np.clip(curvature / np.pi, 0.0, 1.0)

        target_speed = self.max_speed * curvature_factor
        return np.clip(target_speed, self.min_speed, self.max_speed)

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

        return gas, brake
