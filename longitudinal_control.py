from operator import is_
import numpy as np
from collections import deque
import copy

class LongitudinalControl:
    def __init__(self):
        # Speed limits
        self.max_speed = 105.0
        self.min_speed = 35.0

        # Speed prediction weights
        self.curvature_weight = 11.3
        self.deviation_weight = 9

        # threshhold for curvature and deviation to act as deadzone
        self.curvature_threshold = 0.05
        self.deviation_threshold = 3

        # Speed limits for hairpin turns
        self.hairpin_curvature = 0.34
        self.hairpin_curvature_weight = 8
        self.hairpin_max_speed = 55
        self.hairpin_timeout = 40
        self.hairpin_timeout_counter = self.hairpin_timeout

        # PID controller parameters
        self.kp = 1.2
        self.ki = 0.05
        self.kd = 0.03

        # PID internal state
        self.prev_error = 0.0
        self.integral = 0.0
        self.integral_max = 6.0  # Anti-windup limit

        # Acceleration/brake scaling
        self.accel_gain = 1.1
        self.brake_gain = 0.019

        # Steering safety filter
        self.steer_safety_threshold = 0.9
        self.steer_history = deque(maxlen=4)
        self.deviation_history = deque(maxlen=6)
        self.curvature_history = deque(maxlen=20)
        self.hairpin_curvature_history = deque(maxlen=1)

    def predict_target_speed(self, trajectory):
        # cut all points that are behind the car
        closest_index = np.argmin(np.linalg.norm(trajectory - [48, 64], axis=1))
        trajectory = trajectory[closest_index:]

        deviation_speed = self.x_deviation_target_speed(copy.deepcopy(trajectory))
        (curvature_speed, is_hairpin) = self.curvature_target_speed(copy.deepcopy(trajectory))
        if is_hairpin:
            return curvature_speed

        return (curvature_speed + deviation_speed) / 2.0

    def x_deviation_target_speed(self, trajectory):
        # get first n points
        n = min(15, len(trajectory))
        trajectory = trajectory[:n]

        # transform points to be relative to car
        trajectory = trajectory - np.array([48, 64])

        # calc target speed based on average x value of trajectory
        avg_x_deviation = np.mean(np.abs(trajectory[:, 0]))
        self.deviation_history.append(avg_x_deviation)

        deviation = max(0, float(np.mean(self.deviation_history)) - self.deviation_threshold)

        return np.clip(self.max_speed - deviation * self.deviation_weight, self.min_speed, self.max_speed)


    def curvature_target_speed(self, trajectory):
                # Check if the trajectory is valid
        if trajectory is None or len(trajectory) < 3:
            curvature = 0.0
        else:
            # Limit n to the number of available points
            n = len(trajectory)

            # Calculate curvature for consecutive triplets of points
            curvatures = []
            for i in range(n - 2):  # Ensure we have at least 3 points for each triplet
                p1, p2, p3 = map(np.array, [trajectory[i], trajectory[i + 1], trajectory[i + 2]])
                vec1, vec2 = p2 - p1, p3 - p2
                norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)

                if norm1 > 1e-4 and norm2 > 1e-4:
                    dot = np.clip(np.dot(vec1, vec2) / (norm1 * norm2), -1.0, 1.0)
                    curvature = np.arccos(dot)
                    curvatures.append(abs(curvature))

            # Calculate the average curvature
            curvature = np.mean(curvatures) if curvatures else 0.0
        
        self.curvature_history.append(curvature)
        self.hairpin_curvature_history.append(curvature)

        curv = max(0.0, float(np.mean(self.curvature_history)) - self.curvature_threshold)


        if np.mean(self.hairpin_curvature_history) > self.hairpin_curvature:
            self.hairpin_timeout_counter = 0
        else:
            self.hairpin_timeout_counter += 1

        # hairpin is recognized by a high curvature and points that wrap around and go back
        if self.hairpin_timeout_counter < self.hairpin_timeout:
            curvature_factor = 1.0 - self.hairpin_curvature_weight * np.clip(curv / np.pi, 0.0, 1.0)
            target_speed = self.hairpin_max_speed * curvature_factor
            
            # if hairpin is just passed, take it a bit slower to not spin out
            if self.hairpin_timeout_counter > 2:
                target_speed -= 10
            is_hairpin = True
        else:
            curvature_factor = 1.0 - self.curvature_weight * np.clip(curv / np.pi, 0.0, 1.0)
            target_speed = self.max_speed * curvature_factor
            is_hairpin = False

        return np.clip(target_speed, self.min_speed, self.max_speed), is_hairpin


    def control(self, current_speed, target_speed, steer):
        # PID controller
        speed_error = target_speed - current_speed
        dt = 0.04

        self.integral += speed_error * dt
        self.integral = np.clip(self.integral, -self.integral_max, self.integral_max)

        derivative = (speed_error - self.prev_error) / dt
        self.prev_error = speed_error

        control_value = self.kp * speed_error + self.ki * self.integral + self.kd * derivative

        # Convert control value to gas/brake output
        if control_value > 0:
            gas = np.clip(control_value * self.accel_gain, 0.0, 0.5 if self.hairpin_timeout_counter < self.hairpin_timeout else 1.0)
            brake = 0.0
        else:
            max_brake_val = 0.4 if self.hairpin_timeout_counter < self.hairpin_timeout else 1.0
            brake = np.clip(-control_value * self.brake_gain, 0.0, max_brake_val)
            gas = 0.0

        if abs(steer) > 0.7:
            gas = np.clip(gas, 0.0, 0.7)
        elif abs(steer) > 0.5:
            gas = np.clip(gas, 0.0, 0.9)

        # Steering safety - if we steer too strong too long, no gas/brakes
        self.steer_history.append(abs(steer))
        if np.mean(self.steer_history) > self.steer_safety_threshold:
            gas = np.clip(gas, 0.0, 0.2)
            brake = np.clip(brake, 0.0, 0.2)

        return gas, brake
