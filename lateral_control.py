import numpy as np


class LateralControl:

    def __init__(self):
        self._car_position = np.array([48, 62])

        # parameter for stanley controller
        # gain for correcting crosstrack error
        self.k_e = .5
        # damping for high speed
        self.k_s = 0.8
        self.steering_gain = 1

        self.pure_pursuit_speed = 0
        self.pure_pursuit_k = 0.34  # Proportionality constant for lookahead distance

        self.closest_traj_point = [0, 0]

    def control(self, trajectory, speed):
        if len(trajectory) < 1:
            return 0.0

        if speed < self.pure_pursuit_speed:
            return self.stanley_control(trajectory, speed) * self.steering_gain

        # return self.stanley_control(trajectory, speed) * self.steering_gain
        return self.pure_pursuit_control(trajectory, speed) * self.steering_gain

    def stanley_control(self, trajectory, speed):
        # Get closest point on trajectory in front of car
        closest_index = np.argmin(np.linalg.norm(trajectory - self._car_position, axis=1))
        self.closest_traj_point = trajectory[closest_index]
        

        # Calculate tangent vector of trajectory
        if closest_index == 0:
            # Use the next point if it's the first point
            tangent_vector = trajectory[closest_index + 1] - self.closest_traj_point
        elif closest_index == len(trajectory) - 1:
            # Use the previous point if it's the last point
            tangent_vector = self.closest_traj_point - trajectory[closest_index - 1]
        else:
            # Use the difference between the next and previous points
            tangent_vector = trajectory[closest_index + 1] - trajectory[closest_index - 1]

        # Calculate the angle of the tangent vector (in radians)
        angle = np.arctan2(tangent_vector[1], tangent_vector[0]) + np.pi / 2


        # Project error onto front axle vector to calculate crosstrack error
        front_axle_vec = [1, 0]
        error_front_axle = np.dot(self.closest_traj_point - self._car_position, front_axle_vec)

        # Calculate the steering angle using the Stanley controller
        steering_angle = angle + np.arctan2(self.k_e * error_front_axle, speed + self.k_s)

        self.closest_traj_point = self.closest_traj_point
        return np.clip(steering_angle, -1, 1)

    def pure_pursuit_control(self, trajectory, speed):
        WB = 6  # Wheelbase of the car

        # Calculate the dynamic lookahead distance
        Lf = self.pure_pursuit_k * max(speed, 0.1)

        # for extra tight corners, dont look too far ahead to not cut the corner too much
        if (speed < 60):
            Lf = 0.8 * Lf

        # Find the lookahead point
        closest_index = np.argmin(np.linalg.norm(trajectory - self._car_position, axis=1))
        lookahead_point = None

        for i in range(closest_index, len(trajectory)):
            distance = np.linalg.norm(trajectory[i] - self._car_position)
            if distance >= Lf:
                lookahead_point = trajectory[i]
                break

        # If no point satisfies the lookahead distance, use the last point
        if lookahead_point is None:
            lookahead_point = trajectory[-1]

        # Calculate the steering angle using the Pure Pursuit controller
        dx = self._car_position[0] - lookahead_point[0]
        dy = (self._car_position[1] + WB) - lookahead_point[1]  # Adjust for rear axle position
        alpha = np.arctan2(dy, dx) - np.pi / 2

        # Calculate the steering angle
        delta = np.atan2(2.0 * WB * np.sin(alpha) / Lf, 1.0)

        return np.clip(delta, -1, 1)


