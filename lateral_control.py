import numpy as np


class LateralControl:

    def __init__(self):
        self._car_position = np.array([48, 62])

        # parameter for stanley controller
        # gain for correcting crosstrack error
        self.k_e = .4
        # damping for high speed
        self.k_s = 1
        self.steering_gain = 1


    def control(self, trajectory, speed):
        if len(trajectory) < 5:
            return 0.0

        # Get closest point on trajectory in front of car
        closest_index = np.argmin(np.linalg.norm(trajectory - self._car_position, axis=1))
        closest_traj_point = trajectory[closest_index]

        # Calculate tangent vector of trajectory
        if closest_index == 0:
            # Use the next point if it's the first point
            tangent_vector = trajectory[closest_index + 1] - closest_traj_point
        elif closest_index == len(trajectory) - 1:
            # Use the previous point if it's the last point
            tangent_vector = closest_traj_point - trajectory[closest_index - 1]
        else:
            # Use the difference between the next and previous points
            tangent_vector = trajectory[closest_index + 1] - trajectory[closest_index - 1]

        # Calculate the angle of the tangent vector (in radians)
        angle = np.arctan2(tangent_vector[1], tangent_vector[0]) + np.pi / 2


        # Project RMS error onto front axle vector to calculate crosstrack error
        front_axle_vec = [1, 0]
        error_front_axle = np.dot(closest_traj_point - self._car_position, front_axle_vec)

        # Calculate the steering angle using the Stanley controller
        steering_angle = angle + np.arctan2(self.k_e * error_front_axle, speed + self.k_s)

        self.closest_traj_point = closest_traj_point
        return steering_angle * self.steering_gain

