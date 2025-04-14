import gymnasium as gym
import numpy as np
import pygame
from scipy.interpolate import splev, splprep

# Constants from the car racing environment
WINDOW_W = 1000
WINDOW_H = 800
STATE_W = 96
STATE_H = 96
ZOOM = 2.7
SCALE = 6.0


class CarRacingEnvWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

    def reset(self, **kwargs):
        observation, info = super().reset(**kwargs)
        info["speed"] = self._get_speed()
        info["trajectory"] = self._get_trajectory_groundtruth()
        left_lane_boundary, right_lane_boundary = self._get_lane_boundary_groundtruth()
        info["left_lane_boundary"] = left_lane_boundary
        info["right_lane_boundary"] = right_lane_boundary

        return observation, info

    def step(self, action: gym.core.ActType):
        observation, reward, done, truncated, info = super().step(np.array(action))
        info["speed"] = self._get_speed()
        info["trajectory"] = self._get_trajectory_groundtruth()
        left_lane_boundary, right_lane_boundary = self._get_lane_boundary_groundtruth()
        info["left_lane_boundary"] = left_lane_boundary
        info["right_lane_boundary"] = right_lane_boundary

        return observation, reward, done, truncated, info

    def _get_speed(self) -> float:
        """
        Extracts the speed of the car from the environment.

        Returns:
            float: The speed of the car.
        """
        return np.linalg.norm(self.unwrapped.car.hull.linearVelocity)

    def _get_lane_boundary_groundtruth(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes the ground truth lane boundaries from the environment.

        Returns:
            tuple[np.ndarray, np.ndarray]: The left and right lane boundary points.
        """
        left_lane_boundary = []
        right_lane_boundary = []

        road_segments = self._get_road_segments()

        # Iterate through the road segments
        for segment in road_segments:
            left_lane_boundary.append(
                np.asarray([segment[0][0], STATE_H - segment[0][1]])
            )
            right_lane_boundary.append(
                np.asarray([segment[1][0], STATE_H - segment[1][1]])
            )

        left_lane_boundary = np.asarray(left_lane_boundary)
        right_lane_boundary = np.asarray(right_lane_boundary)

        if left_lane_boundary.shape[0] > 1:
            # noinspection PyTupleAssignmentBalance
            left_boundary_tck, _ = splprep(
                [left_lane_boundary[:, 0], left_lane_boundary[:, 1]], s=1, k=1
            )

            # Create spline points from spline tck
            t = np.linspace(0, 1, 10000)
            left_lane_boundary = np.array(splev(t, left_boundary_tck)).reshape(2, -1).T

        if right_lane_boundary.shape[0] > 1:
            # noinspection PyTupleAssignmentBalance
            right_boundary_tck, _ = splprep(
                [right_lane_boundary[:, 0], right_lane_boundary[:, 1]], s=1, k=1
            )

            # Create spline points from spline tck
            t = np.linspace(0, 1, 10000)
            right_lane_boundary = (
                np.array(splev(t, right_boundary_tck)).reshape(2, -1).T
            )

        # Filter out points that are not in the image
        left_lane_boundary = left_lane_boundary[left_lane_boundary[:, 0] < 96]
        left_lane_boundary = left_lane_boundary[left_lane_boundary[:, 0] > 0]
        left_lane_boundary = left_lane_boundary[left_lane_boundary[:, 1] < 84]
        left_lane_boundary = left_lane_boundary[left_lane_boundary[:, 1] > 0]
        right_lane_boundary = right_lane_boundary[right_lane_boundary[:, 0] < 96]
        right_lane_boundary = right_lane_boundary[right_lane_boundary[:, 0] > 0]
        right_lane_boundary = right_lane_boundary[right_lane_boundary[:, 1] < 84]
        right_lane_boundary = right_lane_boundary[right_lane_boundary[:, 1] > 0]

        return left_lane_boundary, right_lane_boundary

    def _get_trajectory_groundtruth(self) -> np.ndarray:
        """
        Computes the ground truth trajectory from the environment.

        Returns:
            np.ndarray: The ground truth trajectory.
        """
        trajectory = []
        road_segments = self._get_road_segments()

        # Iterate through the road segments
        for segment in road_segments:
            trajectory.append(
                np.asarray(
                    [
                        (segment[0][0] + segment[1][0]) / 2,
                        (STATE_H - segment[0][1] + STATE_H - segment[1][1]) / 2,
                    ]
                )
            )

        trajectory = np.asarray(trajectory)

        # Filter out points that are not in the image
        trajectory = trajectory[trajectory[:, 0] < 96]
        trajectory = trajectory[trajectory[:, 0] > 0]
        trajectory = trajectory[trajectory[:, 1] < 84]
        trajectory = trajectory[trajectory[:, 1] > 0]

        return trajectory

    def _get_road_segments(self) -> list[np.ndarray]:
        """
        Extracts the road segments from the environment.

        Returns:
            list[np.ndarray]: The road segments.
        """
        road_segments = []

        # Compute the transformation
        angle = -self.unwrapped.car.hull.angle
        zoom = 0.1 * SCALE * max(1 - self.unwrapped.t, 0) + ZOOM * SCALE * min(
            self.unwrapped.t, 1
        )
        scroll_x = -(self.unwrapped.car.hull.position[0]) * zoom
        scroll_y = -(self.unwrapped.car.hull.position[1]) * zoom
        trans = pygame.math.Vector2((scroll_x, scroll_y)).rotate_rad(angle)
        trans = (WINDOW_W / 2 + trans[0], WINDOW_H / 4 + trans[1])

        # Iterate through the road polygons
        for poly, color in self.unwrapped.road_poly:
            # Convert to pixel coordinates
            poly = [(p[0], p[1]) for p in poly]
            color = [int(c) for c in color]

            # Skip the curbs
            if color == [255, 0, 0] or color == [255, 255, 255]:
                continue

            # Apply the transformation
            poly = [pygame.math.Vector2(c).rotate_rad(angle) for c in poly]
            poly = [(c[0] * zoom + trans[0], c[1] * zoom + trans[1]) for c in poly]
            poly = np.multiply(
                np.asarray(poly[:2]), [STATE_W / WINDOW_W, STATE_H / WINDOW_H]
            )

            road_segments.append(poly)

        return road_segments
