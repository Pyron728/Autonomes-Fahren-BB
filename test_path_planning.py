import argparse

import cv2
import gymnasium as gym
import numpy as np
from env_wrapper import CarRacingEnvWrapper
from input_controller import InputController
from path_planning import PathPlanning


def run(env, input_controller: InputController):
    path_planning = PathPlanning()

    seed = int(np.random.randint(0, int(1e6)))
    state_image, info = env.reset(seed=seed)
    total_reward = 0.0

    while not input_controller.quit:
        way_points, curvature = path_planning.plan(
            info["left_lane_boundary"], info["right_lane_boundary"]
        )

        cv_image = np.asarray(state_image, dtype=np.uint8)
        way_points = np.array(way_points, dtype=np.int32)
        for point in way_points:
            if 0 < point[0] < 96 and 0 < point[1] < 84:
                cv_image[int(point[1]), int(point[0])] = [255, 255, 255]
        for point in info["left_lane_boundary"]:
            if 0 < point[0] < 96 and 0 < point[1] < 84:
                cv_image[int(point[1]), int(point[0])] = [255, 0, 0]
        for point in info["right_lane_boundary"]:
            if 0 < point[0] < 96 and 0 < point[1] < 84:
                cv_image[int(point[1]), int(point[0])] = [0, 0, 255]

        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
        cv_image = cv2.resize(cv_image, (cv_image.shape[1] * 6, cv_image.shape[0] * 6))
        cv2.imshow("Car Racing - Path Planning", cv_image)
        cv2.waitKey(1)

        # Step the environment
        input_controller.update()
        a = [
            input_controller.steer,
            input_controller.accelerate,
            input_controller.brake,
        ]
        state_image, r, done, trunc, info = env.step(a)
        total_reward += r

        # Reset environment if the run is skipped
        input_controller.update()
        if done or input_controller.skip:
            print(f"seed: {seed:06d}     reward: {total_reward:06.2F}")

            input_controller.skip = False
            seed = int(np.random.randint(0, int(1e6)))
            state_image, info = env.reset(seed=seed)
            total_reward = 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_display", action="store_true", default=False)
    args = parser.parse_args()

    render_mode = "rgb_array" if args.no_display else "human"
    env = CarRacingEnvWrapper(
        gym.make("CarRacing-v3", render_mode=render_mode, domain_randomize=False)
    )
    input_controller = InputController()

    run(env, input_controller)
    env.reset()


if __name__ == "__main__":
    main()
