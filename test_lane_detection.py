import argparse

import cv2
import gymnasium as gym
import numpy as np
from env_wrapper import CarRacingEnvWrapper
from input_controller import InputController
from lane_detection import LaneDetection


def run(env, input_controller: InputController):
    lane_detection = LaneDetection()

    seed = int(np.random.randint(0, int(1e6)))
    state_image, info = env.reset(seed=seed)
    total_reward = 0.0

    while not input_controller.quit:
        lane_detection.detect(state_image)

        cv_image = np.asarray(lane_detection.debug_image, dtype=np.uint8)
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
        cv_image = cv2.resize(cv_image, np.asarray(state_image.shape[:2]) * 6)
        cv2.imshow("Car Racing - Lane Detection", cv_image)
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
