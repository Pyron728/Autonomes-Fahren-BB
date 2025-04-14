import argparse

import gymnasium as gym
import numpy as np
from env_wrapper import CarRacingEnvWrapper
from input_controller import InputController
from longitudinal_control import LongitudinalControl
from matplotlib import pyplot as plt

fig = plt.figure()
plt.ion()
plt.show()


def run(env, input_controller: InputController):
    longitudinal_control = LongitudinalControl()

    seed = int(np.random.randint(0, int(1e6)))
    state_image, info = env.reset(seed=seed)
    total_reward = 0.0

    speed_history = []
    target_speed_history = []

    while not input_controller.quit:
        target_speed = longitudinal_control.predict_target_speed(
            info["trajectory"], info["speed"], input_controller.steer
        )
        acceleration, braking = longitudinal_control.control(
            info["speed"], target_speed, input_controller.steer
        )

        speed_history.append(info["speed"])
        target_speed_history.append(target_speed)

        # Longitudinal control plot
        plt.gcf().clear()
        plt.plot(speed_history, c="green")
        plt.plot(target_speed_history)
        try:
            fig.canvas.flush_events()
        except:
            pass

        # Step the environment
        input_controller.update()
        a = [input_controller.steer, acceleration, braking]
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

            speed_history = []
            target_speed_history = []


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
