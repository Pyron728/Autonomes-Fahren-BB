import argparse

import gymnasium as gym
import numpy as np
from car import Car
from env_wrapper import CarRacingEnvWrapper


def evaluate(env, eval_runs=50, eval_length=600):

    episode_rewards = []
    for episode in range(eval_runs):

        seed = int(np.random.randint(0, int(1e6)))
        state_image, info = env.reset(seed=seed)

        car = Car()

        episode_rewards.append(0.0)
        for t in range(eval_length):

            action = car.next_action(state_image, info)

            state_image, r, done, trunc, info = env.step(action)
            episode_rewards[-1] += r

            if done or trunc:
                break

        print(
            f"episode: {episode:02d}     "
            f"seed: {seed:06d}     "
            f"reward: {episode_rewards[-1]:06.2F}     "
            f"avg 5 reward: {np.mean(np.asarray(episode_rewards[-5:])):06.2f}     "
            f"avg reward: {np.mean(np.asarray(episode_rewards)):06.2f}     "
        )

    print("---------------------------")
    print(" avg score: %f" % (np.mean(np.asarray(episode_rewards))))
    print(" std diff:  %f" % (np.std(np.asarray(episode_rewards))))
    print(" max score: %f" % (np.max(np.asarray(episode_rewards))))
    print(" min score: %f" % (np.min(np.asarray(episode_rewards))))
    print("---------------------------")
    print(" top 5 avg score: %f" % (np.mean(np.sort(np.asarray(episode_rewards))[-5:])))
    print(" low 5 avg score: %f" % (np.mean(np.sort(np.asarray(episode_rewards))[:5])))
    print("---------------------------")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_display", action="store_true", default=False)
    parser.add_argument("--domain_randomize", action="store_true", default=False)
    args = parser.parse_args()

    render_mode = "rgb_array" if args.no_display else "human"
    env = CarRacingEnvWrapper(
        gym.make(
            "CarRacing-v3",
            render_mode=render_mode,
            domain_randomize=args.domain_randomize,
        )
    )

    evaluate(env, eval_length=1000)

    env.reset()


if __name__ == "__main__":
    main()
