import sys
import gym
import gym_environments
import numpy as np
from agent_sarsa import SARSA
from agent_exsarsa import ExpectedSARSA


def calculate_states_size(env):
    max = env.observation_space.high
    min = env.observation_space.low
    sizes = (max - min) * np.array([10, 100]) + 1
    return int(sizes[0]) * int(sizes[1])


def calculate_state(env, value):
    min = env.observation_space.low
    values = (value - min) * np.array([10, 100])
    return int(values[1]) * 19 + int(values[0])


def run(env, agent, selection_method, episodes):
    for episode in range(1, episodes + 1):
        observation, _ = env.reset()
        action = agent.get_action(calculate_state(
            env, observation), selection_method)
        terminated, truncated = False, False
        total_reward = 0
        while not (terminated or truncated):
            new_observation, reward, terminated, truncated, _ = env.step(
                action)
            next_action = agent.get_action(
                calculate_state(env, new_observation), selection_method
            )
            agent.update(
                calculate_state(env, observation),
                action,
                calculate_state(env, new_observation),
                next_action,
                reward,
                terminated,
                truncated,
            )
            observation = new_observation
            action = next_action
            total_reward += reward

    return total_reward


if __name__ == "__main__":
    episodes = 4000 if len(sys.argv) == 1 else int(sys.argv[1])

    env = gym.make("MountainCar-v0")

    sarsa = ExpectedSARSA(
        calculate_states_size(env),
        env.action_space.n,
        alpha=0.1,
        gamma=0.9,
        epsilon=0.1,
    )

    exsarsa = ExpectedSARSA(
        calculate_states_size(env),
        env.action_space.n,
        alpha=0.1,
        gamma=0.9,
        epsilon=0.1,
    )

    # Train
    run(env, sarsa, "epsilon-greedy", episodes)
    env.close()

    # Train
    env = gym.make("MountainCar-v0")
    run(env, exsarsa, "epsilon-greedy", episodes)
    env.close()

    # Play
    total_reward = 0
    env = gym.make("MountainCar-v0")
    for _ in range(10):
        reward = run(env, sarsa, "greedy", 1)
        total_reward += reward
        sarsa.render()
    env.close()
    print(total_reward/10)

    # Play
    total_reward = 0
    env = gym.make("MountainCar-v0")
    for _ in range(10):
        reward = run(env, exsarsa, "greedy", 1)
        total_reward += reward
        exsarsa.render()
    print(total_reward/10)
    env.close()
