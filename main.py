import sys
import gym
import numpy as np
from agent_sarsa import SARSA
from agent_exsarsa import ExpectedSARSA
import time
import datetime
import csv
import matplotlib.pyplot as plt
import multiprocessing

episodes, gamma, epsilon = 4000, 0.9, 0.6

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


def execute_experiment(alpha):
    env = gym.make("MountainCar-v0")
    entries = []
    sarsa = SARSA(
        calculate_states_size(env),
        env.action_space.n,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
    )

    exsarsa = ExpectedSARSA(
        calculate_states_size(env),
        env.action_space.n,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
    )

    # Train
    run(env, sarsa, "epsilon-greedy", episodes)
    env.close()

    # Train
    env = gym.make("MountainCar-v0")
    run(env, exsarsa, "epsilon-greedy", episodes)
    env.close()

    # Play
    total_reward, plays = 0, 100
    env = gym.make("MountainCar-v0")
    for _ in range(plays):
        reward = run(env, sarsa, "greedy", 1)
        total_reward += reward
    env.close()
    entries.append(('greedy', 'epsilon-greedy', alpha, gamma,
                   epsilon, episodes, total_reward/plays, 'sarsa'))
    print(
        f'SARSA AVG_Reward {total_reward/plays} for {gamma:.2f} and {plays} and {episodes} episodes')

    # Play
    total_reward = 0
    env = gym.make("MountainCar-v0")
    for _ in range(plays):
        reward = run(env, exsarsa, "greedy", 1)
        total_reward += reward
    entries.append(('greedy', 'epsilon-greedy', alpha, gamma, epsilon,
                   episodes, total_reward/plays, 'expected_sarsa'))
    print(
        f'Expected SARSA AVG_Reward {total_reward/plays} for {gamma:.2f} and {plays} and {episodes} episodes')
    env.close()

    return entries


if __name__ == "__main__":
    start = time.time()
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    result = pool.map_async(execute_experiment, [i for i in np.arange(0.0, 1.1, 0.1)])
    values = result.get()

    with open('results.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['play_algorithm', 'learning_algorithm', 'alpha', 'gamma',
                        'epsilon', 'learning_episodes', 'avg_reward', 'agent'])
        for entries in values:
            for entry in entries:
                writer.writerow(entry)
            
        plt.plot([i for i in np.arange(0.0, 1.1, 0.1)], [entry[6]
        for entries in values for entry in entries if entry[7] == 'sarsa'], label='SARSA')
        plt.plot([i for i in np.arange(0.0, 1.1, 0.1)], [entry[6]
        for entries in values for entry in entries if entry[7] == 'expected_sarsa'], label='Expected SARSA')

    print(f'Time taken: {datetime.timedelta(seconds=time.time() - start)}')

    plt.xlabel(f'Alpha\nGamma={gamma} Epsilon={epsilon}')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.show()
