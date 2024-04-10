from tensorforce import Environment, Agent, Runner
import numpy as np
from RL_environment import CustomEnvironment   


env = CustomEnvironment()


environment = Environment.create(
    environment=env, max_episode_timesteps=10
)

agent = Agent.create(
    agent='ppo', environment=environment, batch_size=10, learning_rate=1e-2, exploration=0.5, actions={"type":'float', "shape":(2,), "min_value":-1.0, "max_value":1.0}
)

# runner = Runner(
#     agent=agent,
#     environment=environment,
#     max_episode_timesteps=10
# )
# runner.run(num_episodes=100, evaluation=True)# , save_best_agent=True)

# Train for 100 episodes
for _ in range(100):
    episode_states = list()
    episode_internals = list()
    episode_actions = list()
    episode_terminal = list()
    episode_reward = list()

    states = environment.reset()
    internals = agent.initial_internals()
    terminal = False
    while not terminal:
        episode_states.append(states)
        episode_internals.append(internals)
        actions, internals = agent.act(
            states=states, internals=internals, independent=True
        )
        episode_actions.append(actions)
        states, terminal, reward = environment.execute(actions=actions)
        episode_terminal.append(terminal)
        episode_reward.append(float(reward))

    agent.experience(
        states=episode_states, internals=episode_internals,
        actions=episode_actions, terminal=episode_terminal,
        reward=episode_reward
    )

    action_bounds = (np.min(episode_actions), np.max(episode_actions))

    print(f"ep {_}: highest reward: {np.max(episode_reward):.2f}, actions: {action_bounds[0]:.2f}, {action_bounds[1]:.2f}")

    agent.update()


# Evaluate for 100 episodes
recorded_states = []
recorded_rewards = []
recorded_actions = []
sum_rewards = 0.0
for _ in range(10):
    states = environment.reset()
    internals = agent.initial_internals()
    terminal = False
    while not terminal:
        actions, internals = agent.act(
            states=states, internals=internals,
            independent=True, deterministic=True
        )
        states, terminal, reward = environment.execute(actions=actions)
        sum_rewards += reward

        recorded_states += [states]
        recorded_rewards += [reward]
        recorded_actions += [actions]


print('Mean episode reward:', sum_rewards / 100)

recorded_states = np.array(recorded_states)
recorded_rewards = np.array(recorded_rewards)
recorded_actions = np.array(recorded_actions)

# Close agent and environment
agent.close()
environment.close()


import matplotlib.pyplot as plt

fig = plt.figure()
plt.plot(recorded_actions[:,0], label="action dim 1")
plt.plot(recorded_actions[:,1], label="action dim 2")
plt.legend()
plt.title("actios")
plt.savefig("actions.png", dpi=300)
plt.close()

fig = plt.figure()
plt.plot(recorded_states)
plt.title("states")
plt.savefig("states.png", dpi=300)
plt.close()

fig = plt.figure()
plt.plot(recorded_rewards)
plt.title("rewards")
plt.savefig("rewards.png", dpi=300)
plt.close()



#runner.close()


print("done")
