from tensorforce.environments import Environment
from tensorforce.agents import Agent
from tensorforce.execution import Runner

environment = Environment.create(
    environment='gym', level='CartPole', max_episode_timesteps=500
)

environment = Environment.create(environment='gym', level='CartPole-v1')


agent = Agent.create(
    agent='tensorforce', environment=environment, update=64,
    optimizer=dict(optimizer='adam', learning_rate=1e-3),
    objective='policy_gradient', reward_estimation=dict(horizon=20)
)

runner = Runner(
    agent=agent,
    environment=dict(environment='gym', level='CartPole'),
    max_episode_timesteps=500
)

runner.run(num_episodes=200)

runner.run(num_episodes=100, evaluation=True)

runner.close()

print("done")