import gym
from agents import evaluate
from agents.ppo_agent import PPOAgent, CustomCarRacing


if __name__ == '__main__':

    # Create environment
    env = gym.wrappers.TimeLimit(CustomCarRacing(), 1000)
    env.seed(0)

    # Create and train agent
    ppo = PPOAgent(env)
    ppo.train()

    # Evaluate agent
    evaluate(env, ppo, 'video.mp4')
