import gym
from agents import evaluate
from agents.ppo_agent import PPOAgent, CustomCarRacing


if __name__ == '__main__':

    # Create environment
    env = CustomCarRacing()
    env.seed(0)

    # Create and train agent
    ppo = PPOAgent(env)
    ppo.train(episodes=3000)

    # Evaluate agent
    evaluate(env, ppo, 'video.mp4')
