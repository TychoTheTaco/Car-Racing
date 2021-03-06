from agents import evaluate
from agents.ppo_agent import PPOAgent
from environments.custom_car_racing import CustomCarRacing


if __name__ == '__main__':

    # Create environment
    env = CustomCarRacing()

    # Create and train agent
    agent = PPOAgent(env)
    agent.train(episodes=2000, batch_size=512)

    # Evaluate agent
    env = CustomCarRacing(step_repeat=1)
    evaluate(env, agent, 'video.mp4')
