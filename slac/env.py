import gym
import minitouch.env

def make_dmc():
    env = gym.make("Pushing-v2")
    env._max_episode_steps = 200
    env.seed(0)
    return env
