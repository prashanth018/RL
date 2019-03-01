import gym
import numpy as np
import sys

if "../" not in sys.path:
  sys.path.append("../")

from libs.envs.windy_gridworld import WindyGridworldEnv

env = WindyGridworldEnv()

print(env.reset())
env.render()


for i in range(8):
    print(env.step(1))
    env.render()