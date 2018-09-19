
import pickle
import numpy as np
import time
import osim
from osim.env import ProstheticsEnv as RunEnv

from rpm import rpm # import threading

import tensorflow as tf
import canton as ct
from canton import *

import matplotlib.pyplot as plt

from process_dict_obs import *
from reward_shaping import *

env = RunEnv(visualize=False)

env.change_model(model='3D', prosthetic=True, difficulty=0, seed=None)

observation_d = env.reset(project = False)

print(observation_d.keys())

env.change_model(model='3D', prosthetic=True, difficulty=1, seed=None)

observation_d = env.reset(project = False)

print(observation_d.keys())

print(observation_d['target_vel'])
