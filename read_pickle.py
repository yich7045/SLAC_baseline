import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import collections


object = pd.read_pickle(r'evaluation_rewards.pkl')
steps = pd.read_pickle(r'evaluation_steps.pkl')
object = savgol_filter(object, 31, 3) # window size 51, polynomial order 3
plt.plot(steps, object)
plt.show()

reward_list = []
success_rate = []
total_reward = []
length = 100
for i in object:
    total_reward.append(i)
    if len(success_rate) <= length:
        reward_list.append(i)
    else:
        reward_list = collections.deque(reward_list)
        reward_list.rotate(-1)
        reward_list = list(reward_list)
        reward_list[-1] = i

    success = sum(1 for i in reward_list if i > 1)
    success_rate.append(success/length)

plt.plot(steps, success_rate)
plt.title('SLAC')
plt.xlabel('steps')
plt.show()

plt.plot(object)
plt.plot(filterd_reward)
plt.ylabel('reward')
plt.xlabel('episode')
plt.title('SLAC')
plt.savefig('slac.png')
plt.show()