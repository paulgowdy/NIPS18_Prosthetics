import pickle
import matplotlib.pyplot as plt 

with open('reward_history_gpu.p', 'rb') as f:

	z = pickle.load(f)

plt.figure()
plt.plot(z)
plt.show()