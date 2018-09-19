import pickle

with open('data_saves/observation_list_elements.p', 'rb') as f:

	z = pickle.load(f)

for i in z:
	print(i)