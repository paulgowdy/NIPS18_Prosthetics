import numpy as np
import pickle

print("Loading means and stds")

with open('data_saves/means_1.p', 'rb') as f:

	means = pickle.load(f)

with open('data_saves/stds_1.p', 'rb') as f:

	stds = pickle.load(f)

def flatten(d):
	res = []  # Result list

	if isinstance(d, dict):
		for key, val in sorted(d.items()):

			#print(key)
			res.extend(flatten(val))

	elif isinstance(d, list):
		res = d
	else:
		res = [d]

	return res

def xz_relative_pos(obs_list):

	pelvis_x = obs_list[78]
	pelvis_z = obs_list[80]

	x_ids = [66, 69, 72, 75, 81, 84, 87, 90, 93, 96, 330]
	z_ids = [a + 2 for a in x_ids]

	relative_obs_list = list(obs_list)

	for i in range(len(x_ids)):

		relative_obs_list[x_ids[i]] -= pelvis_x
		relative_obs_list[z_ids[i]] -= pelvis_z

	return relative_obs_list

def process_obs_dict(d):

	# Flatten
	f_obs = flatten(d)

	# Relative X, Z
	rel_obs = xz_relative_pos(f_obs)

	# Normalize
	rel_obs = np.array(rel_obs)
	rel_obs -= means
	rel_obs /= stds

	#Target Velocity Vector
	rel_obs = np.append(rel_obs, [3.0, 0.0, 0.0])
	# ON difficulty >0
	#rel_obs.append(d['target_vel'])



	return rel_obs





'''
def rel(listA, listB):

	return [listA[i] - listB[i] for i in range(len(listA))]

def process_obs_dict(obs_dict):

	p_p = obs_dict['body_pos']['pelvis']
	p_v = obs_dict['body_vel']['pelvis']
	p_a = obs_dict['body_acc']['pelvis']

	obs = []

	obs.extend(rel(obs_dict['misc']['mass_center_pos'], p_p)) # x, y, z
	obs.extend(rel(obs_dict['misc']['mass_center_vel'], p_v)) # x, y, z
	obs.extend(rel(obs_dict['misc']['mass_center_acc'], p_a)) # x, y, z

	# Absolute Joint Positions
	obs.extend(obs_dict['joint_pos']['ground_pelvis'])

	obs.extend(obs_dict['joint_pos']['hip_r'])
	obs.extend(obs_dict['joint_pos']['knee_r'])
	obs.extend(obs_dict['joint_pos']['ankle_r'])

	obs.extend(obs_dict['joint_pos']['hip_l'])
	obs.extend(obs_dict['joint_pos']['knee_l'])
	obs.extend(obs_dict['joint_pos']['ankle_l'])

	obs.extend(obs_dict['joint_vel']['ground_pelvis'])

	obs.extend(obs_dict['joint_vel']['hip_r'])
	obs.extend(obs_dict['joint_vel']['knee_r'])
	obs.extend(obs_dict['joint_vel']['ankle_r'])

	obs.extend(obs_dict['joint_vel']['hip_l'])
	obs.extend(obs_dict['joint_vel']['knee_l'])
	obs.extend(obs_dict['joint_vel']['ankle_l'])

	# Absolute Joint Acc

	obs.extend(obs_dict['joint_acc']['ground_pelvis'])

	obs.extend(obs_dict['joint_acc']['hip_r'])
	obs.extend(obs_dict['joint_acc']['knee_r'])
	obs.extend(obs_dict['joint_acc']['ankle_r'])

	obs.extend(obs_dict['joint_acc']['hip_l'])
	obs.extend(obs_dict['joint_acc']['knee_l'])
	obs.extend(obs_dict['joint_acc']['ankle_l'])

	b = ['body_pos', 'body_vel', 'body_acc']
	parts = ['femur_r', 'pros_tibia_r', 'pros_foot_r', 'femur_l', 'tibia_l', 'talus_l', 'calcn_l', 'toes_l', 'torso', 'head']
	rel_pel = [p_p, p_v, p_a]

	for i in b:

		for j in parts:

			obs.extend(rel(obs_dict[i][j], rel_pel[b.index(i)]))

	#obs_dict.append(obs_dict['body_pos']['pelvis'][0]) #x
	obs.append(obs_dict['body_pos']['pelvis'][1]) #y
	obs.append(obs_dict['body_pos']['pelvis'][2]) #z

	return np.array(obs)

'''


'''
def process_obs_dict(obs_dict):

	obs = []

	obs.extend(obs_dict['misc']['mass_center_pos']) # x, y, z
	obs.extend(obs_dict['misc']['mass_center_vel']) # x, y, z
	obs.extend(obs_dict['misc']['mass_center_acc']) # x, y, z

	# Absolute Joint Positions
	obs.extend(obs_dict['joint_pos']['ground_pelvis'])

	obs.extend(obs_dict['joint_pos']['hip_r'])
	obs.extend(obs_dict['joint_pos']['knee_r'])
	obs.extend(obs_dict['joint_pos']['ankle_r'])

	obs.extend(obs_dict['joint_pos']['hip_l'])
	obs.extend(obs_dict['joint_pos']['knee_l'])
	obs.extend(obs_dict['joint_pos']['ankle_l'])

	obs.extend(obs_dict['joint_vel']['ground_pelvis'])

	obs.extend(obs_dict['joint_vel']['hip_r'])
	obs.extend(obs_dict['joint_vel']['knee_r'])
	obs.extend(obs_dict['joint_vel']['ankle_r'])

	obs.extend(obs_dict['joint_vel']['hip_l'])
	obs.extend(obs_dict['joint_vel']['knee_l'])
	obs.extend(obs_dict['joint_vel']['ankle_l'])

	# Absolute Joint Acc

	obs.extend(obs_dict['joint_acc']['ground_pelvis'])

	obs.extend(obs_dict['joint_acc']['hip_r'])
	obs.extend(obs_dict['joint_acc']['knee_r'])
	obs.extend(obs_dict['joint_acc']['ankle_r'])

	obs.extend(obs_dict['joint_acc']['hip_l'])
	obs.extend(obs_dict['joint_acc']['knee_l'])
	obs.extend(obs_dict['joint_acc']['ankle_l'])

	b = ['body_pos', 'body_vel', 'body_acc']
	parts = ['pelvis', 'femur_r', 'pros_tibia_r', 'pros_foot_r', 'femur_l', 'tibia_l', 'talus_l', 'calcn_l', 'toes_l', 'torso', 'head']

	for i in b:

		for j in parts:

			obs.extend(obs_dict[i][j])











	return np.array(obs)

'''
