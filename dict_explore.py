import opensim as osim
from osim.env import ProstheticsEnv

import matplotlib.pyplot as plt 


env = ProstheticsEnv(visualize=True)

#env.change_model(model = '2D', prosthetic = False, difficulty = 0, seed = None)

#observation = env.reset(project = True)

#print(observation)

observation = env.reset(project = False)

#print('')
#print(observation)

'''
#max(0,k-0.1)
rk = []
lk = []


for i in range(50):

	a = env.action_space.sample()

	# L Medial Hamstring
	a[11] = 1

	# L Lateral Hamstring
	a[12] = 1

	a[13] = 0
	a[14] = 0
	a[15] = 0

	observation, reward, done, info = env.step(a, project = False)


	#print(observation)

	print(i, observation['joint_pos']['knee_r'], observation['joint_pos']['knee_l'])
	print(i, max(0,observation['joint_pos']['knee_r'][0]-0.1), max(0,observation['joint_pos']['knee_l'][0]-0.1))
	print('')

	rk.append(observation['joint_pos']['knee_r'])
	lk.append(observation['joint_pos']['knee_l'])

	obs = observation

	#print(obs[18], obs[23])
	#print('joints:', observation['joint_pos']['knee_r'], observation['joint_pos']['knee_l'])


	#_ = input('>')


plt.figure()
plt.plot(rk)
plt.plot(lk)
plt.show()

'''

outer_keys = observation.keys()



'''
for ok in outer_keys:

	inner_keys = observation[ok].keys()

	print(ok)

	for ik in inner_keys:

		

		print(ik, observation[ok][ik])
			

	print('')
'''

obs = []

def rel_to_A(i, A):

	return [A[x] - i[x] for x in range(len(i))]



obs.extend(observation['misc']['mass_center_pos']) # 3
obs.extend(observation['misc']['mass_center_vel']) # 3
obs.extend(observation['misc']['mass_center_acc']) # 3

# joint body, positions and vels relative to pelvis



# Absolute Joint Positions
obs.extend(observation['joint_pos']['ground_pelvis'])

obs.extend(observation['joint_pos']['hip_r'])
obs.extend(observation['joint_pos']['knee_r'])
obs.extend(observation['joint_pos']['ankle_r'])

obs.extend(observation['joint_pos']['hip_l'])
obs.extend(observation['joint_pos']['knee_l'])
obs.extend(observation['joint_pos']['ankle_l'])



#print(obs[18], obs[23])
#print('joints:', observation['joint_pos']['knee_r'], observation['joint_pos']['knee_l'])


'''

# Relative Joint Positions
#print(observation['joint_pos']['ground_pelvis'])
obs.extend(observation['joint_pos']['ground_pelvis']) # 6 elements

#print(rel_to_A(observation['joint_pos']['hip_r'], observation['body_pos']['pelvis']))
obs.extend(rel_to_A(observation['joint_pos']['hip_r'], observation['body_pos']['pelvis'])) # 3e
obs.extend(rel_to_A(observation['joint_pos']['knee_r'], observation['body_pos']['pelvis'])) # 1e 18
obs.extend(rel_to_A(observation['joint_pos']['ankle_r'], observation['body_pos']['pelvis'])) # 1e

obs.extend(rel_to_A(observation['joint_pos']['hip_l'], observation['body_pos']['pelvis'])) # 3e
obs.extend(rel_to_A(observation['joint_pos']['knee_l'], observation['body_pos']['pelvis'])) # 1e 23
obs.extend(rel_to_A(observation['joint_pos']['ankle_l'], observation['body_pos']['pelvis'])) # 1e
'''

# Absolute Joint Vel

obs.extend(observation['joint_vel']['ground_pelvis'])

obs.extend(observation['joint_vel']['hip_r'])
obs.extend(observation['joint_vel']['knee_r'])
obs.extend(observation['joint_vel']['ankle_r'])

obs.extend(observation['joint_vel']['hip_l'])
obs.extend(observation['joint_vel']['knee_l'])
obs.extend(observation['joint_vel']['ankle_l'])

# Absolute Joint Acc

obs.extend(observation['joint_acc']['ground_pelvis'])

obs.extend(observation['joint_acc']['hip_r'])
obs.extend(observation['joint_acc']['knee_r'])
obs.extend(observation['joint_acc']['ankle_r'])

obs.extend(observation['joint_acc']['hip_l'])
obs.extend(observation['joint_acc']['knee_l'])
obs.extend(observation['joint_acc']['ankle_l'])

#print(len(obs))

b = ['body_pos', 'body_vel', 'body_acc'] #, 'body_pos_rot', 'body_vel_rot', 'body_acc_rot'
parts = ['pelvis', 'femur_r', 'pros_tibia_r', 'pros_foot_r', 'femur_l', 'tibia_l', 'talus_l', 'calcn_l', 'toes_l', 'torso', 'head']

for i in b:

	print(i, 'len obs:', len(obs))

	for j in parts:

		obs.extend(observation[i][j])


'''
forces_subkeys = observation['forces'].keys()

for k in forces_subkeys:

	obs.extend(observation['forces'][k])
'''

# calculate knee angles
# L



# pelvix x,y,z = 57,58,59
# head x,y,z = 87.88.89


#print(obs)
print('obs len:', len(obs))

'''
print(observation['body_pos']['pelvis'])
print(observation['body_pos']['head'])

print(obs[57])
print(obs[58])
print(obs[59])

print(obs[87])
print(obs[88])
print(obs[89])
'''