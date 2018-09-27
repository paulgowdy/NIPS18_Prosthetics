def reward_shaping(action, obs_dict, input_reward, done):


	p_x = obs_dict['body_pos']['pelvis'][0]
	h_x = obs_dict['body_pos']['head'][0]

	l_knee_angle = obs_dict['joint_pos']['knee_l'][0]
	r_knee_angle = obs_dict['joint_pos']['knee_r'][0]


	lean_penalty = 10.0 * min(0.3, max(0, p_x - h_x - 0.3))
	left_knee_bend_penalty = 10.0 * max(0, l_knee_angle + 0.1)# - 0.1)
	right_knee_bend_penalty = 10.0 * max(0, r_knee_angle + 0.1)# - 0.1)

	a_list = list(action)

	muscle_activation_penalty = 0.1 * sum([i ** 2 for i in a_list])

	#print('input reward:', input_reward)
	#print('pelvis x', p_x)
	#print('head x', h_x)
	#print(lean_penalty)
	#print('knee angles:', l_knee_angle, r_knee_angle)
	#print('knee penalty', left_knee_bend_penalty, right_knee_bend_penalty)

	#print(obs_dict['body_pos']['pros_foot_r'])
	#print(obs_dict['body_pos']['toes_l'])
	#print(obs_dict['body_pos']['talus_l'])
	#print('')

	r = input_reward - lean_penalty - left_knee_bend_penalty - right_knee_bend_penalty - muscle_activation_penalty

	#print('learning reward:', r)
	#print('')

	d = done

	# To consider:
	# If feet cross: done and penalty
	# reward based on keeping COM over the line between the feet
	# something about lifting the feet


	return r, d






'''
def reward_shaping(obs_dict, input_reward):
	# simple version...

	# Assuming that:
	# X is along the direction of movement
	# Y is up and down
	# Z is Left and right

	#com_z = obs_dict['misc']['mass_center_pos'][2]

	#head_x = obs_dict['body_pos']['head'][0]
	pelvis_x = obs_dict['body_pos']['pelvis'][0]
	pelvis_y = obs_dict['body_pos']['pelvis'][1]

	head_x = obs_dict['body_pos']['head'][0]

	move_forward = 10.0 * pelvis_x
	stay_up = 10.0 * max(0, 0.9 - pelvis_y)
	lean_forward_penalty = 5.0 * min(0.3, max(0, pelvis_x - head_x - 0.3))

	return move_forward + stay_up + lean_forward_penalty


def reward_shaping(obs_dict, input_reward):

	# Assuming that:
	# X is along the direction of movement
	# Y is up and down
	# Z is Left and right

	com_z = obs_dict['misc']['mass_center_pos'][2]

	head_x = obs_dict['body_pos']['head'][0]
	pelvis_x = obs_dict['body_pos']['pelvis'][0]

	head_z = obs_dict['body_pos']['head'][2]
	pelvis_z = obs_dict['body_pos']['pelvis'][2]

	# Not sure what the units are here...
	# Just using the penalty from the stanford guy
	l_knee_angle = obs_dict['joint_pos']['knee_l'][0]
	r_knee_angle = obs_dict['joint_pos']['knee_r'][0]


	lean_forward_penalty = 0.05 * min(0.3, max(0, pelvis_x - head_x - 0.3))
	dont_lean_sideways_penalty = 0 #1.0 * (head_z - pelvis_z) ** 2
	stay_centered_penalty = 0.1 * (pelvis_z ** 2)
	left_knee_bend_penalty = 0.50 * max(0, l_knee_angle)# - 0.1)
	right_knee_bend_penalty = 0.50 * max(0, r_knee_angle)# - 0.1)

	#print(lean_forward_penalty, dont_lean_sideways_penalty, stay_centered_penalty, left_knee_bend_penalty)

	penalty = lean_forward_penalty + dont_lean_sideways_penalty + stay_centered_penalty + left_knee_bend_penalty + right_knee_bend_penalty

	return input_reward + penalty


def reward_shaping(obs_dict, input_reward):

	# Assuming that:
	# X is along the direction of movement
	# Y is up and down
	# Z is Left and right

	com_z = obs_dict['misc']['mass_center_pos'][2]

	head_x = obs_dict['body_pos']['head'][0]
	pelvis_x = obs_dict['body_pos']['pelvis'][0]

	head_z = obs_dict['body_pos']['head'][2]
	pelvis_z = obs_dict['body_pos']['pelvis'][2]

	# Not sure what the units are here...
	# Just using the penalty from the stanford guy
	l_knee_angle = obs_dict['joint_pos']['knee_l'][0]


	lean_forward_penalty = 1.0 * min(0.3, max(0, pelvis_x - head_x - 0.3))
	dont_lean_sideways_penalty = 1.0 * (head_z - pelvis_z) ** 2
	stay_centered_penalty = 1.0 * (pelvis_z ** 2)
	left_knee_bend_penalty = 0.30 * max(0, l_knee_angle - 0.1)

	#print(lean_forward_penalty, dont_lean_sideways_penalty, stay_centered_penalty, left_knee_bend_penalty)

	penalty = lean_forward_penalty + dont_lean_sideways_penalty + stay_centered_penalty + left_knee_bend_penalty

	return input_reward + penalty

'''
