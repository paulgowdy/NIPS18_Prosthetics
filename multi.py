from multiprocessing import Process, Pipe

# FAST ENV

# this is a environment wrapper. it wraps the RunEnv and provide interface similar to it. The wrapper do a lot of pre and post processing (to make the RunEnv more trainable), so we don't have to do them in the main program.

#from observation_processor import generate_observation as go

import numpy as np
import math

class fastenv:
    def __init__(self,e,skipcount):
        self.e = e
        #print('pgpg', e.observation_space)
        self.stepcount = 0

        #self.old_observation = None
        self.skipcount = skipcount # 4
    
    
    def obg(self,plain_obs):
        # observation generator
        # derivatives of observations extracted here.
        #print('pg multi.py 21, plain_obs:', len(plain_obs))

        #processed_observation, self.old_observation = go(plain_obs, self.old_observation, step=self.stepcount)

        observation = plain_obs
        obs = []

        obs.extend(observation['misc']['mass_center_pos']) # x, y, z
        obs.extend(observation['misc']['mass_center_vel']) # x, y, z
        obs.extend(observation['misc']['mass_center_acc']) # x, y, z

        # joint body, positions and vels relative to pelvis



        # Absolute Joint Positions
        obs.extend(observation['joint_pos']['ground_pelvis'])

        obs.extend(observation['joint_pos']['hip_r'])
        obs.extend(observation['joint_pos']['knee_r'])
        obs.extend(observation['joint_pos']['ankle_r'])

        obs.extend(observation['joint_pos']['hip_l'])
        obs.extend(observation['joint_pos']['knee_l'])
        obs.extend(observation['joint_pos']['ankle_l'])


        '''

        # Relative Joint Positions
        #print(observation['joint_pos']['ground_pelvis'])
        obs.extend(observation['joint_pos']['ground_pelvis']) # 6 elements

        #print(rel_to_A(observation['joint_pos']['hip_r'], observation['body_pos']['pelvis']))
        obs.extend(rel_to_A(observation['joint_pos']['hip_r'], observation['body_pos']['pelvis'])) # 3e
        obs.extend(rel_to_A(observation['joint_pos']['knee_r'], observation['body_pos']['pelvis'])) # 1e
        obs.extend(rel_to_A(observation['joint_pos']['ankle_r'], observation['body_pos']['pelvis'])) # 1e

        obs.extend(rel_to_A(observation['joint_pos']['hip_l'], observation['body_pos']['pelvis'])) # 3e
        obs.extend(rel_to_A(observation['joint_pos']['knee_l'], observation['body_pos']['pelvis'])) # 1e
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

        b = ['body_pos', 'body_vel', 'body_acc'] #, 'body_pos_rot', 'body_vel_rot', 'body_acc_rot'
        parts = ['pelvis', 'femur_r', 'pros_tibia_r', 'pros_foot_r', 'femur_l', 'tibia_l', 'talus_l', 'calcn_l', 'toes_l', 'torso', 'head']

        for i in b:

            for j in parts:

                obs.extend(observation[i][j])

        '''
        forces_subkeys = observation['forces'].keys()

        for k in forces_subkeys:

            obs.extend(observation['forces'][k])
        '''


        #print('pg multi.py 25, proc_obs:', len(processed_observation))

        return np.array(obs)
    

    def step(self,action):
        action = [float(action[i]) for i in range(len(action))]

        
        for num in action:
            if math.isnan(num):
                print('NaN met',action)
                raise RuntimeError('this is bullshit')

        sr = 0
        real_reward = 0
        
        for j in range(self.skipcount):
            self.stepcount+=1
            oo,r,d,i = self.e.step(action)

            real_reward += r

            headx = oo[87] - oo[57]
            px = oo[57]
            #print('reward shaping!')
            com_z = oo[2]

            #py = oo[2]
            #kneer = oo[7]
            #kneel = oo[10]

            #l_knee = oo[23]

            # height_penalty = max(0, 0.65-py) * 0.1

            lean_penalty = min(0.3, max(0, px-headx-0.3)) * 0.3 #0.03

            center_penalty = 10.0 * (com_z ** 2)

            #joint_penalty = sum([max(0,k-0.1) for k in [kneer,kneel]]) * 0.02
            #knee_penalty = 0.06 * max(0, l_knee - 0.1)

            penalty = lean_penalty + center_penalty #+ knee_penalty#+ joint_penalty# + height_penalty

            # action_penalty = np.mean(np.array(action))*1e-3
            # penalty += action_penalty

            #print(penalty)

            o = oo #self.obg(oo)
            sr += r - penalty

            if d == True:
                break

        # # alternative reward scheme
        # delta_x = oo[1] - self.lastx
        # sr = delta_x * 1
        # self.lastx = oo[1]

        return o,sr,d,i, real_reward

    def reset(self):
        self.stepcount=0
        #self.old_observation = None

        oo = self.e.reset()
        # o = self.e.reset(difficulty=2)
        #self.lastx = oo[1]

        #print(len(oo))
        #print(type(oo))
        #print(oo)

        #o = self.obg(oo)
        return oo
