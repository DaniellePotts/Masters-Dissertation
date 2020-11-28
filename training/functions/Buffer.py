import json
import gym

from collections import deque

import numpy as np

from anyrl.rollouts import replay
from ActionCombos import match_actions
from DataHelper import parse_actions, parse_done

from Utils import save_data, load_data
from DataHelper import parse_load_data, parse_actions, parse_done

def populate_buffer(data, replay_buffer, action_combos, environment):
  states = parse_load_data(data, environment)
  # save_data("processed_states.json", states)
  print("Saved processed data!")
  print("Populating buffer...")
  states_length = len(states)

  curr_state_deque = deque()
  next_state_deque = deque()
  dones_deque = deque()
  frames_deque = deque()
  actions_deque = deque()
 
  rep_buffer = []

  rewards = []

  parse_ts = 0
  episode_start_ts = 0
  curr_obs = states[0]['curr_obs']['obs']
  last_state_was_done = False

  for i in range(1, states_length):
    episode_start_ts += 1
    parse_ts += 1

    next_obs = states[i]['curr_obs']['obs']
    reward = states[i]['reward']
    _a = parse_actions(states[i]['action'])
    _a = match_actions(_a, action_combos)
    done = parse_done(states[i]['done'])

    reward = np.sign(reward) * np.log(1.+np.abs(reward))
 
    curr_state_deque.append(curr_obs)
    actions_deque.append(_a)
    rewards.append(reward)
    next_state_deque.append(next_obs)
    dones_deque.append(done)

    print(actions_deque)
    if episode_start_ts > 10:
      add_transition(replay_buffer, curr_state_deque, actions_deque, rewards, next_state_deque, dones_deque, curr_obs)
    if done == 'True':
      add_transition(replay_buffer, curr_state_deque, actions_deque, rewards, next_state_deque, dones_deque, curr_obs)
      curr_obs = states[(i + 1)]['curr_obs']['obs']
      i = i + 2

      curr_state_deque.clear()
      actions_deque.clear()
      rewards.clear()
      next_state_deque.clear()
      dones_deque.clear()

      episode_start_ts = 0
    else:
      curr_obs = next_obs

  add_transition(replay_buffer, curr_state_deque, actions_deque, rewards, next_state_deque, dones_deque, curr_obs)

  return replay_buffer

# def parse_data(data):
# 	states = []

# 	for d in data:
# 		sequence_length = len(d['actions'])
# 		for i in range(0, sequence_length):
# 			states.append({
# 				'done':d['dones'][i],
# 				'action':d['actions'][i],
# 				'reward':d['rewards'][i],
# 				'curr_obs':{
# 					'obs':d['curr_states']['pov'][i],
# 					'compassAngle':d['curr_states']['compassAngle'][i],
# 					'inventory':{'dirt':d['curr_states']['inventory']['dirt'][i]}
# 				},
# 				'next_state':{
# 					'obs':d['next_states']['pov'][i],
# 					'compassAngle':d['next_states']['compassAngle'][i],
# 					'inventory':{'dirt':d['next_states']['inventory']['dirt'][i]}
# 				}
# 			})
	
# 	return states

def parse_actions(actions):
  keys = actions.keys()
  
  parsed = []

  for key in keys:
    if key == 'camera':
        parsed.append([actions[key][0],actions[key][1]])
    else:
      parsed.append(actions[key])
  return parsed

def add_transition(replay_buffer, curr_states, actions, rewards, next_steps, dones, curr_state, empty_deque=False, ns=10, ns_gamma=0.99,is_done=True):
		ns_rew_sum = 0.
		trans = {}

		if empty_deque:
			while len(rewards) > 0:
				for i in range(len(rewards)):
					ns_rew_sum += rewards[i] * ns_gamma ** i
				
				trans['sample'] = [curr_states.popleft(), actions.popleft(), rewards.pop(0),
												next_steps.popleft(), is_done, ns_rew_sum, curr_state]
				replay_buffer.add_sample(trans)
		else:
			for i in range(ns):
				ns_rew_sum += rewards[i] * ns_gamma ** i
			trans['sample'] =  [curr_states.popleft(), actions.popleft(), rewards.pop(0),
								next_steps.popleft(), dones.popleft(), ns_rew_sum, curr_state]
			replay_buffer.add_sample(trans)