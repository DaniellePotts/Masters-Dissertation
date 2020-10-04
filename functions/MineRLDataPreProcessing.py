import minerl
import gym

import numpy as np

import collections

import pickle
import json

import os
from os import path

import pickle

def parse_unique_placements(unique_places):
  if (len(unique_places) == 2):
    return [0, 1] #0 == none, 1 == whatever the other item is
  else:
    int_placements = []
    for i in len(unique_actions):
      int_placements.append(i)
    return int_placements
	
def parse_demo(data):
  sequences = []
  i = 0

  for current_state, action, reward, next_state, done \
      in data.batch_iter(
          batch_size=1, num_epochs=1, seq_len=32):
          sequences.append({
              'curr_state':{'pov':current_state['pov'][0], 'compassAngle':current_state['compassAngle'][0], 'inventory':current_state['inventory']},
              'next_state':{'pov':next_state['pov'][0], 'compassAngle':next_state['compassAngle'][0], 'inventory':next_state['inventory']},
              'action':action,
              'reward':reward,
              'done':done,
              'sequence':i
            })
          i = i + 1
  return sequences
def parse_demonstration_data(pickle_file_path, pickle_file_name):
	print('Parsing batches')
	sequences = parse_batches()
	print('Batches parsed - parsing sequences')
	parsed_sequences = parse_sequences(sequences)
	print('Sequences parsed. Saving parsed data')
	pickle.dump(parsed_sequences, open('{}/{}'.format(pickle_file_path, pickle_file_name), 'wb'))
def parse_batches():
	sequences = []
	index = 0

	for current_state, action, reward, next_state, done \
		in data.batch_iter(
			batch_size=1, num_epochs=1, seq_len=32):
			sequences.append({
				'curr_state':{'pov':current_state['pov'][0], 'compassAngle':current_state['compassAngle'][0], 'inventory':current_state['inventory']},
				'next_state':{'pov':next_state['pov'][0], 'compassAngle':next_state['compassAngle'][0], 'inventory':next_state['inventory']},
				'action':action,
				'reward':reward,
				'done':done,
				'sequence':index
				})
			index = index + 1
	return sequences
def parse_sequences(sequences):
	reward_length = len(sequences[0]['reward'][0])
	done_length = len(sequences[0]['done'][0])
	action_length = len(sequences[0]['action']['camera'][0])
	sequences_length = len(sequences)
	parsed_sequences = []
	
	for i in range(0, 100):
		actions = get_actions(sequences[i]['action'], action_length)
		rewards = get_rewards(sequences[i]['reward'], reward_length)
		dones = get_dones(sequences[i]['done'], done_length)
		sequences[i]['curr_state']['pov'] = sequences[i]['curr_state']['pov']
		sequences[i]['curr_state']['compassAngle'] = sequences[i]['curr_state']['compassAngle']
		sequences[i]['curr_state']['inventory']["dirt"] = sequences[i]['curr_state']['inventory']["dirt"][0]
	
		sequences[i]['next_state']['pov'] = sequences[i]['next_state']['pov']
		sequences[i]['next_state']['compassAngle'] = sequences[i]['next_state']['compassAngle']
		sequences[i]['next_state']['inventory']["dirt"] = sequences[i]['next_state']['inventory']["dirt"][0]

		parsed_sequences.append({
			'sequence':i,
			'actions':actions,
			'rewards':rewards['rewards'],
			'dones':dones,
			'curr_states':sequences[i]['curr_state'],
			'next_states':sequences[i]['next_state'],
			'total_reward':rewards['total_reward']
		})
	
	return parsed_sequences
def get_actions(actions, action_length):
  keys = list(actions.keys())
  all_actions = []

  for i in range(0, action_length):
    a = collections.OrderedDict()

    for key in keys:
      vals = 0
      if (key == 'camera'):
        if i > 0:
            total_prev_angle = (all_actions[(len(all_actions) - 1)][key][0]) + (all_actions[(len(all_actions) - 1)][key][1])
            total_curr_angle = (actions[key][0][(i - 1)][0]) + (actions[key][0][(i - 1)][1])
            if ((total_curr_angle - total_prev_angle) > 11.5):
              vals = actions[key][0][i].tolist()
            else:
              vals = all_actions[(len(all_actions) - 1)][key]
        else:  
          vals = actions[key][0][i].tolist()
      else:
        if isinstance(actions[key][0][i],(np.int64)):
          vals = int(actions[key][0][i])
      
      a[key] = vals

    all_actions.append(a)

  return all_actions

def get_unique_place_actions(sequences):
  unique_places = []
  sequences_length = len(sequences)
  
  for i in range(sequences_length):
    place_length = len(sequences[i]['action']['place'][0])

    if i == 0:
      unique_places.append(sequences[i]['action']['place'][0][0])
    for j in range(place_length):      
        unique_place = sequences[i]['action']['place'][0][j]

        u = [u for u in unique_places if u == unique_place]

        if len(u) == 0:
          unique_places.append(unique_place)
  return unique_places

def get_unqiue_actions(actions_dict):
  unique_actions = []
  for key in actions_dict.keys():
    unique_actions.append(key)
  
  return unique_actions

def get_rewards(rewards, rewards_length):
	all_rewards = {
		'rewards':[],
		'total_reward':0
	}

	for i in range(0, rewards_length):
		all_rewards['rewards'].append(float(rewards[0][i]))
		all_rewards['total_reward'] = all_rewards['total_reward'] + rewards[0][i]
	
	return all_rewards
def get_dones(dones, dones_length):
	all_done = []

	for i in range(0, dones_length):
		all_done.append(str(dones[0][i]))

	return all_done

def get_all_unique_actions(unique_actions,  unique_camera_angles, unique_places=None):
  sample_actions = collections.OrderedDict()

  for action in unique_actions:
    if (action == 'camera'):
      sample_actions[action] = unique_camera_angles
    elif (action == 'place'):
      sample_actions[action] = parse_unique_placements(unique_places)
    else:
      sample_actions[action] = [0, 1]

  return sample_actions