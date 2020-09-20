import minerl
import gym

import numpy as np

import collections

import pickle
import json

import os
from os import path

import pickle

os.environ['MINERL_DATA_ROOT']='./content/data'

class MineRLDataPreProcessing:
	def __init__(self, env):
		if (path.exists('./content/data/download/v3/{}.tar'.format(env)) == False):
			print('Retrieving dataset {}'.format(env))
			os.system('python3 -m minerl.data.download "{}"'.format(env))
		else:
			print('Dataset {} already exists.'.format(env))
		print('Setting data...')
		self.data = minerl.data.make(env)
	def parse_demonstration_data(self, pickle_file_path, pickle_file_name):
		print('Parsing batches')
		sequences = self.parse_batches()
		print('Batches parsed - parsing sequences')
		parsed_sequences = self.parse_sequences(sequences)
		print('Sequences parsed. Saving parsed data')
		pickle.dump(parsed_sequences, open('{}/{}'.format(pickle_file_path, pickle_file_name), 'wb'))
	def parse_batches(self):
		sequences = []
		index = 0

		for current_state, action, reward, next_state, done \
			in self.data.batch_iter(
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
	def parse_sequences(self, sequences):
		reward_length = len(sequences[0]['reward'][0])
		done_length = len(sequences[0]['done'][0])
		action_length = len(sequences[0]['action']['camera'][0])
		sequences_length = len(sequences)
		parsed_sequences = []
		
		for i in range(0, 100):
			actions = self.get_actions(sequences[i]['action'], action_length)
			rewards = self.get_rewards(sequences[i]['reward'], reward_length)
			dones = self.get_dones(sequences[i]['done'], done_length)
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
	def get_actions(self, actions, action_length):
		keys = list(actions.keys())
		all_actions = []

		for i in range(0, action_length):
			a = collections.OrderedDict()

			for key in keys:
				vals = 0
				if isinstance(actions[key][0][i], (np.ndarray)):
					vals = actions[key][0][i].tolist()
				else:
					if isinstance(actions[key][0][i],(np.int64)):
						vals = int(actions[key][0][i])
				
				a[key] = vals

			all_actions.append(a)

		return all_actions
	def get_rewards(self, rewards, rewards_length):
		all_rewards = {
			'rewards':[],
			'total_reward':0
		}

		for i in range(0, rewards_length):
			all_rewards['rewards'].append(float(rewards[0][i]))
			all_rewards['total_reward'] = all_rewards['total_reward'] + rewards[0][i]
		
		return all_rewards
	def get_dones(self, dones, dones_length):
		all_done = []

		for i in range(0, dones_length):
			all_done.append(str(dones[0][i]))

		return all_done