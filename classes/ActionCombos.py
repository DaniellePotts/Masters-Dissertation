import itertools
from itertools import combinations

import collections

import itertools as it

class ActionCombos:
	def get_default_actions(self, actions):
		sample_actions = collections.OrderedDict()

		sample_actions['attack'] = [0,1]
		sample_actions['back'] = [0,1]
		sample_actions['camera'] = [0.,0.]
		sample_actions['forward'] = [0,1]
		sample_actions['jump'] = [0,1]
		sample_actions['left'] = [0,1]
		sample_actions['place'] = [0,1]
		sample_actions['right'] = [0,1]
		sample_actions['sneak'] = [0,1]
		sample_actions['sprint'] = [0,1]

		return sample_actions
	
	def get_all_action_combos(self, actions):
		sample_actions = get_default_actions(actions)
		allNames = sorted(sample_actions)
		combinations = it.product(*(sample_actions[Name] for Name in allNames))
		return list(combinations)