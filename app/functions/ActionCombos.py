import itertools as it
from itertools import combinations

import collections

import pickle

import pandas as pd
import numpy as np

def match_actions(action, combos):
  return [i for i in range(0, len(combos)) if list(combos[i]) == action][0]
  
def get_all_unique_actions(unique_actions,  unique_camera_angles, unique_places=None):
  sample_actions = collections.OrderedDict()

  for action in unique_actions:
    if (action == 'camera'):
      sample_actions[action] = unique_camera_angles
    elif (action == 'place'):
      sample_actions[action] = parse_unique_placements(unique_actions, unique_places)
    else:
      sample_actions[action] = [0, 1]

  return sample_actions

def parse_unique_placements(unique_actions, unique_places):
  if (len(unique_places) == 2):
    return [0, 1] #0 == none, 1 == whatever the other item is
  else:
    int_placements = []
    for i in len(unique_actions):
      int_placements.append(i)
    return int_placements

def get_all_action_combos(unique_actions, unique_angles, unique_places=None):
	sample_actions = get_all_unique_actions(unique_actions, unique_angles, unique_places)
	allNames = sorted(sample_actions)
	combinations = it.product(*(sample_actions[Name] for Name in allNames))
	return list(combinations)

def match_actions(action, combos):
  return [i for i in range(0, len(combos)) if list(combos[i]) == action][0]

def int_action_to_dict(action_keys, action_ints):
    actions = {}
    if(len(action_keys) == len(action_ints)):
        for i in range(len(action_ints)):
            actions[action_keys[i]] = action_ints[i]
        if('place' in actions):
            actions['place'] = 'none'
        if('camera' not in actions):
            actions['camera'] = [0.,0.]
    return actions

def get_unique_angles(data):
  parsed_data_length = len(data)

  unique_angles = []
 
  for i in range(parsed_data_length):
    action_length = len(data[i]['actions'])
    for j in range(action_length):
      if j == 0:
        unique_angles.append(data[i]['actions'][j]['camera'])
      else:
        total = (data[i]['actions'][j]['camera'][0]) + (data[i]['actions'][j]['camera'][1])
        
        u = [u for u in unique_angles if (u[0] + u[1]) == total]

        if (len(u) == 0):
          unique_angles.append(data[i]['actions'][j]['camera'])
  return unique_angles

def load_combos(combos_file):
	return pickle.load(open("../resources/{0}.sav".format(combos_file), 'rb'))

def action_dict_to_ints(action_dict, unique_angles):
    angles_df = pd.DataFrame(unique_angles, columns=['x','y'])
    angles_x = angles_df['x'].values
    angles_values = angles_df.values
    actions = []

    for key in action_dict.keys():
        if key == 'camera':
          closest_x = angles_x[np.abs(np.array(angles_x) - action_dict[key][0]).argmin()]
          both_angles = [angle for angle in angles_values if angle[0] == closest_x][0]
          actions.append([both_angles[0],both_angles[1]])
        elif key == 'place':
            if action_dict[key].upper() != 'NONE':
                actions.append(1)
            else:
                actions.append(0)
        else:
            actions.append(action_dict[key].item())
    return actions

def match_batch_actions(actions, combos, unique_angles):
    converted = []
    for action in actions:
        if (isinstance(action, (int, np.integer))):
            converted.append(action)
        else:
            _c = convert_match_actions(action, combos, unique_angles)
            converted.append(_c)
    return converted

def convert_match_actions(action_dict, combos, unique_angles):
    _a = action_dict_to_ints(action_dict, unique_angles)
    matched_action = match_actions(_a, combos)
    
    return matched_action