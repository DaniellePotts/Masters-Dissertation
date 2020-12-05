import minerl

from ActionCombos import get_actions

def format_states_batch(states):
    final_states = []
    for i in range(len(states)):
        final_states.append(states[0]['pov'])
    
    return final_states

def get_unique_actions(actions_dict):
  unique_actions = []
  for key in actions_dict.keys():
    unique_actions.append(key)
  
  return unique_actions

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

def parse_load_data(data, environment):
  states = []

  for d in data:
    sequence_length = len(d['actions'])
    for i in range(0, sequence_length):
      states.append({
          'done':d['dones'][i],
          'action':d['actions'][i],
          'reward':d['rewards'][i],
          'curr_obs':{
              'obs':d['curr_states']['pov'][i]
          },
          'next_states':{
              'obs':d['next_states']['pov'][i]
          }
      })

      if(environment.lower() != 'treechop'):
          states[i]['curr_obs']['compassAngle']=d['curr_states']['compassAngle'][i],
          states[i]['curr_obs']['inventory']={'dirt':d['curr_states']['inventory']['dirt'][i]}

          states[i]['next_states']['compassAngle']=d['next_states']['compassAngle'][i],
          states[i]['next_states']['inventory']={'dirt':d['next_states']['inventory']['dirt'][i]}

  
  return states

def parse_done(done):
  if done == 'True':
    return 1
  else:
    return 0

def parse_actions(actions):
  keys = actions.keys()
  
  parsed = []

  for key in keys:
    if key != 'camera':
    #   parsed.append(actions[key][0])
    #   parsed.append(actions[key][1])
    # else:
      parsed.append(actions[key])
  return parsed



def parse_navigate(current_state, action, next_state, reward, done, index):
  return {
    'curr_state':{'pov':current_state['pov'][0], 'compassAngle':current_state['compassAngle'][0], 'inventory':current_state['inventory']},
    'next_state':{'pov':next_state['pov'][0], 'compassAngle':next_state['compassAngle'][0], 'inventory':next_state['inventory']},
    'action':action,
    'reward':reward,
    'done':done,
    'sequence':index
  }


def parse_treechop(current_state, action, next_state, reward, done, index):
    return {
      'curr_state':{'pov':current_state['pov'][0]},
      'next_state':{'pov':next_state['pov'][0]},
      'action':action,
      'reward':reward,
      'done':done,
      'sequence':index
  }

def parse_sequences(sequences):
  sequences_copy = sequences
  reward_length = len(sequences[0]['reward'][0])
  done_length = len(sequences[0]['done'][0])
  action_length = len(sequences[0]['action']['camera'][0])
  sequences_length = len(sequences)
  sequences_2 = []
  

  for i in range(0, 100):
    actions = get_actions(sequences_copy[i]['action'], action_length)
    rewards = get_rewards(sequences_copy[i]['reward'], reward_length)
    dones = get_dones(sequences_copy[i]['done'], done_length)
    sequences_copy[i]['curr_state']['pov'] = sequences_copy[i]['curr_state']['pov']
    if 'compassAngle' in sequences_copy[i]['curr_state']:
      sequences_copy[i]['curr_state']['compassAngle'] = sequences_copy[i]['curr_state']['compassAngle']
      sequences_copy[i]['next_state']['compassAngle'] = sequences_copy[i]['next_state']['compassAngle']
    if 'inventory' in sequences_copy[i]['next_state']:
      sequences_copy[i]['curr_state']['inventory']["dirt"] = sequences_copy[i]['curr_state']['inventory']["dirt"][0]
      sequences_copy[i]['next_state']['inventory']["dirt"] = sequences_copy[i]['next_state']['inventory']["dirt"][0]

    sequences_2.append({
        'sequence':i,
        'actions':actions,
        'rewards':rewards['rewards'],
        'dones':dones,
        'curr_states':sequences_copy[i]['curr_state'],
        'next_states':sequences_copy[i]['next_state'],
        'total_reward':rewards['total_reward']
    })
  
  return sequences_2

def get_dones(dones, dones_length):
  all_done = []

  for i in range(0, dones_length):
    all_done.append(str(dones[0][i]))

  return all_done

def get_rewards(rewards, rewards_length):
  all_rewards = {
      'rewards':[],
      'total_reward':0
  }

  for i in range(0, rewards_length):
    all_rewards['rewards'].append(float(rewards[0][i]))
    all_rewards['total_reward'] = all_rewards['total_reward'] + rewards[0][i]
  
  return all_rewards

def parse_demo(env):
    data = minerl.data.make(env)
    sequences = []
    index = 0

    for current_state, action, reward, next_state, done \
        in data.batch_iter(
            batch_size=1, num_epochs=1, seq_len=32):
            #  if ("navigate" in env.lower()):
            #    sequences.append(parse_navigate(current_state, action, next_state, reward, done, index))
            #  elif ("treechop" in env.lower()):
            sequences.append(parse_treechop(current_state, action, next_state, reward, done, index))
            index = index + 1
    return sequences