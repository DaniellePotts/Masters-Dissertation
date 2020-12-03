import numpy as np
import random
import sys
import gym
import minerl
from functions.ActionCombos import load_combos, get_all_action_combos, int_action_to_dict, convert_match_actions, match_batch_actions
from functions.Utils import write_json_file


def get_agent_action(env, model, curr_obs, action_combos, n_action, action_keys, unique_angles_treechop, do_random, level_of_randomness):
    empty_by_one = np.zeros((1, 1))
    empty_exp_action_by_one = np.zeros((1, 2))
    empty_action_len_by_one = np.zeros((1, n_action))

    action_was_random = False
    
    if (do_random == True) and (random.random() <= level_of_randomness):
        action_command = env.action_space.sample()
        action_was_random = True
    else:
        temp_curr_obs = np.array(curr_obs)
        temp_curr_obs = temp_curr_obs.tolist()['pov'].reshape(1,temp_curr_obs.tolist()['pov'].shape[0], temp_curr_obs.tolist()['pov'].shape[1], temp_curr_obs.tolist()['pov'].shape[2])
        q, _, _ = model.predict([temp_curr_obs, temp_curr_obs,empty_by_one, empty_exp_action_by_one,empty_action_len_by_one])
        action_command = np.argmax(q)
        
    action_to_store = None
    action_to_take = None
   
    if (isinstance(action_command, (int, np.integer))):
        action_to_store = action_command

        combo = action_combos[action_command]
        action_to_take = int_action_to_dict(action_keys, action_combos[action_command])
    else:
        action_to_store = convert_match_actions(action_command, action_combos, unique_angles_treechop)
        action_to_take = int_action_to_dict(action_keys, action_combos[action_to_store])
        
    return action_to_take, action_was_random

def run_rl_agent(env, model, action_combos, n_action, action_keys, unique_angles_treechop,  do_random, level_of_randomness):
    curr_obs = env.reset()
    done = False
    prev_obs = []
    episode = 1

    waitForMax = 40
    while not done:
        action, action_was_random = get_agent_action(env, model, curr_obs, action_combos, n_action, action_keys, unique_angles_treechop,  do_random, level_of_randomness)
        waitForStep = 1

        if do_random == True:
            write_json_file("actions.json", {'action':action, 'was_random':action_was_random})
        while(waitForStep < waitForMax):
            curr_obs, reward, done, _ = env.step(action)
            waitForStep = waitForStep + 1
            
        episode = episode + 1
        if done: print(done)

def start_environment(environment, port):
    print('Starting up... {0}'.format(environment))
    
    env = gym.make(environment)
    env.make_interactive(port=port, realtime=True)
    return env
