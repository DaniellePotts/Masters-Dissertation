import random
import sys
import gym
import minerl

import numpy as np

from functions.ActionCombos import get_all_action_combos, int_action_to_dict, convert_match_actions, match_batch_actions
from functions.Utils import write_json_file
from functions.DQfDModel import get_model_prediction

#Gets agent action from the model - returns action as a dictionary and bool determining if the action was random
def get_agent_action(env, model, curr_obs, action_combos, n_action, action_keys, unique_angles, do_random, level_of_randomness):

    action_was_random = False
    
    #if do_random is true and the random value generated is less or equal to level_of_randomness
    #generate a random action - else get an action from the model based on the current observation
    if (do_random == True) and (random.random() <= level_of_randomness):
        action_command = env.action_space.sample()
        action_was_random = True
    else:
        action_command = get_model_prediction(model, n_action, curr_obs)
        
    action_to_take = None
   
   #convert action to dict
    if (isinstance(action_command, (int, np.integer))):
        action_to_take = int_action_to_dict(action_keys, action_combos[action_command])
    else:
        action_to_take = int_action_to_dict(action_keys, action_combos[convert_match_actions(action_command, action_combos, unique_angles)])

    #return action and if the action was random   
    return action_to_take, action_was_random
    
#Run the model in the environment
def run_rl_agent(env, model, action_combos, n_action, action_keys, unique_angles, do_random, level_of_randomness):

    curr_obs = env.reset()
    done = False

    #number of tick
    wait_for_max = 40

    actions = []
    while not done:
        action, action_was_random = get_agent_action(env, model, curr_obs, action_combos, n_action, action_keys, unique_angles,  do_random, level_of_randomness)
        wait_for_step = 1

        if do_random == True:
            actions.append({'action':action, 'was_random':action_was_random})
            write_json_file("actions.json", actions)
        while(wait_for_step < wait_for_max):
            #run action and get back observation
            curr_obs, reward, done, _ = env.step(action)
            wait_for_step = wait_for_step + 1

        if done: print(done)

#starts up environment and sets it to interactive
def start_environment(environment, port):
    print('Starting up {0} environment - this may take some time'.format(environment))
    
    env = gym.make(environment)
    env.make_interactive(port=port, realtime=True)

    return env
