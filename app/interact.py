import sys
import configparser

if __name__ == '__main__':

    try:
        config = configparser.ConfigParser()
        config.read('./app/settings.ini')
        env_to_run = 'Treechop'
        if((len(sys.argv)) > 1):
            valid_envs = config['environments']['valid_environments'].split(',')
            if((len([env for env in valid_envs if env.upper() == sys.argv[1].upper()]) > 0) == False):
                raise Exception('No valid argument was chosen. Chosen argument: {0}. Valid arguments are {1}'.format(sys.argv[1], valid_envs))
        else:
            env_to_run

        from functions.Utils import load_data, save_data
        from functions.DQfDModel import build_model, load_model
        from functions.EnvironmentManager import run_rl_agent, start_environment

        import numpy as np
        import random

        model_weights_path = "./models/MineRL{0}-v0_model.h5".format(env_to_run)
        action_combos_path = "./resources-actions/action_combos_{0}.sav".format(env_to_run.lower())
        unique_angles_path = "./resources-actions/unique_angles_{0}.sav".format(env_to_run.lower())

        action_keys = ['attack','back','camera','forward','jump','left','right','sneak','sprint']
        
        config['environments']['action_keys_{0}'.format(env_to_run.lower())].split(',')
        
        print('loading in data...')

        action_combos = load_data(action_combos_path)
        n_action = len(action_combos)
        model = load_model(n_action, model_weights_path)
        unique_angles = load_data(unique_angles_path)

        print('loaded data.')

        env = start_environment("MineRL{0}-v0".format(env_to_run), config['environments']['port'])

        print('started up environment')
        print('running interaction')
        run_rl_agent(env, model, action_combos, n_action, action_keys, unique_angles)
    except:
        print("Unexpected error:", sys.exc_info()[1])
    #check a valid env was picked


   

    

