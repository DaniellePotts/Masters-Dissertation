if __name__ == '__main__':

    import asyncio

    from functions.Utils import load_data, save_data
    from functions.DQfDModel import build_model, load_model
    from functions.interact_with_env import run_agent_interactive, start_environment_interactive

    import numpy as np
    import random

    model_weights_path = "./models/expert_model_1603572051_749999.h5"
    action_combos_treechop_path = "./resources/action_combos_treechop.sav"
    unique_angles_treechop_path = "./resources/unique_angles_treechop.sav"

    action_keys = ['attack',
    'back',
    'camera',
    'forward',
    'jump',
    'left',
    'right',
    'sneak',
    'sprint']

    n_action_treechop = 35840

    print('loading in data...')
    model = load_model(n_action_treechop, model_weights_path)
    action_combos = load_data(action_combos_treechop_path)
    unique_angles_treechop = load_data(unique_angles_treechop_path)
    print('loaded data.')
    # env = gym.make("MineRLTreechop-v0")
    # env.make_interactive(port=6666, realtime=True)
    env = start_environment_interactive("MineRLTreechop-v0", 4000)
    print('started up environment')
    print('running interaction')
    run_agent_interactive(env, model, action_combos, n_action_treechop, action_keys, unique_angles_treechop)