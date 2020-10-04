import sys

from functions.Utils import save_data, load_data
from functions.ActionCombos import get_unique_angles, get_all_action_combos

import pickle
if __name__ == "__main__":
    dataset_file_path = sys.argv[1]
    unique_actions_path = sys.argv[2]
    unique_placements_path = sys.argv[3]

    print('Loading in data from {0}, {1}, {2}...'.format(dataset_file_path, unique_actions_path, unique_placements_path))

    parsed_data = load_data(dataset_file_path)
    unique_actions = load_data(unique_actions_path)
    unique_placements = load_data(unique_placements_path)

    print('Loaded in data...')

    print('Computing unique angles')
    unique_angles = get_unique_angles(parsed_data)
    print('Getting all unique action combinations')
    combos = get_all_action_combos(unique_actions, unique_angles, unique_placements)
    print("Saving combos...")
    save_data("./resources/combos.sav", combos)