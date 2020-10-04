import functions.MineRLDataPreProcessing as mrrldpp

from functions.Utils import save_data, load_data
from functions.Buffer import populate_buffer

from anyrl.rollouts import replay

import gym
import collections
import sys

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print('Required arguments not provided.')
    else:
        dataset_file_path = sys.argv[1]
        combos_file_path = sys.argv[2]

        print('Loading in data from {0}, {1}...'.format(dataset_file_path, combos_file_path))

        parsed_data = load_data(dataset_file_path)
        combos = load_data(combos_file_path)

        print('Loaded in data. Parsing data for buffer.')

        replay_buffer = replay.PrioritizedReplayBuffer(500000, alpha=0.4, beta=0.6, epsilon=0.001)
        buffer = populate_buffer(parsed_data, replay_buffer, combos)
        print('Buffer populated.')

        print("Saving buffer...")
        save_data("./resources/buffer.sav", buffer)