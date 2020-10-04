import minerl
import gym

import numpy as np

import collections

import pickle

import sys

import functions.MineRLDataPreProcessing as mrrldpp

from functions.Utils import save_data, load_data

if __name__ == "__main__":
    data = minerl.data.make(sys.argv[1])

    sequences = mrrldpp.parse_demo(data)

    unique_places = mrrldpp.get_unique_place_actions(sequences)
    unique_actions = mrrldpp.get_unique_place_actions(sequences[0]['actions'])

    print('Parsed demonstration data. Parsing sequences...')

    parsed_sequences = mrrldpp.parse_sequences(sequences)

    print('Parsed sequences. Saving sequences into file.')
    save_data("./resources/parsed_data.sav", parsed_sequences)
    save_data("./resources/unique_places.sav", unique_places)
    save_data("./resources/unique_actions.sav", unique_actions)
