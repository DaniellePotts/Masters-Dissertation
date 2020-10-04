import functions.MineRLDataPreProcessing as mrrldpp
from functions.Utils import save_data, load_data

import minerl
import sys

if __name__ == "__main__":
    if len(sys.argv) != 1:
        print('Required arguments not provided.')
    else:
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
