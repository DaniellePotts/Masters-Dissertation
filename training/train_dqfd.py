import sys

from MineRLDataPreProcessing import MineRLDataPreProcessing
from ActionCombos import ActionCombos

def main():
	env = sys.argv[1]

	mrldpp = MineRLDataPreProcessing('{0}-v0'.format(env))
	mrldpp.parse_demonstration_data('./datasets', 'processed_{}.pkl'.format(env))

if __name__ == "__main__":
    main()