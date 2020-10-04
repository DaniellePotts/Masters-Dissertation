import sys

import os
import minerl

if __name__ == "__main__":

    if len(sys.argv) != 3:
        print('Required arguments not provided.')
    else:
        directory = sys.argv[1]
        minerl_data = sys.argv[2]

        os.environ['MINERL_DATA_ROOT'] = directory

        os.system('python -m minerl.data.download "{0}"'.format(minerl_data))