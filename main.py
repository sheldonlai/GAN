import sys
from util import utils
from train import train

if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == "sound":
            utils.analyze_sound(sys.argv[2])
        elif sys.argv[1] == "train":
            train()
        elif sys.argv[1] == "convert":
            utils.fetch_sample_files()
        elif sys.argv[1] == "generate":
            train(train=False)
