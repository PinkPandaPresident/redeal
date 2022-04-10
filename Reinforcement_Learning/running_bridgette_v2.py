from Reinforcement_Learning.training_bridgette_v2 import fit
import sys
import logging

logging.basicConfig(level=logging.DEBUG)


def main(mode="play"):
    if mode == "fit":
        res = fit()
        sys.stdout.buffer.write(res)
        sys.stdout.flush()
    elif mode == "play":
        pass

if __name__ == "__main__":
    main("fit")


