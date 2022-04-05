from redeal import *


class Game:
    def __init__(self):
        dealer = Deal.prepare()
        deal1 = dealer()


    def reset(self):
        pass

    def step(self, action):
        # return (new_state, reward, if_game_end, debug_info)
        return 0, 0, 0, 0
