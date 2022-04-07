from redeal import *
import numpy

class Game:
    def __init__(self):
        self.giving_info = None
        self.bidding_history = None
        self.vuls = None
        self.cards = None
        self.deal = None
        self.our_cards = None
        self.last_proper_bid = None


    def create_cards(self, deal):
        """
        Returns 52-bit array representing cards. Order: Split into 4 players by NESW first; then into suits by
        spades hearts diamonds clubs, then by values AKQ...32.
        """
        converter_dict = {
            "A": 0,
            "K": 1,
            "Q": 2,
            "J": 3,
            "T": 4,
            "9": 5,
            "8": 6,
            "7": 7,
            "6": 8,
            "5": 9,
            "4": 10,
            "3": 11,
            "2": 12,

        }

        # Note: _pbn_str returns suits in form spades, hearts, diamonds, clubs; high -> low; . = switch of suits; NESW

        all_the_hands = numpy.zeros((4, 4, 13))

        # Populating all_the_hands with 1s and 0s
        for i in range(4):
            dealt_hand = deal._pbn_str().split()[i + 1]
            # print(dealt_hand)
            if i == 0:
                dealt_hand = dealt_hand[3:]
            if i == 3:
                dealt_hand = dealt_hand[:-2]
            dealt_hand = dealt_hand.split('.')
            # print(dealt_hand)

            for n, suit in enumerate(dealt_hand):
                for card in suit:
                    all_the_hands[i][n][converter_dict[card]] = 1

        # print(deal._pbn_str())
        # print(deal._long_str())
        # for hand in all_the_hands:
        #     for suit in hand:
        #         print(suit)
        #     print()
        #
        # print(all_the_hands.flatten())

        return all_the_hands.flatten()

    def create_bidding_history(self, deal):
        return numpy.zeros(318)


    def reset(self):
        """
        Resets bridge game. Returns initial observation, i.e. 52 cards held, bidding history, vulnerabilities.
        """
        dealer = Deal.prepare()
        self.deal = dealer()

        self.cards = self.create_cards(self.deal)
        self.our_cards = self.cards[:52]
        self.vuls = numpy.random.randint(1, size=(2)).astype(float)


        self.giving_info = numpy.concatenate((self.our_cards, self.vuls, self.bidding_history))
        return self.giving_info




    def step(self, action):
        # return (new_state, reward, if_game_end, debug_info)
        # action is a number from 0 to 37 (including 37)
        # assumes action is legal






        return 0, 0, 0, 0
