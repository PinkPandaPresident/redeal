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
        self.last_bid = None
        self.last_bid_position = None
        self.number_of_passes = 0
        self.bidding_over = False
        self.reward = 0
        self.end_contract = None
        self.current_player = None

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

    def create_bidding_history(self):
        return numpy.zeros(318+1)

    def print_bidding_history(self, opt_bidding_history=None):

        if opt_bidding_history is None:
            opt_bidding_history = self.bidding_history

        print(opt_bidding_history[:3])
        for i in range(35):
            if opt_bidding_history[3+(i*9)] == 1:

                print(f"{1+(i//5)}{'C' if i%5==0 else 'D' if i%5==1 else 'H' if i%5==2 else 'S' if i%5==3 else 'NT'}: {opt_bidding_history[3+(i*9):12+(i*9)]}")

        #print(self.bidding_history)

    @classmethod
    def acc_print_bidding_history(cls, bidding_rec):
        to_print = []
        for i in bidding_rec[:3]:
            if i == 1:
                to_print.append("Pass")

        for i in range(35):
            if bidding_rec[3+(i*9)] == 1:
                to_print.append(f"{1+(i//5)}{'C' if i%5==0 else 'D' if i%5==1 else 'H' if i%5==2 else 'S' if i%5==3 else 'NT'}")
                if bidding_rec[3 + (i * 9) + 1] == 1:
                    to_print.append("Pass")
                if bidding_rec[3 + (i * 9) + 2] == 1:
                    to_print.append("Pass")

                if bidding_rec[3 + (i * 9) + 3] == 1:
                    to_print.append("Double")

                if bidding_rec[3 + (i * 9) + 4] == 1:
                    to_print.append("Pass")
                if bidding_rec[3 + (i * 9) + 5] == 1:
                    to_print.append("Pass")

                if bidding_rec[3 + (i * 9) + 6] == 1:
                    to_print.append("Redouble")

                if bidding_rec[3 + (i * 9) + 7] == 1:
                    to_print.append("Pass")
                if bidding_rec[3 + (i * 9) + 8] == 1:
                    to_print.append("Pass")

        return to_print



    @classmethod
    def legal_bids(cls, observation):
        """
        Returns set of legal bids, with same convention described in self.step, i.e. 35 = pass, 0 = 1c, etc.
        """
        temp_bidding_history = observation[-319:]
        if len(numpy.nonzero(temp_bidding_history)[0]) == 0:
            return numpy.arange(36)
        last_bid = numpy.nonzero(temp_bidding_history)[0][-1]


        if last_bid < 3:
            # Can do anything except double or redouble since nothing has been bid so far
            return numpy.arange(36)

        last_proper_bid = (last_bid-3) // 9
        pot_range = numpy.arange(last_proper_bid+1, 36, 1)
        if temp_bidding_history[3+(last_proper_bid*9)+6] == 1:
            return pot_range
        if temp_bidding_history[3+(last_proper_bid*9)+3] == 1:
            rel_seq = temp_bidding_history[3+(last_proper_bid*9)+4:3+(last_proper_bid*9)+6]
            if numpy.array_equal(rel_seq, numpy.array((0, 0))) or numpy.array_equal(rel_seq, numpy.array((1, 1))):
                pot_range = numpy.append(pot_range, 37)
            return pot_range
        rel_seq = temp_bidding_history[3+(last_proper_bid*9)+1:3+(last_proper_bid*9)+3]
        if numpy.array_equal(rel_seq, numpy.array((0, 0))) or numpy.array_equal(rel_seq, numpy.array((1, 1))):
            pot_range = numpy.append(pot_range, 36)


        return set(pot_range)


    def calculate_reward(self):
        if self.end_contract is None:
            return 0

        return -self.deal.dd_score(self.end_contract)

    def reset(self):
        """
        Resets bridge game. Returns initial observation, i.e. 52 cards held, bidding history, vulnerabilities.
        """

        self.giving_info = None
        self.bidding_history = None
        self.vuls = None
        self.cards = None
        self.deal = None
        self.our_cards = None
        self.last_proper_bid = None
        self.last_bid = None
        self.last_bid_position = None
        self.number_of_passes = 0
        self.bidding_over = False
        self.reward = 0
        self.end_contract = None
        self.current_player = None


        dealer = Deal.prepare()
        self.deal = dealer()

        # North = 0, East = 1, South = 2, West = 3

        self.current_player = 0
        self.cards = self.create_cards(self.deal)
        self.our_cards = self.cards[:52]
        self.vuls = numpy.random.randint(1, size=(2)).astype(float)

        self.bidding_history = self.create_bidding_history()

        self.giving_info = numpy.concatenate((self.our_cards, self.vuls, self.bidding_history))
        return self.giving_info.astype(numpy.float32)

    def step(self, action):
        """
        Returns (new_state, reward, if_game_end, debug_info), when given an action.
        The action is a number from 0 to 37 (including 37), where:
        0,1,2,3,4 = 1c, 1d, 1h, 1s, 1nt
        ...
        35 = pass, 36 = double, 37 = redouble

        Function assumes input is legal.

        Bidding History works as follows:

        0 1 2 = pass pass pass
        3...11 = 1c bid (potential sequence could go 1c pass pass double pass pass redouble pass pass)
        12...20 = 1d bid
        ...
        309...317 = 7nt bid
        318 = final pass

        Described in more detail in https://arxiv.org/pdf/1903.00900.pdf. Added final pass, because unclear how they
        dealt with it.
        """
        # assumes action is legal

        num_to_direc = {0: "N", 1: "E", 2: "S", 3: "W"}


        if action < 35:
            self.number_of_passes = 0
            self.bidding_history[3 + (action*9)] = 1
            self.last_bid_position = 3 + (action*9)
        elif action == 35:
            self.number_of_passes += 1

            if self.last_bid is None:
                self.bidding_history[0] = 1
                self.last_bid_position = 0
            elif self.number_of_passes == 3:

                if self.last_bid_position == 1:
                    self.bidding_history[2] = 1
                    self.last_bid_position = 2

                else:

                    self.bidding_history[318] = 1
                    self.bidding_over = True
                    self.last_bid_position = 318
                    if self.last_proper_bid is None:
                        pass
                    else:
                        doubled, redoubled = False, False
                        if self.bidding_history[3 + (self.last_proper_bid*9) + 3] == 1:
                            doubled = True
                        if self.bidding_history[3 + (self.last_proper_bid*9) + 6] == 1:
                            redoubled = True

                        i = self.last_proper_bid

                        contract = f"{1+(i//5)}{'C' if i%5==0 else 'D' if i%5==1 else 'H' if i%5==2 else 'S' if i%5==3 else 'NT'}"
                        self.end_contract = f"{contract}{'X' if doubled else 'XX' if redoubled else ''}{num_to_direc[self.current_player]}"

            elif self.number_of_passes == 4:
                self.bidding_history[318] = 1
                self.bidding_over = True
                self.last_bid_position = 318


            else:
                self.bidding_history[self.last_bid_position+1] = 1
                self.last_bid_position += 1

        elif action == 36:
            self.number_of_passes = 0
            self.bidding_history[3 + (self.last_proper_bid*9) + 3] = 1
            self.last_bid_position = 3 + (self.last_proper_bid*9) + 3

        elif action == 37:
            self.number_of_passes = 0
            self.bidding_history[3 + (self.last_proper_bid*9) + 6] = 1
            self.last_bid_position = 3 + (self.last_proper_bid * 9) + 6

        # if self.last_bid is None:
        #     if action == 35:
        #         self.bidding_history[0] = 1
        #     else:
        #         self.bidding_history[3+(action*9)] = 1
        #
        # elif self.last_proper_bid is None:
        #     last_pass = numpy.nonzero(self.bidding_history)[0][-1]
        #     if action == 0:
        #         if last_pass == 2:
        #             self.bidding_history[last_pass + 1] = 1
        #
        if action < 35:
            self.last_proper_bid = action

        self.last_bid = action
        self.giving_info = numpy.concatenate((self.our_cards, self.vuls, self.bidding_history))

        if self.bidding_over:
            self.reward = self.calculate_reward()

        self.current_player = (self.current_player + 1 ) % 4

        debug_info = {
            "First_3": self.bidding_history[:3]


        }

        return self.giving_info.astype(numpy.float32), self.reward, self.bidding_over, debug_info

