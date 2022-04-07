from Reinforcement_Learning.main_model import DQN
from redeal import *
from Reinforcement_Learning import main_model, creating_data, utils
from Reinforcement_Learning.main_model import DQN
def making_best_contract_table():
    all_hands = ('N', 'S', 'E', 'W')
    all_suits = ('S', 'H', 'D', 'C', 'N')
    all_doubles = ('', 'X', 'XX')
    all_numbers = (i for i in range(1, 8))

    all_contracts = []
    for number in all_numbers:
        for suit in all_suits:
            for hand in all_hands:
                for status in all_doubles:
                    all_contracts.append(str(str(number) + suit + status + hand))
    #print(all_contracts)
    return all_contracts


def all_contracts_results(deal, double=True):
    if double:
        all_contracts = ['1SN', '1SXN', '1SXXN', '1SS', '1SXS', '1SXXS', '1SE', '1SXE', '1SXXE', '1SW', '1SXW', '1SXXW',
                     '1HN', '1HXN', '1HXXN', '1HS', '1HXS', '1HXXS', '1HE', '1HXE', '1HXXE', '1HW', '1HXW', '1HXXW',
                     '1DN', '1DXN', '1DXXN', '1DS', '1DXS', '1DXXS', '1DE', '1DXE', '1DXXE', '1DW', '1DXW', '1DXXW',
                     '1CN', '1CXN', '1CXXN', '1CS', '1CXS', '1CXXS', '1CE', '1CXE', '1CXXE', '1CW', '1CXW', '1CXXW',
                     '1NN', '1NXN', '1NXXN', '1NS', '1NXS', '1NXXS', '1NE', '1NXE', '1NXXE', '1NW', '1NXW', '1NXXW',
                     '2SN', '2SXN', '2SXXN', '2SS', '2SXS', '2SXXS', '2SE', '2SXE', '2SXXE', '2SW', '2SXW', '2SXXW',
                     '2HN', '2HXN', '2HXXN', '2HS', '2HXS', '2HXXS', '2HE', '2HXE', '2HXXE', '2HW', '2HXW', '2HXXW',
                     '2DN', '2DXN', '2DXXN', '2DS', '2DXS', '2DXXS', '2DE', '2DXE', '2DXXE', '2DW', '2DXW', '2DXXW',
                     '2CN', '2CXN', '2CXXN', '2CS', '2CXS', '2CXXS', '2CE', '2CXE', '2CXXE', '2CW', '2CXW', '2CXXW',
                     '2NN', '2NXN', '2NXXN', '2NS', '2NXS', '2NXXS', '2NE', '2NXE', '2NXXE', '2NW', '2NXW', '2NXXW',
                     '3SN', '3SXN', '3SXXN', '3SS', '3SXS', '3SXXS', '3SE', '3SXE', '3SXXE', '3SW', '3SXW', '3SXXW',
                     '3HN', '3HXN', '3HXXN', '3HS', '3HXS', '3HXXS', '3HE', '3HXE', '3HXXE', '3HW', '3HXW', '3HXXW',
                     '3DN', '3DXN', '3DXXN', '3DS', '3DXS', '3DXXS', '3DE', '3DXE', '3DXXE', '3DW', '3DXW', '3DXXW',
                     '3CN', '3CXN', '3CXXN', '3CS', '3CXS', '3CXXS', '3CE', '3CXE', '3CXXE', '3CW', '3CXW', '3CXXW',
                     '3NN', '3NXN', '3NXXN', '3NS', '3NXS', '3NXXS', '3NE', '3NXE', '3NXXE', '3NW', '3NXW', '3NXXW',
                     '4SN', '4SXN', '4SXXN', '4SS', '4SXS', '4SXXS', '4SE', '4SXE', '4SXXE', '4SW', '4SXW', '4SXXW',
                     '4HN', '4HXN', '4HXXN', '4HS', '4HXS', '4HXXS', '4HE', '4HXE', '4HXXE', '4HW', '4HXW', '4HXXW',
                     '4DN', '4DXN', '4DXXN', '4DS', '4DXS', '4DXXS', '4DE', '4DXE', '4DXXE', '4DW', '4DXW', '4DXXW',
                     '4CN', '4CXN', '4CXXN', '4CS', '4CXS', '4CXXS', '4CE', '4CXE', '4CXXE', '4CW', '4CXW', '4CXXW',
                     '4NN', '4NXN', '4NXXN', '4NS', '4NXS', '4NXXS', '4NE', '4NXE', '4NXXE', '4NW', '4NXW', '4NXXW',
                     '5SN', '5SXN', '5SXXN', '5SS', '5SXS', '5SXXS', '5SE', '5SXE', '5SXXE', '5SW', '5SXW', '5SXXW',
                     '5HN', '5HXN', '5HXXN', '5HS', '5HXS', '5HXXS', '5HE', '5HXE', '5HXXE', '5HW', '5HXW', '5HXXW',
                     '5DN', '5DXN', '5DXXN', '5DS', '5DXS', '5DXXS', '5DE', '5DXE', '5DXXE', '5DW', '5DXW', '5DXXW',
                     '5CN', '5CXN', '5CXXN', '5CS', '5CXS', '5CXXS', '5CE', '5CXE', '5CXXE', '5CW', '5CXW', '5CXXW',
                     '5NN', '5NXN', '5NXXN', '5NS', '5NXS', '5NXXS', '5NE', '5NXE', '5NXXE', '5NW', '5NXW', '5NXXW',
                     '6SN', '6SXN', '6SXXN', '6SS', '6SXS', '6SXXS', '6SE', '6SXE', '6SXXE', '6SW', '6SXW', '6SXXW',
                     '6HN', '6HXN', '6HXXN', '6HS', '6HXS', '6HXXS', '6HE', '6HXE', '6HXXE', '6HW', '6HXW', '6HXXW',
                     '6DN', '6DXN', '6DXXN', '6DS', '6DXS', '6DXXS', '6DE', '6DXE', '6DXXE', '6DW', '6DXW', '6DXXW',
                     '6CN', '6CXN', '6CXXN', '6CS', '6CXS', '6CXXS', '6CE', '6CXE', '6CXXE', '6CW', '6CXW', '6CXXW',
                     '6NN', '6NXN', '6NXXN', '6NS', '6NXS', '6NXXS', '6NE', '6NXE', '6NXXE', '6NW', '6NXW', '6NXXW',
                     '7SN', '7SXN', '7SXXN', '7SS', '7SXS', '7SXXS', '7SE', '7SXE', '7SXXE', '7SW', '7SXW', '7SXXW',
                     '7HN', '7HXN', '7HXXN', '7HS', '7HXS', '7HXXS', '7HE', '7HXE', '7HXXE', '7HW', '7HXW', '7HXXW',
                     '7DN', '7DXN', '7DXXN', '7DS', '7DXS', '7DXXS', '7DE', '7DXE', '7DXXE', '7DW', '7DXW', '7DXXW',
                     '7CN', '7CXN', '7CXXN', '7CS', '7CXS', '7CXXS', '7CE', '7CXE', '7CXXE', '7CW', '7CXW', '7CXXW',
                     '7NN', '7NXN', '7NXXN', '7NS', '7NXS', '7NXXS', '7NE', '7NXE', '7NXXE', '7NW', '7NXW', '7NXXW']
    else:
        all_contracts = ['1SN', '1SS', '1SE', '1SW', '1HN', '1HS', '1HE', '1HW', '1DN', '1DS', '1DE', '1DW', '1CN', '1CS', '1CE', '1CW',
         '1NN', '1NS', '1NE', '1NW', '2SN', '2SS', '2SE', '2SW', '2HN', '2HS', '2HE', '2HW', '2DN', '2DS', '2DE', '2DW',
         '2CN', '2CS', '2CE', '2CW', '2NN', '2NS', '2NE', '2NW', '3SN', '3SS', '3SE', '3SW', '3HN', '3HS', '3HE', '3HW',
         '3DN', '3DS', '3DE', '3DW', '3CN', '3CS', '3CE', '3CW', '3NN', '3NS', '3NE', '3NW', '4SN', '4SS', '4SE', '4SW',
         '4HN', '4HS', '4HE', '4HW', '4DN', '4DS', '4DE', '4DW', '4CN', '4CS', '4CE', '4CW', '4NN', '4NS', '4NE', '4NW',
         '5SN', '5SS', '5SE', '5SW', '5HN', '5HS', '5HE', '5HW', '5DN', '5DS', '5DE', '5DW', '5CN', '5CS', '5CE', '5CW',
         '5NN', '5NS', '5NE', '5NW', '6SN', '6SS', '6SE', '6SW', '6HN', '6HS', '6HE', '6HW', '6DN', '6DS', '6DE', '6DW',
         '6CN', '6CS', '6CE', '6CW', '6NN', '6NS', '6NE', '6NW', '7SN', '7SS', '7SE', '7SW', '7HN', '7HS', '7HE', '7HW',
         '7DN', '7DS', '7DE', '7DW', '7CN', '7CS', '7CE', '7CW', '7NN', '7NS', '7NE', '7NW']

    # all_contracts = deal.all_possible_contracts(True)
    # print(len(all_contracts))
    all_results = []
    for contract in all_contracts:
        vul = False
        tricks_made = deal.dd_tricks(contract)
        got_score = Contract.from_str(contract[:-1], vul=vul).score(tricks_made)

        all_results.append([contract, tricks_made, got_score])
    return all_results


def test_one(double=True):
    """
    Checks All Contract Results Method in Class Deal, for doubles and then without
    """
    print(f"Test {'1a' if double else '1b'} Commencing...")
    dealer = Deal.prepare()
    deal1 = dealer()


    big_table = deal1.dd_all_results(double)


    # for i in big_table:
    #    print(i)
    # print(deal1._long_str())

    one = set(tuple(x) for x in big_table)

    big_table = all_contracts_results(deal1,double=double)
    # for i in big_table:
    #    print(i)
    # print(deal1._long_str())

    two = set(tuple(x) for x in big_table)


    if one==two:
        print(f"Test {'1a' if double else '1b'} Passed")
    else:
        print(f"Test {'1a' if double else '1b'} Failed")

def test_two(PATH):
    """
    Just verifying that a bid is picked. Expect to fail for now.
    """

    try:


        temp_agent = main_model.Agent(gamma=0.99, epsilon=0.0, batch_size=64, n_actions=4, eps_end=0.01,
                      input_dims=8, lr=0.003, predone_model_pth=PATH)
        temp_game = creating_data.Game()

        starting_state = temp_game.reset()
        chosen_action = temp_agent.choose_action(starting_state)

        temp_agent.pretty_print_action(chosen_action)
        print("Test 2 Passed")


    except:
        print("Test 2 Failed")


def test_three():

    temp_game = creating_data.Game()
    x = temp_game.reset()
    if len(x) == 318 + 52 + 2:
        print("Test 3 Passed")
    else:
        print("Test 3 Failed")









if __name__ == '__main__':
    # test_one(double=True)
    # test_one(double=False)
    # test_two("C:/Users/adidh/PycharmProjects/redeal/Reinforcement_Learning/current_model.pth")
    test_three()