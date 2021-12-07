import pandas as pd
import numpy as np
import data
import utils
import sys

# Parameters to tune.
FIXED_COST = 10
UNIT_PRICE = 3
PENALTY = 50
STORAGE_LIMIT = 20

#----------------------------------------------------------------------#T = 4
T = 4

# Read data.
weekly_cases, weekly_tests = data.read_data()    
trans_prob, mapping_from_infection_to_test = data.get_matrices(weekly_cases, weekly_tests)

# The algorithm.
value = np.zeros([T + 1, STORAGE_LIMIT, utils.get_num_cases_intervals()], dtype=float)
policy = - np.ones([T + 1, STORAGE_LIMIT, utils.get_num_cases_intervals()], dtype=float)


for t in range(T - 1, -1, -1):
    for storage in range(STORAGE_LIMIT):
        for curr_cases in range(utils.get_num_cases_intervals()):
            # Find the best action based on current situation.
            best_val = sys.maxsize
            best_action = -1
            for a in range(STORAGE_LIMIT - storage):
                curr_val = a * UNIT_PRICE
                if a != 0:
                    curr_val += FIXED_COST

                # Calculate value weighed by probability of different testing demands.
                for test_demand in range(utils.get_num_tests_intervals()):
                    test_prob = mapping_from_infection_to_test[curr_cases][test_demand]

                    # If the supply does not meet the demand, add penalty.
                    remaining_kit = a + storage - test_demand
                    if remaining_kit < 0:
                        curr_val += PENALTY * abs(remaining_kit) * test_prob
                        remaining_kit = 0

                    # Add the value for the next stage
                    for next_cases in range(utils.get_num_cases_intervals()):
                        case_prob = trans_prob[curr_cases][next_cases]
                        curr_val += value[t + 1][remaining_kit][next_cases] * test_prob * case_prob

                if best_val > curr_val:
                    best_val = curr_val
                    best_action = a

            # Record the best value and the best action.
            value[t][storage][curr_cases] = best_val
            policy[t][storage][curr_cases] = best_action

for t in range(T):
    print(f'---------------------------- t = {t} ----------------------------')
    print('Values:')
    print(value[t])
    print('Policies:')
    print(policy[t])