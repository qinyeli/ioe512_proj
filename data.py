import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

import utils


def read_data():
    df = pd.read_csv('UM-covid-data.csv')

    def smooth(raw):
        box_pts = 3  # smooth over 3 weeks
        box = np.ones(box_pts)/box_pts
        return np.convolve(raw, box, mode='same')

    weekly_cases = smooth(df['cases'])
    weekly_tests = smooth(df['tests'])
    return weekly_cases[:-1], weekly_tests[1:]


def fit_model(cases, tests):
    # We want to use the last week's data to 
    cases = cases.reshape(-1, 1)
    reg = linear_model.LinearRegression()
    reg.fit(cases, tests)
    return reg.predict(cases)


def get_transition_probability(weekly_cases):
    # Infection rates are in the rage 0.0001 ~ 0.0004
    trans_prob = np.zeros([5,5])
    for i in range(len(weekly_cases) - 1):
        curr = utils.weekly_cases_to_index(weekly_cases[i])
        next = utils.weekly_cases_to_index(weekly_cases[i + 1])
        trans_prob[curr][next] = trans_prob[curr][next] + 1
    trans_prob = (trans_prob.T / np.sum(trans_prob, axis=1)).T
    return trans_prob


def get_mapping_from_infection_to_test(weekly_cases, weekly_tests):
    trans_prob = np.zeros([
        utils.get_num_cases_intervals(),
        utils.get_num_tests_intervals()])
    for i in range(len(weekly_cases)):
        cases = utils.weekly_cases_to_index(weekly_cases[i])
        tests = utils.weekely_tests_to_index(weekly_tests[i])
        trans_prob[cases][tests] = trans_prob[cases][tests] + 1
    trans_prob = (trans_prob.T / np.sum(trans_prob, axis=1)).T
    return trans_prob


def get_matrices(weekly_cases, weekly_tests, verbose=False):
    trans_prob = get_transition_probability(weekly_cases)
    mapping_from_infection_to_test = get_mapping_from_infection_to_test(weekly_cases, weekly_tests)
    
    if verbose:
        np.set_printoptions(precision=3)
        print('\nTransition probability is :')
        print(trans_prob)
        print('\nMapping from infection rate to testing demand is')
        print(mapping_from_infection_to_test)

    return trans_prob, mapping_from_infection_to_test


def plot_heatmap(data, row_labels, col_labels):
    im, cbar = utils.heatmap(
        data,
        row_labels,
        col_labels,
        cmap="YlGn",
        cbarlabel="Probability for demand")
    utils.annotate_heatmap(im, valfmt="{x:.1f}")
    plt.show()


def fit_linear_model(weekly_cases, weekly_tests):
    predicted_tests = fit_model(weekly_cases, weekly_tests)
    plt.plot(weekly_cases, label='new cases')
    plt.plot(weekly_tests, label='tests')
    plt.plot(predicted_tests, label='predicted tests')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    weekly_cases, weekly_tests = read_data()    
    trans_prob, mapping_from_infection_to_test = get_matrices(
            weekly_cases,
            weekly_tests,
            verbose=True)

    # Plot heatmap
    # infection_rates = [utils.index_to_weekly_cases(i) for i in range(5)]
    # testing_kit_demand = [utils.index_to_weekly_tests(i) for i in range(6)]
    # plot_heatmap(trans_prob, infection_rates, infection_rates)
    # plot_heatmap(mapping_from_infection_to_test, infection_rates, testing_kit_demand)

    fit_linear_model(weekly_cases, weekly_tests)

    # plt.plot(weekly_cases)
    # plt.xlabel('Weeks')
    # plt.ylabel('Positive cases')
    # plt.title('Weekly cases')
    # plt.show()

    # plt.plot(weekly_tests)
    # plt.xlabel('Weeks')
    # plt.ylabel('Test conducted')
    # plt.title('Weekly tested')
    # plt.show()