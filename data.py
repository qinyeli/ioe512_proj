import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model


def read_data():
    df = pd.read_csv('US.timeseries.csv')
    first_Sunday = 54  # The first Sunday that we have test data
    last_Sunday = 530  # We only consider up to 2021-07-04, bc the data is missing on the next day
    
    def get_weekly(cumulative):
        diff = [cumulative[n]-cumulative[n-1] for n in range(first_Sunday, last_Sunday)]
        weekly = np.array([np.mean(diff[7 * i: 7 * i + 6]) for i in range(len(diff) // 7)])
        return weekly
    
    def smooth(raw):
        box_pts = 3  # smooth over 3 weeks
        box = np.ones(box_pts)/box_pts
        return np.convolve(raw, box, mode='same')

    return smooth(get_weekly(df['actuals.cases'])), \
           smooth(get_weekly(df['actuals.positiveTests'] + df['actuals.negativeTests']))


def fit_model(cases, tests):
    # We want to use the last week's data to 
    cases = cases.reshape(-1, 1)
    reg = linear_model.LinearRegression()
    reg.fit(cases, tests)
    return reg.predict(cases)


if __name__ == '__main__':
    weekly_cases, weekly_tests = read_data()
    predicted_tests = fit_model(weekly_cases, weekly_tests)
    weekly_cases = weekly_cases[:-2]
    weekly_tests = weekly_tests[1:]

    plt.plot(weekly_cases, label='new cases')
    plt.plot(weekly_tests, label='tests')
    plt.plot(predicted_tests, label='predicted tests')
    plt.legend()
    plt.show()