from ast import main
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
from datetime import datetime
from scipy.stats import norm

data = pd.read_csv('starter_data.csv')
data = data.set_index(data['Date'])
data = data.iloc[:,1:]

portfolios = data.iloc[:, -2:]
factors = data.iloc[:, :-2]

def cumulative_monthly_returns(dataframe):
    return (1 + dataframe).cumprod() - 1

def convert_to_quarterly(dataframe):
    return dataframe.resample('Q').agg(lambda x: (x+1).cumprod() - 1)

def std_full_5y_3y_rolling(monthly_returns):
    '''
    Calculates the full standard deviation as well as 3 year and 5 year rolling std.
    '''
    full_std = np.std(monthly_returns)
    rolling_5y = monthly_returns.rolling(60).std().dropna()
    rolling_3y = monthly_returns.rolling(36).std().dropna()

    return full_std, rolling_5y, rolling_3y

def annualised_volatility(std_dev):
    '''
    Converts monthly std_dev into annualised volatility.
    '''
    return std_dev*np.sqrt(12)

def correlation(portfolios, factors):
    '''
    Calculates the correlation between a portfolio and set of factors, a 3year and a 5 year rolling correlation
    '''
    df = pd.concat([portfolios, factors], axis=1)
    correlation = df.corr().iloc[:,:portfolios.shape[1]]
    rolling_correlation_3y = df.rolling(36).corr().iloc[:,portfolios.shape[1]].dropna()
    rolling_correlation_5y = df.rolling(60).corr().iloc[:,portfolios.shape[1]].dropna()
    return correlation, rolling_correlation_3y, rolling_correlation_5y

def tracking_error(main_portfolio, benchmark):
    '''
    Calculates the:
    - Total tracking error
    - Rolling 2 year error
    - Rolling 5 year error
    Tracking error is the standard deviation of the difference between the returns of an investment and its benchmark. 
    Given a sequence of returns for an investment or portfolio and its benchmark, tracking error is calculated as follows: 
    Tracking Error = Standard Deviation of (P - B)
    '''
    total_te = (main_portfolio - benchmark).std()
    te_rolling_2y = (main_portfolio - benchmark).rolling(24).std().dropna()
    te_rolling_5y = (main_portfolio - benchmark).rolling(60).std().dropna()
    
    return total_te, te_rolling_2y, te_rolling_5y

def information_ratio(main_portfolio, benchmark):
    '''
    The information ratio (IR) is a measurement of portfolio returns beyond the returns of a benchmark, usually an index, 
    compared to the volatility of those returns. 

    IR = (Portfolio Return - Benchmark Return) / Tracking Return

    Function calculates total IR, 2y rolling and 5y rolling

    *** in this example, portfolio return will be "Real Estate Portfolio" and benchmark "Equity Index" in the factor df
    '''

    te, rolling_2y_te, rolling_5y_te = tracking_error(main_portfolio, benchmark)

    total_ir = (main_portfolio - benchmark) / te
    rolling_2y_ir = ((main_portfolio - benchmark) / (main_portfolio - benchmark).std()).rolling(24).mean().dropna()
    rolling_5y_ir = ((main_portfolio - benchmark) / (main_portfolio - benchmark).std()).rolling(60).mean().dropna()

    return total_ir, rolling_2y_ir, rolling_5y_ir

def VaR(dataframe, alpha=0.05):
    '''
    Calculates 5% VaR for a pandas dataframe
    '''
    mean = dataframe.mean()
    std_dev = dataframe.std()
    var_95 = norm.ppf(1-alpha, mean, std_dev)

    df = pd.DataFrame(index=dataframe.columns)
    df['5% VaR'] = var_95

    return df

def beta(dataframe):

    cov = dataframe.cov().iloc[:,-2:]
    var = dataframe.var()

    beta_df = pd.DataFrame(columns=dataframe.columns[-2:],
                           index = ['Beta ' + i for i in cov.index])

    for i in range(0, cov.shape[1]):
        beta_df.iloc[:,i] = list(cov.iloc[:,i] / var)


    return beta_df


def plot(returns, name=''):
    plt.figure(figsize=(20,20))
    plt.plot(returns, label=returns.columns)
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel(name)
    plt.show()


if __name__ == "__main__":
    
    monthly_returns = cumulative_monthly_returns(data)
    monthly_returns.index = pd.to_datetime(monthly_returns.index, infer_datetime_format=True)
    quarterly_returns = convert_to_quarterly(monthly_returns)
    full_std, rolling_5y, rolling_3y = std_full_5y_3y_rolling(monthly_returns)
    correlation_matrix, rolling_3y_correlation, rolling_5y_correlation = correlation(portfolios, factors)
    total_te, te_rolling_2y, te_rolling_5y = tracking_error(portfolios.iloc[:,0], portfolios.iloc[:,1])
    total_ir, ir_rolling_2y, ir_rolling_5y = information_ratio(portfolios.iloc[:,0], factors.iloc[:,0])
    var_95 = VaR(data)
    beta_df = beta(data)
    print(correlation_matrix)
    print(beta_df)

    


