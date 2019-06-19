import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["IMP_NUM_THREADS"] = "1"

import datetime
import time
import math
import pandas as pd
import numpy as np
import pandas_datareader.data as web
from pandas import Series, DataFrame
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from threading import Thread
from scipy.stats import pearsonr as corr
from itertools import combinations

from sklearn import preprocessing
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor as RFR 
from sklearn.svm import SVR

def add_comb(best, comb):
    if(not(comb in best)):
        best[comb] = 1
    else:
        best[comb] += 1
    return best

def fitting(model, X_train, y_train, fits, i):
    try:
        fits[i] = model.fit(X_train, y_train)
        return True
    except:
        return False


def get_ema(data, forecast_out):
    close_vals = data['Adj Close'].values
    ema = [close_vals[0]]
    for idx, v in enumerate(close_vals[1:]):
        ema.append(ema[idx] + 2/(forecast_out+1) * (close_vals[idx+1]-ema[idx]))
    return ema


def test_model():
    start = datetime.datetime(2010, 1, 1)
    end = datetime.datetime(2017, 1, 11)
    df = web.DataReader("AAPL", 'yahoo', start, end)
    #print(df.tail)
    close_px = df['Adj Close']
    mavg = close_px.rolling(window=100).mean()
    rets = close_px/close_px.shift(1)-1
    fig, ax = plt.subplots()
    #ax.plot(close_px.values)
    #ax.plot(mavg.values)
    ax.plot(rets)
    plt.show()


def test_comp():
    start = datetime.datetime(2010, 1, 1)
    end = datetime.datetime(2017, 1, 11)
    dfcomp = web.DataReader(['AAPL', 'GE', 'GOOG', 'IBM', 'MSFT'], 'yahoo',
                            start=start, end=end)['Adj Close']
    retscomp = dfcomp.pct_change()
    corr = retscomp.corr()
    #print(corr)
    #fig, ax = plt.subplots()
    #ax.scatter(retscomp.AAPL, retscomp.GE)
    #ax.set(xlabel='Returns AAPL', ylabel='RETURNS GE')
    #plt.show()
    #pd.plotting.scatter_matrix(retscomp, diagonal='kde')
    plt.imshow(corr, cmap='hot', interpolation='none')
    plt.colorbar()
    plt.xticks(range(len(corr)), corr.columns)
    plt.yticks(range(len(corr)), corr.columns)
    plt.show()


def risk():
    start = datetime.datetime(2010, 1, 1)
    end = datetime.datetime(2017, 1, 11)
    dfcomp = web.DataReader(['AAPL', 'GE', 'GOOG', 'IBM', 'MSFT'], 'yahoo',
                            start=start, end=end)['Adj Close']
    retscomp = dfcomp.pct_change()
    corr = retscomp.corr()

    fig, ax = plt.subplots()
    ax.scatter(retscomp.mean(), retscomp.std())
    ax.set(xlabel="Exptected Returns", ylabel="Risk")
    for label, x, y in zip(retscomp.columns, retscomp.mean(), retscomp.std()):
        ax.annotate(label, xy=(x, y), xytext=(20, -20), textcoords='offset points',
                    ha='right', va='bottom', bbox=dict(boxstyle='round,pad=0.5', fc='yellow',
                    alpha=0.5))#,
                    #arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad0'))

    plt.show()


def test_ml(stock='AMZN', forecast_out=5, month=None, day=None, plot=False):
    # Want to separate 1 percent of the data to forecast
    days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    # Today info
    if(month==None or day==None):
        today = datetime.datetime.now()
        month = today.month
        day = today.day

    # NEED TO SELECT SO THAT forecast_out TRADING DAYS HAVE HAPPENED (WEEKEND AND HOLIDAY)
    days = 0
    for i in range(2*forecast_out):

        trial_day = day-i if((day-i) > 0) else \
                    days[month-2]+day-i
        trial_month = month-1 if((day-i) <= 0) else month
        #print(trial_month, trial_day)

        if(datetime.datetime(2019, trial_month, trial_day).weekday()<5):
            days += 1
            #print("WAS WEEKDAY")
            if(days == forecast_out):
                pred_month = trial_month
                pred_day = trial_day
                break

    #pred_month = month-1 if((day-forecast_out) <= 0) else month
    #pred_day = day-forecast_out if((day-forecast_out) > 0) else \
    #       days[month-2]+day-forecast_out

    # For prediction
    start = datetime.datetime(2015, 1, 1)
    end = datetime.datetime(2019, pred_month, pred_day-2)
    df = web.DataReader(stock, 'yahoo', start, end)
    dfreg = df.loc[:,['Adj Close', 'Volume']]
    dfreg['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100.0
    dfreg['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0
    dfreg['EMA'] = get_ema(dfreg, forecast_out)
    dfreg['old Adj Close'] = dfreg['Adj Close']
    dfreg['Adj Close'] = dfreg['EMA']

    # For validation
    new_start = datetime.datetime(2019, pred_month, pred_day)
    new_end = datetime.datetime(2019, month, day)
    new_df = web.DataReader(stock, 'yahoo', new_start, new_end)
    new_dfreg = new_df.loc[:,['Adj Close', 'Volume']]
    new_dfreg['HL_PCT'] = (new_df['High'] - new_df['Low']) / new_df['Close'] * 100.0
    new_dfreg['PCT_change'] = (new_df['Close'] - new_df['Open']) / new_df['Open'] * 100.0

    # Drop missing value
    dfreg.fillna(value=-99999, inplace=True)
    new_dfreg.fillna(value=-99999, inplace=True)

    # Searating the label here, we want to predict the AdjClose
    forecast_col = 'Adj Close'
    dfreg['label'] = dfreg[forecast_col].shift(-forecast_out)
    X = np.array(dfreg.drop(['label'], 1))

    # Scale X for linear regression
    X = preprocessing.scale(X)

    # Finally want late X and early X for model
    X_lately = X[-forecast_out:]
    X = X[:-forecast_out]

    # Separate label and identify it as y
    y = np.array(dfreg['label'])
    y = y[:-forecast_out]

    # Training and testing sets
    X_train = X[:len(X)-forecast_out]
    X_test = X[len(X)-forecast_out:]

    y_train = y[:len(y)-forecast_out]
    y_test = y[len(y)-forecast_out:]

    # LinReg
    clfreg = LinearRegression(n_jobs=-1)

    # QuadReg2
    clfpoly2 = make_pipeline(PolynomialFeatures(2), Ridge())

    # QuadReg3
    clfpoly3 = make_pipeline(PolynomialFeatures(3), Ridge())

    # QuadReg4
    clfpoly4 = make_pipeline(PolynomialFeatures(4), Ridge())

    # QuadReg5
    clfpoly5 = make_pipeline(PolynomialFeatures(5), Ridge())
    
    # KNN Regression
    clfknn = KNeighborsRegressor(n_neighbors=2)

    # Bayesian Ridge
    clfbayr = BayesianRidge()

    # Neural Network
    clfmlp = MLPRegressor(hidden_layer_sizes=(100,100,100), learning_rate='adaptive',
              solver='adam', max_iter=5, verbose=False)

    # Random Forest Regressor
    clfrfr = RFR(n_estimators=15)

    # Support Vector Regressor
    clfsvr = SVR(gamma='auto')

    threads = []
    models = [clfreg, clfpoly2, clfpoly3, clfpoly4, clfpoly5, clfknn, clfbayr, clfrfr, clfsvr]
    fits = ['']*len(models)
    for i in range(len(models)):
        process = Thread(target=fitting, args=[models[i], X_train, y_train, fits, i], name=stock)
        process.start()
        threads.append(process)

    for process in threads:
        process.join()

    start = time.time()
    reg_forecast = fits[0].predict(X_lately)
    poly2_forecast = fits[1].predict(X_lately)
    poly3_forecast = fits[2].predict(X_lately)
    poly4_forecast = fits[3].predict(X_lately)
    poly5_forecast = fits[4].predict(X_lately)
    knn_forecast = fits[5].predict(X_lately)
    bayr_forecast = fits[6].predict(X_lately)
    rfr_forecast = fits[7].predict(X_lately)
    svr_forecast = fits[8].predict(X_lately)
    #mlp_forecast = clfmlp.predict(X_lately)
    mlp_forecast = clfbayr.predict(X_lately)

    # Set up dataframe
    dfreg['reg_forecast'] = np.nan
    dfreg['poly2_forecast'] = np.nan
    dfreg['poly3_forecast'] = np.nan
    dfreg['poly4_forecast'] = np.nan
    dfreg['poly5_forecast'] = np.nan
    dfreg['knn_forecast'] = np.nan
    dfreg['bayr_forecast'] = np.nan
    dfreg['mlp_forecast'] = np.nan
    dfreg['rfr_forecast'] = np.nan
    dfreg['svr_forecast'] = np.nan

    last_date = dfreg.iloc[-1].name
    last_unix = last_date
    next_unix = last_unix + datetime.timedelta(days=1)
    for i in zip(reg_forecast, poly2_forecast, poly3_forecast, poly4_forecast, poly5_forecast, 
                 knn_forecast, bayr_forecast, mlp_forecast, rfr_forecast, svr_forecast):
        next_date = next_unix
        next_unix += datetime.timedelta(days=1)
        dfreg.loc[next_date] = list([np.nan for _ in range(len(dfreg.columns)-10)]+list(i))

    #dfreg['mean_forecast'] = dfreg[['poly2_forecast', 'poly3_forecast']].mean(axis=1)
    #print(dfreg.tail(forecast_out+1))
    dfreg['mean_forecast'] = dfreg[['reg_forecast', 'poly2_forecast', 'poly3_forecast',
                                    'knn_forecast', 'bayr_forecast',# mlp_forecast,
                                    'rfr_forecast', 'svr_forecast']].mean(axis=1)

    as_list = dfreg.index.tolist()
    as_list[-forecast_out:] = new_df.index.tolist()
    dfreg.index = as_list
    dfreg[-forecast_out:].index = new_df.index.tolist()

    #
    # Trying to do all combinations
    #
    forecasts = ['reg_forecast', 'poly2_forecast', 'poly3_forecast', 'poly4_forecast',
                 'poly5_forecast', 'knn_forecast', 'bayr_forecast', 'rfr_forecast',
                 'svr_forecast']

    if(plot):
        dfreg['old Adj Close'].tail(20).plot(figsize=(20,12), lw=2)
        dfreg['Adj Close'].tail(20).plot(figsize=(20,12), lw=2)
        dfreg['reg_forecast'].tail(20).plot(lw=0.5)
        dfreg['poly2_forecast'].tail(20).plot(lw=0.5)
        dfreg['poly3_forecast'].tail(20).plot(lw=0.5)
        dfreg['poly4_forecast'].tail(20).plot(lw=0.5)
        dfreg['poly5_forecast'].tail(20).plot(lw=0.5)
        dfreg['knn_forecast'].tail(20).plot(lw=0.5)
        dfreg['bayr_forecast'].tail(20).plot(lw=0.5)
        dfreg['mean_forecast'].tail(20).plot(c='k')
        #dfreg['mlp_forecast'].tail(20).plot()
        dfreg['rfr_forecast'].tail(20).plot(lw=0.5)
        dfreg['svr_forecast'].tail(20).plot(lw=0.5)
    

    new_dfreg['Actual Adj Close'] = new_df['Adj Close']
    if(plot):
        new_dfreg['Actual Adj Close'].tail(20).plot(c='g', lw=2)
    fit = np.polyfit([i for i in range(forecast_out)],
                     dfreg['mean_forecast'].values[-forecast_out:], deg=1)

    #print("CALCULATING CORRELATION BETWEEN METHOD AND ACTUAL")
    actual = new_dfreg['Actual Adj Close'].tail(forecast_out)

    highest_corr = 0
    best_comb = ''
    num_combs = 0
    for j in range(2,9):
        for comb in combinations(forecasts, j):
            #print("COMBINATION: {}".format(comb))
            num_combs += 1
            comb_dat = dfreg[[*list(combinations(forecasts, j))[0]]].mean(axis=1).tail(
                                      forecast_out)
            #print(comb_dat)
            if(corr(comb_dat, actual)[0] > highest_corr):
                #print("New highest correlation: {}".format(corr(comb_dat, actual)[0]))
                #print(comb)
                highest_corr = corr(comb_dat, actual)[0]
                best_comb = comb

    #print("TOTAL NUMBER OF COMBINATIONS: {}".format(num_combs))
    #exit(1)

    reg_dat = dfreg['reg_forecast'].tail(forecast_out)
    reg_corr = corr(reg_dat, actual)
    #print("Linear Regression: {}".format(reg_corr))

    poly2_dat = dfreg['poly2_forecast'].tail(forecast_out)
    poly2_corr = corr(poly2_dat, actual)
    #print("Poly2: {}".format(poly2_corr))

    poly3_dat = dfreg['poly3_forecast'].tail(forecast_out)
    poly3_corr = corr(poly3_dat, actual)
    #print("Poly3: {}".format(poly3_corr))

    poly4_dat = dfreg['poly4_forecast'].tail(forecast_out)
    poly4_corr = corr(poly4_dat, actual)
    #print("Poly3: {}".format(poly3_corr))

    poly5_dat = dfreg['poly5_forecast'].tail(forecast_out)
    poly5_corr = corr(poly5_dat, actual)
    #print("Poly3: {}".format(poly3_corr))

    knn_dat = dfreg['knn_forecast'].tail(forecast_out)
    knn_corr = corr(knn_dat, actual)
    #print("K Nearest Neighbors: {}".format(knn_corr))
    
    bayr_dat = dfreg['bayr_forecast'].tail(forecast_out)
    bayr_corr = corr(bayr_dat, actual)
    #print("Bayesian: {}".format(bayr_corr))

    rfr_dat = dfreg['rfr_forecast'].tail(forecast_out)
    rfr_corr = corr(rfr_dat, actual)
    #print("Random Forest: {}".format(rfr_corr))

    svr_dat = dfreg['svr_forecast'].tail(forecast_out)
    svr_corr = corr(svr_dat, actual)
    #print("Support Vector: {}".format(rfr_corr))

    mean_dat = dfreg['mean_forecast'].tail(forecast_out)
    mean_corr = corr(mean_dat, actual)

    if(plot):
        plt.legend(loc='best')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title(stock)
        plt.savefig("./test_plots/{1}_{2}/{0}_{1}_{2}_{3}".format(stock, month, day,
                                                                  forecast_out))
        plt.close()
    return (reg_corr[0], poly2_corr[0], poly3_corr[0], poly4_corr[0], poly5_corr[0],\
           knn_corr[0], bayr_corr[0], rfr_corr[0], mean_corr[0], svr_corr[0]), best_comb
    

def buy_ml(stock, forecast_out=5, month=None, day=None, plot=False):
    # Want to separate 1 percent of the data to forecast
    days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    # Today info
    if((month==None) or (day==None)):
        today = datetime.datetime.now()
        month = today.month if((today.day+forecast_out)<=days[today.month-1]) else today.month+1
        day = today.day+forecast_out if((today.day-forecast_out)<=days[today.month-1]) else \
              today.day+forecast_out-days[today.month-1]
        day = today.day+forecast_out if(today.day+forecast_out == days[today.month-1]) else \
              (today.day+forecast_out)%days[today.month-1]

    # For prediction
    start = datetime.datetime(2015, 1, 1)
    end = datetime.datetime(2019, today.month, today.day) 
    df = web.DataReader(stock, 'yahoo', start, end)
    dfreg = df.loc[:,['Adj Close', 'Volume']]
    dfreg['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100.0
    dfreg['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0

    # Drop missing value
    dfreg.fillna(value=-99999, inplace=True)

    # Searating the label here, we want to predict ht eAdjClose
    forecast_col = 'Adj Close'
    dfreg['label'] = dfreg[forecast_col].shift(-forecast_out)
    X = np.array(dfreg.drop(['label'], 1))

    # Scale X for linear regression
    X = preprocessing.scale(X)

    # Finally want late X and early X for model
    X_lately = X[-forecast_out:]
    X = X[:-forecast_out]

    # Separate label and identify it as y
    y = np.array(dfreg['label'])
    y = y[:-forecast_out]

    # Training and testing sets
    X_train = X[:len(X)-forecast_out]
    X_test = X[len(X)-forecast_out:]

    y_train = y[:len(y)-forecast_out]
    y_test = y[len(y)-forecast_out:]

    # LinReg
    clfreg = LinearRegression(n_jobs=-1)

    # QuadReg2
    clfpoly2 = make_pipeline(PolynomialFeatures(2), Ridge())

    # QuadReg3
    clfpoly3 = make_pipeline(PolynomialFeatures(3), Ridge())
    
    # QuadReg4
    clfpoly4 = make_pipeline(PolynomialFeatures(4), Ridge())
    
    # QuadReg4
    clfpoly5 = make_pipeline(PolynomialFeatures(5), Ridge())
    
    # KNN Regression
    clfknn = KNeighborsRegressor(n_neighbors=2)

    # Bayesian Ridge
    clfbayr = BayesianRidge()

    # Neural Network
    #clfmlp = MLPRegressor(hidden_layer_sizes=(100,100,100), learning_rate='adaptive',
    #          solver='adam', max_iter=5, verbose=False)

    # Random Forest Regressor
    clfrfr = RFR(n_estimators=15, random_state=0)

    # Fitting in parallel
    new_threads = []
    models = [clfreg, clfpoly2, clfpoly3, clfpoly4, clfpoly5, clfknn, clfbayr, clfrfr]
    more_fits = ['']*len(models)
    #print("STOCK: {}".format(stock))
    for i in range(len(models)):
        #print("HERE")
        process = Thread(target=fitting, args=[models[i], X_train, y_train, more_fits, i],
                         name=stock)
        process.start()
        new_threads.append(process)
    for process in new_threads:
        process.join()

    # Evaluation
    #confidencereg = clfreg.score(X_train, y_train)
    #confidencepoly2 = clfpoly2.score(X_train, y_train)
    #confidencepoly3 = clfpoly3.score(X_train, y_train)
    #confidenceknn = clfknn.score(X_train, y_train)
    #confidencebayr = clfbayr.score(X_train, y_train)

    # Predictions
    #print(more_fits)
    reg_forecast = more_fits[0].predict(X_lately)
    poly2_forecast = more_fits[1].predict(X_lately)
    poly3_forecast = more_fits[2].predict(X_lately)
    poly4_forecast = more_fits[3].predict(X_lately)
    poly5_forecast = more_fits[4].predict(X_lately)
    knn_forecast = more_fits[5].predict(X_lately)
    bayr_forecast = more_fits[6].predict(X_lately)
    #mlp_forecast = clfmlp.predict(X_lately)
    rfr_forecast = more_fits[7].predict(X_lately)

    # Set up dataframe
    dfreg['reg_forecast'] = np.nan
    dfreg['poly2_forecast'] = np.nan
    dfreg['poly3_forecast'] = np.nan
    dfreg['poly4_forecast'] = np.nan
    dfreg['poly5_forecast'] = np.nan
    dfreg['knn_forecast'] = np.nan
    dfreg['bayr_forecast'] = np.nan
    dfreg['mlp_forecast'] = np.nan
    dfreg['rfr_forecast'] = np.nan

    last_date = dfreg.iloc[-1].name
    last_unix = last_date
    next_unix = last_unix + datetime.timedelta(days=1)
    for i in zip(reg_forecast, poly2_forecast, poly3_forecast, poly4_forecast, poly5_forecast, 
                 knn_forecast, bayr_forecast, rfr_forecast):
        next_date = next_unix
        next_unix += datetime.timedelta(days=1)
        dfreg.loc[next_date] = list([np.nan for _ in range(len(dfreg.columns)-8)]+list(i))

    dfreg['mean_forecast'] = dfreg[['reg_forecast', 'poly2_forecast', 'poly3_forecast',
                                 'knn_forecast', 'bayr_forecast',# 'mlp_forecast',
                                 'rfr_forecast']].mean(axis=1)
    if(plot): 
        dfreg['Adj Close'].tail(50).plot(lw=2, figsize=(20,12))
        dfreg['mean_forecast'].tail(50).plot(lw=2, c='k')
        dfreg['bayr_forecast'].tail(50).plot(lw=0.5)
        dfreg['knn_forecast'].tail(50).plot(lw=0.5)
        dfreg['reg_forecast'].tail(50).plot(lw=0.5)
        dfreg['poly2_forecast'].tail(50).plot(lw=0.5)
        dfreg['poly3_forecast'].tail(50).plot(lw=0.5)
        dfreg['poly4_forecast'].tail(50).plot(lw=0.5)
        dfreg['poly5_forecast'].tail(50).plot(lw=0.5)
        #dfreg['mlp_forecast'].tail(50).plot(lw=0.5)
        dfreg['rfr_forecast'].tail(50).plot(lw=0.5)
        plt.legend(loc='best')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title(stock)
        plt.savefig("./pred_plots/{}_{}/price/{}_{}_{}".format(today.day, today.month,
                                                         stock,today.day,today.month))
        #plt.show()
        plt.close()
        #dfreg['Volume'].tail(200).plot()
        #plt.title(stock)
        #plt.show()

    try:
        fit = np.polyfit([i for i in range(forecast_out)],
                         dfreg['mean_forecast'].values[-forecast_out:], deg=1)
    except:
        print("FORECASTING {} DAY OUT".format(forecast_out))
        fit = [dfreg['mean_forecast'].values[-1] - dfreg['Adj Close'].values[-1], 2]

    string = "SHOULD GO UP" if(fit[0] > 0) else "SHOULD GO DOWN"
    #print("{} {}".format(stock, string))
    if(fit[0] > 0):
        return fit[0]
    else:
        return fit[0]


def buy_ml_vol(stock, forecast_out=5, month=None, day=None, plot=False):
    # Want to separate 1 percent of the data to forecast
    days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    # Today info
    if((month==None) or (day==None)):
        today = datetime.datetime.now()
        month = today.month if((today.day+forecast_out)<=days[today.month-1]) else today.month+1
        day = today.day+forecast_out if((today.day-forecast_out)<=days[today.month-1]) else \
              today.day+forecast_out-days[today.month-1]
        day = today.day+forecast_out if(today.day+forecast_out == days[today.month-1]) else \
              (today.day+forecast_out)%days[today.month-1]

    # For prediction
    start = datetime.datetime(2015, 1, 1)
    end = datetime.datetime(2019, today.month, today.day) 
    df = web.DataReader(stock, 'yahoo', start, end)
    dfreg = df.loc[:,['Adj Close', 'Volume']]
    dfreg['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100.0
    dfreg['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0

    # Drop missing value
    dfreg.fillna(value=-99999, inplace=True)

    # Searating the label here, we want to predict ht eAdjClose
    forecast_col = 'Volume'
    dfreg['label'] = dfreg[forecast_col].shift(-forecast_out)
    X = np.array(dfreg.drop(['label'], 1))

    # Scale X for linear regression
    X = preprocessing.scale(X)

    # Finally want late X and early X for model
    X_lately = X[-forecast_out:]
    X = X[:-forecast_out]

    # Separate label and identify it as y
    y = np.array(dfreg['label'])
    y = y[:-forecast_out]

    # Training and testing sets
    X_train = X[:len(X)-forecast_out]
    X_test = X[len(X)-forecast_out:]

    y_train = y[:len(y)-forecast_out]
    y_test = y[len(y)-forecast_out:]

    # LinReg
    clfreg = LinearRegression(n_jobs=-1)

    # QuadReg2
    clfpoly2 = make_pipeline(PolynomialFeatures(2), Ridge())

    # QuadReg3
    clfpoly3 = make_pipeline(PolynomialFeatures(3), Ridge())
    
    # KNN Regression
    clfknn = KNeighborsRegressor(n_neighbors=2)

    # Bayesian Ridge
    clfbayr = BayesianRidge()

    # Neural Network
    #clfmlp = MLPRegressor(hidden_layer_sizes=(100,100,100), learning_rate='adaptive',
    #          solver='adam', max_iter=5, verbose=False)
    #clfmlp.fit(X_train, y_train)

    # Random Forest Regressor
    clfrfr = RFR(n_estimators=15, random_state=0)

    # Fitting
    threads = []
    models = [clfreg, clfpoly2, clfpoly3, clfknn, clfbayr, clfrfr]
    fits = ['']*len(models)
    for i in range(len(models)):
        process = Thread(target=fitting, args=[models[i], X_train, y_train, fits, i], name=stock)
        process.start()
        threads.append(process)

    for process in threads:
        process.join()

    # Evaluation
    #confidencereg = clfreg.score(X_train, y_train)
    #confidencepoly2 = clfpoly2.score(X_train, y_train)
    #confidencepoly3 = clfpoly3.score(X_train, y_train)
    #confidenceknn = clfknn.score(X_train, y_train)
    #confidencebayr = clfbayr.score(X_train, y_train)

    # Predictions
    reg_forecast = fits[0].predict(X_lately)
    poly2_forecast = fits[1].predict(X_lately)
    poly3_forecast = fits[2].predict(X_lately)
    knn_forecast = fits[3].predict(X_lately)
    bayr_forecast = fits[4].predict(X_lately)
    #mlp_forecast = clfmlp.predict(X_lately)
    rfr_forecast = fits[5].predict(X_lately)

    # Set up dataframe
    dfreg['reg_forecast'] = np.nan
    dfreg['poly2_forecast'] = np.nan
    dfreg['poly3_forecast'] = np.nan
    dfreg['knn_forecast'] = np.nan
    dfreg['bayr_forecast'] = np.nan
    dfreg['mlp_forecast'] = np.nan
    dfreg['rfr_forecast'] = np.nan

    last_date = dfreg.iloc[-1].name
    last_unix = last_date
    next_unix = last_unix + datetime.timedelta(days=1)
    for i in zip(reg_forecast, poly2_forecast, poly3_forecast, knn_forecast, bayr_forecast,
                 rfr_forecast):
        next_date = next_unix
        next_unix += datetime.timedelta(days=1)
        dfreg.loc[next_date] = list([np.nan for _ in range(len(dfreg.columns)-6)]+list(i))

    dfreg['mean_forecast'] = dfreg[['reg_forecast', 'poly2_forecast', 'poly3_forecast',
                                 'knn_forecast', 'bayr_forecast',# 'mlp_forecast',
                                 'rfr_forecast']].mean(axis=1)
    if(plot): 
        dfreg['Volume'].tail(50).plot(lw=2, figsize=(20,12))
        dfreg['mean_forecast'].tail(50).plot(lw=2, c='k')
        dfreg['bayr_forecast'].tail(50).plot(lw=0.5)
        dfreg['knn_forecast'].tail(50).plot(lw=0.5)
        dfreg['reg_forecast'].tail(50).plot(lw=0.5)
        dfreg['poly2_forecast'].tail(50).plot(lw=0.5)
        dfreg['poly3_forecast'].tail(50).plot(lw=0.5)
        #dfreg['mlp_forecast'].tail(50).plot(lw=0.5)
        dfreg['rfr_forecast'].tail(50).plot(lw=0.5)
        plt.legend(loc='best')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title(stock)
        plt.savefig("./pred_plots/{}_{}/volume/{}_{}_{}".format(today.day, today.month,
                                                         stock,today.day,today.month))
        #plt.show()
        plt.close()
        #dfreg['Volume'].tail(200).plot()
        #plt.title(stock)
        #plt.show()

    try:
        fit = np.polyfit([i for i in range(forecast_out)],
                         dfreg['mean_forecast'].values[-forecast_out:], deg=1)
    except:
        print("FORECASTING {} DAY OUT".format(forecast_out))
        fit = [dfreg['mean_forecast'].values[-1] - dfreg['Adj Close'].values[-1], 2]

    string = "VOLUME SHOULD GO UP" if(fit[0] > 0) else "VOlUME SHOULD GO DOWN"
    #print("{} {}".format(stock, string))
    return fit[0]

if __name__ == '__main__':
    today = datetime.datetime.now()
    try:
        os.mkdir("./pred_plots/{}_{}".format(today.day, today.month))
    except FileExistsError:
        print("DIRECTORY ALREADY CREATED")

    try:
        os.mkdir("./pred_plots/{}_{}/volume".format(today.day, today.month))
    except FileExistsError:
        print("DIRECTORY ALREADY CREATED")
    try:
        os.mkdir("./pred_plots/{}_{}/price".format(today.day, today.month))
    except FileExistsError:
        print("DIRECTORY ALREADY CREATED")

    try:
        os.mkdir("./test_plots/{}_{}/".format(today.month, today.day))
    except FileExistsError:
        print("DIRECTORY ALREADY CREATED")

    # Regular stocks
    stocks = ["AMZN", "VTI", "VOO", "QQQ", "MSFT", "AAPL", "VYM", "F", "GE", "AMD",
              "ACB", "APHA", "ZNGA", "NFLX", "TSLA", "BABA", "NVDA", "XRX", "SBUX",
              "TWTR", "GOOG", "FB", "FDX", "DIS", "K", "MNST"]

    # Weed stocks
    weed_stocks = ['ARNA', 'CARA', 'GWPH', 'INSY', 'NTEC', 'MBII', 'PMD', 'TRTC', 'ABBV', 'CANN',
                   'TRPX', 'TLRY', 'VFF', 'ZYNE', 'ACB', 'APHA', 'CGC', 'SHOP', 'CRON',
                   'MCIG', 'STZ', 'HEINY', 'TAP', 'NUGL']
    #stocks.extend(weed_stocks)

    # International Stocks
    international_stocks = ['SPXCY']
    stocks.extend(international_stocks)

    # Penny stocks
    penny_stocks = ['NUGT', 'REKR', 'INSG', 'UAN', 'NVTR', 'CETX', 'SSKN', 'INSY', 'UAMY',
                    'ZIXI', 'GLUU', 'ARTX', 'PLUG', 'MCI', 'INWK', 'TRPX', 'BIOC', 'OTLK',
                    'SESN', 'BCRX', 'CANF', 'ESTR', 'TYME', 'AAU', 'CHKE', 'CTIC', 'EFOI',
                    'SYN', 'MCEP', 'TGB', 'LGCY', 'YOGA', 'IGLD', 'CLIR', 'AMPE', 'AFH',
                    'HEBT', 'PETZ', 'XRF', 'IGC', 'UPL', 'AKG', 'HLTH', 'ENG', 'MYT', 'TWMC',
                    'OPGN', 'XBIO', 'SFUN', 'NGD', 'RVLT', 'HK', 'PES', 'APRN', 'OBLN', 'ADOM',
                    'FCEL', 'NAVB', 'SLS', 'DPW', 'PES', 'AVEO', 'NVCN', 'APVO', 'SFUN',
                    'LIFE', 'YUMA', 'FCEL', 'AMR', 'SNSS', 'PIXY', 'HUSA', 'NAKD',
                    'CYTX', 'TGB', 'LGCY', 'IGC', 'ZNGA', 'TOPS', 'TROV',
                    'JAGX', 'CEI', 'AKR', 'BPMX', 'MYSZ','GNC', 'CHK', 'BLNK', 'SLNO', 'ZIOP']
    penny_stocks = np.unique(penny_stocks)
    #penny_stocks = penny_stocks[:4]
    stocks.extend(penny_stocks)

    # Method list in order
    method = ['Linear Regression', 'Poly2 Regression', 'Poly3 Regression', 'Poly4 Regression',
              'Poly5 Regression', 'K Nearest Neighbors', 'Bayesian Ridge',
              'Multi Layer Perceptron', 'Random Forest Regression', 'Support Vector Regression']

    # Averages To look for
    best_average = 0
    best_forecast = 0
    method_means = [0]*10
    stock_corr_averages = {ps: 0 for ps in penny_stocks}

    #for fo in [2,3,4,5,6,7,8,9,10]:
    forecasts = [2,3,4,5,6,7,8,9,10]
    #forecasts = [7]
    best = {}
    for fo in forecasts:
        start = time.time()
        print("\n\nFORECASTING {} DAYS OUT\n\n".format(fo))
        correlations = np.zeros((len(penny_stocks), 10))
        cor = []

        # Run test for forecast_out
        for idx, ps in enumerate(penny_stocks):
            print(ps)
            vals, best_comb = test_ml(ps, forecast_out=fo, plot=True, month=6, day=18)
            cor.append((ps, vals))
            correlations[idx] = vals

            # Add combination to dict
            best = add_comb(best, best_comb)

        # Compute various correlation values
        correlations = correlations[~np.isnan(correlations).any(axis=1)]
        method_means += np.mean(correlations, axis=0)
        print("INDIVIDUAL CORRELATION MEAN")
        print(np.mean(correlations, axis=0))
        print("ALL CORRELATION MEAN")
        print(np.mean(correlations))


        # Calculate stock-level means and find best one for this forecast_out
        best_mean = 0
        best_stock = ''
        for c in cor:
            #print(c[0], np.mean(c[1]))
            stock_corr_averages[c[0]] += np.mean(c[1])/len(forecasts)
            if(np.mean(c[1]) > best_mean):
                best_mean = np.mean(c[1])
                best_stock = c[0]

        # Check if new best overall forecast-level correlation mean
        if(np.mean(correlations) > best_average):
            best_average = np.mean(correlations)
            best_forecast = fo

        print("\nBEST FROM FORECASTING {} DAYS OUT: {} {}".format(fo, best_mean, best_stock))
        print("TESTING {} FORECAST LENGTH TOOK {} SECONDS".format(fo, time.time()-start))

    print("\n\nBEST FORECAST: {} WITH CORRELATION: {}".format(best_forecast, best_average))

    print("\nAVERAGE METHOD CORRELATION MEANS")#: {}".format(method_means/len(penny_stocks)))
    for k, m in enumerate(method_means):
        print(method[k], m)
        
    print("\nSTOCK AVERAGE CORRELATION OVER DAYS:")
    for key, val in stock_corr_averages.items():
        print(key, val)
    d = {k : v for k,v in filter(lambda t: t[1]>0.15, stock_corr_averages.items())}
    good_stocks = list(d.keys())
    print("STOCKS WITH HIGH CORRELATION: {}".format(good_stocks))
    print("BEST MEAN COMBINATIONS: {}".format(best))
    exit(1)

    start = time.time()
    stock_pred = pd.DataFrame(columns=["Stock", "Price Slope", "Volume Slope", "Good Buy",
                                       "Good Put"])
    for s in good_stocks:
        print(s)
        price_slope = buy_ml(s, best_forecast, plot=True)
        vol_slope = buy_ml_vol(s, best_forecast, plot=True)
        stock_pred = stock_pred.append({"Stock":s, "Price Slope":price_slope,
                         "Volume Slope":vol_slope}, ignore_index=True)

    end = time.time()
    print("\nELAPSED TIME: {}\n".format(end-start))
    stock_pred["Good Buy"] = (stock_pred["Price Slope"]>0) & (stock_pred["Volume Slope"]>0)
    stock_pred["Good Put"] = (stock_pred["Price Slope"]<0) & (stock_pred["Volume Slope"]>0)
    print("Good buys:")
    print(stock_pred[stock_pred['Good Buy']])
    print("\nGood puts:")
    print(stock_pred[stock_pred['Good Put']])
    name="/home/dr/Projects/learn_stocks/towards_data_science/pred_plots/{}_{}/predictions.csv"
    stock_pred.to_csv(name.format(today.day, today.month))

