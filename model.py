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
import pandas_datareader
from datetime import datetime as dt
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


# Pre generated combinations
forecasts = ['reg_forecast', 'poly2_forecast', 'poly3_forecast', 'poly4_forecast',
             'poly5_forecast', 'knn_forecast', 'bayr_forecast', 'rfr_forecast',
             'svr_forecast']
all_combinations = []
for j in range(1,9):
    all_combinations.extend(combinations(forecasts,j))
#print(all_combinations)
#print(len(all_combinations))


def read_data(stock, start, end):                                               
     stock_path = "/home/dr/Projects/multi_model_stock_forecasting/new_stock_data/"+stock+".csv"
     try:
         df = pd.read_csv(stock_path, parse_dates=True, infer_datetime_format=True)
         df = df.set_index(pd.DatetimeIndex(df['Unnamed: 0']))

     except FileNotFoundError:
         print("Data Not Found, Writing {} Data".format(stock))
         new_start = dt(2010, 1, 4)
         os.environ["ALPHAVANTAGE_API_KEY"] = str(np.loadtxt("./api.key", dtype=str))
         df = web.DataReader(stock, 'av-daily', new_start, dt.now(),
                                 access_key=os.getenv("ALPHAVANTAGE_API_KEY"))
         df.index = pd.to_datetime(df.index)
         df.to_csv(stock_path)
         time.sleep(10)

     if(df[start:].empty or not(start in df.index)):
         #print("DATA INCOMPLETE FOR {}. PULLING NEW DATA.".format(stock))
         new_start = dt(2014, 1, 2)
         os.environ["ALPHAVANTAGE_API_KEY"] = str(np.loadtxt("./api.key", dtype=str))
         try:
             #new_df = web.DataReader(stock, 'av-daily', new_start, dt.now(),
             #                    access_key=os.getenv("ALPHAVANTAGE_API_KEY"))
             new_df = web.DataReader(stock, 'av-daily',
                                 access_key=os.getenv("ALPHAVANTAGE_API_KEY"))
         except:
             #print("ERROR IN PULLING DATE FOR: {}".format(stock))
             return pd.DataFrame()

         str_start = start.strftime("%Y-%m-%d")
         str_end = end.strftime("%Y-%m-%d")
         if(not(str_start in new_df.index)):
             #print("I TRIED. THIS IS THE WORST.")
             return pd.DataFrame()
         new_df.to_csv(stock_path)
         new_df.index = pd.to_datetime(new_df.index)
         #print(type(new_df.index))
         #print(new_df[start:end])
         #exit(1)
         #mask = (new_df.index < start) & (new_df.index <= end)
         #print(mask)
         return new_df[start:end]

     return df[start:end]


def holiday(date):
    '''
      Returns True if date is a holiday according to NYSE, else False
    '''
    holidays = [dt(2019,1,1), dt(2019,1,21), dt(2019,2,18), dt(2019,4,19), dt(2019,5,27),
                dt(2019,7,4), dt(2019,9,2), dt(2019,11,28), dt(2019,12,25), 

                dt(2018,1,1), dt(2018,1,15), dt(2018,2,19), dt(2019,3,30), dt(2018,5,28),
                dt(2018,7,4), dt(2018,9,3), dt(2018,11,22), dt(2018,12,25), dt(2018,12,5),
                dt(2018,3,30),
                
                dt(2017,1,1), dt(2017,1,16), dt(2017,2,20), dt(2017,4,14), dt(2017,5,29),
                dt(2017,7,4), dt(2017,9,4), dt(2017,11,23), dt(2017,12,25),
                dt(2016,12,15)]
    return date in holidays


def get_trading_days(years):
    trading_days = []
    days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    for year in years:
        for i in range(1, 13):
            for j in range(1, 32):

                if(j>days[i-1]):
                    continue

                date = dt(year, i, j)
                if(date.weekday() < 5 and not(holiday(date))):
                    trading_days.append(date)


    return np.array(trading_days)


def add_combs(best, combs):
    for comb in combs:
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
    close_vals = data['close'].values
    try:
        ema = [close_vals[0]]
    except IndexError:
        return None

    for idx, v in enumerate(close_vals[1:]):
        ema.append(ema[idx] + 2/(forecast_out+1) * (close_vals[idx+1]-ema[idx]))
    return ema


def test_model():
    start = datetime.datetime(2010, 1, 1)
    end = datetime.datetime(2017, 1, 11)
    df = web.DataReader("AAPL", 'yahoo', start, end)
    #print(df.tail)
    close_px = df['close']
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
                            start=start, end=end)['close']
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
                            start=start, end=end)['close']
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


def test_ml(stock='F', forecast_out=5, month=None, day=None, year=2019, plot=False,
            volume=False):
    # Assume input day is valid trading day
    # Want to separate 1 percent of the data to forecast
    # Today info
    if(month==None or day==None):
        today = datetime.datetime.now()
        month = today.month
        day = today.day

    end_date = dt(year, month, day)
    trading_days = get_trading_days([2017,2018,2019])

    end_idx = np.where(end_date==trading_days)[0][0]
    end = trading_days[end_idx-forecast_out]
    new_start = trading_days[end_idx-forecast_out]
    new_end = trading_days[end_idx]

    # For prediction
    start = datetime.datetime(2016, 4, 1)

    df = read_data(stock, start, end)

    #df = web.DataReader(stock, 'yahoo', start, end)
    #print(df.index)
    df = read_data(stock, start, end)
    if(df.empty):
        #print("SHOULD BE EMPTY")
        return [0]*10, "ERROR"

    df = df[df.index <= end]
    #print(df.tail(forecast_out))
    dfreg = df.loc[:,['close', 'volume']]
    dfreg['HL_PCT'] = (df['high'] - df['low']) / df['close'] * 100.0
    dfreg['PCT_change'] = (df['close'] - df['open']) / df['open'] * 100.0

    # For volume testing
    if(volume):
        dfreg['close'] = dfreg['volume']

    dfreg['EMA'] = get_ema(dfreg, forecast_out)
    if(dfreg['EMA'].empty):
        return [0]*10, "ERROR"

    dfreg['old close'] = dfreg['close']
    dfreg['close'] = dfreg['EMA']

    # For validation
    #print("NEW START: \t{}".format(new_start))
    #print("NEW END: \t{}".format(new_end))
    #print("VALIDATION START: {} END: {}\n".format(new_start, new_end))
    #new_df = web.DataReader(stock, 'yahoo', new_start, new_end)
    new_df = read_data(stock, new_start, new_end)
    #print("TESTING VALIDATION DATA")
    if(new_df.empty):
        return [0]*10, "ERROR"
    #print(new_end)
    new_df = new_df[new_df.index <= new_end]
    #print(new_df)
    #exit(1)
    new_dfreg = new_df.loc[:,['close', 'volume']]
    new_dfreg['HL_PCT'] = (new_df['high'] - new_df['low']) / new_df['close'] * 100.0
    new_dfreg['PCT_change'] = (new_df['close'] - new_df['open']) / new_df['open'] * 100.0

    # Drop missing value
    dfreg.fillna(value=-99999, inplace=True)
    new_dfreg.fillna(value=-99999, inplace=True)

    # Searating the label here, we want to predict the Adjclose
    forecast_col = 'close'
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
    try:
        reg_forecast = fits[0].predict(X_lately)
        poly2_forecast = fits[1].predict(X_lately)
        poly3_forecast = fits[2].predict(X_lately)
        poly4_forecast = fits[3].predict(X_lately)
        poly5_forecast = fits[4].predict(X_lately)
        try:
            knn_forecast = fits[5].predict(X_lately)
        except ValueError:
            #print("KNN ERROR: {}".format(stock))
            #print("Fucking really: {}".format(stock))
            #print(X_lately)
            #print(X_lately.shape)
            knn_forecast = np.zeros(poly5_forecast.shape)
            #exit(1)
        bayr_forecast = fits[6].predict(X_lately)
        rfr_forecast = fits[7].predict(X_lately)
        svr_forecast = fits[8].predict(X_lately)
        mlp_forecast = fits[6].predict(X_lately)
    except AttributeError:
        #print("ISSUES WITH {}".format(stock))
        return [0]*10, {}
        #print(fits)
        #print(threads)
        #print(X_train, y_train)
        #print(X, y)
        #print(stock)
        #print(dfreg)
        #exit(1)
    #mlp_forecast = clfmlp.predict(X_lately)

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
    # I THINK THIS IS FIXED
    #print(as_list[-forecast_out-5:])
    #for asd in as_list[-forecast_out-1:]:
    #    print(asd)
    #print()
    #for asd in new_df.index.tolist():#[:forecast_out]:
    #    print(asd)
    as_list[-forecast_out:] = new_df.index.tolist()[1:]
    try:
        dfreg.index = as_list
    except:
        print("DATA MISALIGNMENT FOR: {}".format(stock))
        #print(new_df)
        #print(dfreg.tail(forecast_out+1))
        #exit(1)
        return [0]*10, {}
    #for asd in as_list[-forecast_out-5:]:
    #    print(asd)
    dfreg[-forecast_out:].index = new_df.index.tolist()[:forecast_out]
    #print(dfreg.tail(forecast_out+1))
    #return [None]*10, None
    #exit(1)

    #
    # Trying to do all combinations
    #
    forecasts = ['reg_forecast', 'poly2_forecast', 'poly3_forecast', 'poly4_forecast',
                 'poly5_forecast', 'knn_forecast', 'bayr_forecast', 'rfr_forecast',
                 'svr_forecast']

    if(plot):
        dfreg['old close'].tail(20).plot(figsize=(20,12), lw=2)
        dfreg['close'].tail(20).plot(figsize=(20,12), lw=2)
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
    

    new_dfreg['Actual close'] = new_df['close']
    if(plot):
        new_dfreg['Actual close'].tail(20).plot(c='g', lw=2)
    fit = np.polyfit([i for i in range(forecast_out)],
                     dfreg['mean_forecast'].values[-forecast_out:], deg=1)

    #print("CALCULATING CORRELATION BETWEEN METHOD AND ACTUAL")
    actual = new_dfreg['Actual close'].tail(forecast_out)

    highest_corr = 0
    best_comb = ''
    num_combs = 0
    correlations = []
    good_combinations = []
    #for j in range(1,9):
    #    for comb in combinations(forecasts, j):
    #        num_combs += 1
    #        comb_dat = dfreg[[*list(comb)]].mean(axis=1).tail(forecast_out)
    #        new_correlation = corr(comb_dat, actual)[0]
    #        correlations.append(new_correlation)
    #        if(new_correlation > 0.4):
    #            good_combinations.append(comb)

    #        if(new_correlation > highest_corr):
    #            highest_corr = new_correlation
    #            best_comb = comb
    for comb in all_combinations:
        num_combs += 1
        comb_dat = dfreg[[*list(comb)]].mean(axis=1).tail(forecast_out)
        new_correlation = corr(comb_dat, actual)[0]
        correlations.append(new_correlation)
        if(new_correlation > 0.4):
            good_combinations.append(comb)

        if(new_correlation > highest_corr):
            highest_corr = new_correlation
            best_comb = comb

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
           knn_corr[0], bayr_corr[0], rfr_corr[0], mean_corr[0], svr_corr[0]), good_combinations
    

def buy_ml(stock, forecast_out=5, month=None, day=None, plot=False, year=2019,
           best_combination=None):
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
              
    trading_days = get_trading_days([2017,2018,2019])
    end_date = dt(year, month, day)
    end_idx = np.where(end_date==trading_days)[0][0]
    end = trading_days[end_idx+forecast_out]

    # For prediction
    start = datetime.datetime(2016, 4, 1)
    end = datetime.datetime(year, month, day) 
    #df = web.DataReader(stock, 'yahoo', start, end)
    df = read_data(stock, start, end)
    #print("BUYING PRICE")
    if(df.empty):
        return [0]*10, "ERROR"
    df = df[df.index <= end]
    dfreg = df.loc[:,['close', 'volume']]
    dfreg['HL_PCT'] = (df['high'] - df['low']) / df['close'] * 100.0
    dfreg['PCT_change'] = (df['close'] - df['open']) / df['open'] * 100.0

    # Drop missing value
    dfreg.fillna(value=-99999, inplace=True)

    # Searating the label here, we want to predict ht eAdjclose
    forecast_col = 'close'
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

    # Support Vector Regressor
    clfsvr = SVR(gamma='auto')

    # Fitting in parallel
    new_threads = []
    models = [clfreg, clfpoly2, clfpoly3, clfpoly4, clfpoly5, clfknn, clfbayr, clfrfr, clfsvr]
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
    svr_forecast = more_fits[8].predict(X_lately)

    # Set up dataframe
    dfreg['reg_forecast'] = np.nan
    dfreg['poly2_forecast'] = np.nan
    dfreg['poly3_forecast'] = np.nan
    dfreg['poly4_forecast'] = np.nan
    dfreg['poly5_forecast'] = np.nan
    dfreg['knn_forecast'] = np.nan
    dfreg['bayr_forecast'] = np.nan
    #dfreg['mlp_forecast'] = np.nan
    dfreg['rfr_forecast'] = np.nan
    dfreg['svr_forecast'] = np.nan

    last_date = dfreg.iloc[-1].name
    last_unix = last_date
    next_unix = last_unix + datetime.timedelta(days=1)
    for i in zip(reg_forecast, poly2_forecast, poly3_forecast, poly4_forecast, poly5_forecast, 
                 knn_forecast, bayr_forecast, rfr_forecast, svr_forecast):
        next_date = next_unix
        next_unix += datetime.timedelta(days=1)
        dfreg.loc[next_date] = list([np.nan for _ in range(len(dfreg.columns)-9)]+list(i))

    dfreg['mean_forecast'] = dfreg[['reg_forecast', 'poly2_forecast', 'poly3_forecast',
                                 'knn_forecast', 'bayr_forecast',# 'mlp_forecast',
                                 'rfr_forecast', 'svr_forecast']].mean(axis=1)
    #print(dfreg.tail(forecast_out)[["reg_forecast", "poly2_forecast", "knn_forecast",
    #                                "poly3_forecast", "poly4_forecast", "poly5_forecast",
    #                                "bayr_forecast", "rfr_forecast", "svr_forecast"]])
    if(plot): 
        dfreg['close'].tail(50).plot(lw=2, figsize=(20,12))
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
        #dfreg['volume'].tail(200).plot()
        #plt.title(stock)
        #plt.show()

    if(not(best_combination==None)):
        #print("USING BEST MEAN: {}".format(best_combination))
        #print("BEST COMBINATION: {}".format(best_combination))
        dfreg['best_mean_forecast'] = dfreg[[*list(best_combination)]].mean(axis=1)
        #print(dfreg['best_mean_forecast'].tail(forecast_out+1))
        #print("FORECASTS: {}".format([i for i in range(forecast_out)]))
        #print("VALUES: {}".format(dfreg['best_mean_forecast'].values[-forecast_out:],deg=1))
        fit = np.polyfit([i for i in range(forecast_out)],
                         dfreg['best_mean_forecast'].values[-forecast_out:], deg=1)
        #print(fit)
           
    else:
        try:
            fit = np.polyfit([i for i in range(forecast_out)],
                             dfreg['mean_forecast'].values[-forecast_out:], deg=1)
        except:
            print("\n\nI DONT REMEMBER WHAT THIS IS\n\n".format(forecast_out))
            fit = [dfreg['mean_forecast'].values[-1] - dfreg['close'].values[-1], 2]

    string = "SHOULD GO UP" if(fit[0] > 0) else "SHOULD GO DOWN"
    #print("{} {}".format(stock, string))
    #print("PRICE HAS BEEN FIT: {}".format(fit[0]))
    #print("FIT SLOPE: {}".format(fit[0]))
    return fit[0], dfreg['close'].values[-forecast_out-1]
    #if(fit[0] > 0):
    #    return fit[0]
    #else:
    #    return fit[0]


def buy_ml_vol(stock, forecast_out=5, month=None, day=None, plot=False, year=2019,
               best_combination=None):
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
    start = datetime.datetime(2016, 4, 1)
    end = datetime.datetime(year, month, day) 
    #df = web.DataReader(stock, 'yahoo', start, end)
    df = read_data(stock, start, end)
    #print("BUYING")
    if(df.empty):
        return [0]*10, "ERROR"
    dfreg = df.loc[:,['close', 'volume']]
    dfreg['HL_PCT'] = (df['high'] - df['low']) / df['close'] * 100.0
    dfreg['PCT_change'] = (df['close'] - df['open']) / df['open'] * 100.0

    # Drop missing value
    dfreg.fillna(value=-99999, inplace=True)

    # Searating the label here, we want to predict ht eAdjclose
    forecast_col = 'volume'
    dfreg['label'] = dfreg[forecast_col].shift(-forecast_out)
    X = np.array(dfreg.drop(['label'], 1))

    # Scale X for linear regression
    try:
        X = preprocessing.scale(X)
    except ValueError:
        print("DATA: {}".format(X))
        print("STOCK: {}".format(stock))
        print("START PERIOD: {}".format(start))
        print("END PERIOD: {}".format(end))
     

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
    #clfmlp = MLPRegressor(hidden_layer_sizes=(100,100,100), learning_rate='adaptive',
    #          solver='adam', max_iter=5, verbose=False)
    #clfmlp.fit(X_train, y_train)

    # Random Forest Regressor
    clfrfr = RFR(n_estimators=15, random_state=0)

    # Support Vector Regressor
    clfsvr = SVR(gamma='auto')

    # Fitting
    threads = []
    models = [clfreg, clfpoly2, clfpoly3, clfpoly4, clfpoly5, clfknn, clfbayr, clfrfr, clfsvr]
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
    poly4_forecast = fits[3].predict(X_lately)
    poly5_forecast = fits[4].predict(X_lately)
    knn_forecast = fits[5].predict(X_lately)
    bayr_forecast = fits[6].predict(X_lately)
    #mlp_forecast = clfmlp.predict(X_lately)
    rfr_forecast = fits[7].predict(X_lately)
    svr_forecast = fits[8].predict(X_lately)

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
                 knn_forecast, bayr_forecast, rfr_forecast, svr_forecast):
        next_date = next_unix
        next_unix += datetime.timedelta(days=1)
        dfreg.loc[next_date] = list([np.nan for _ in range(len(dfreg.columns)-9)]+list(i))

    dfreg['mean_forecast'] = dfreg[['reg_forecast', 'poly2_forecast', 'poly3_forecast',
                                 'knn_forecast', 'bayr_forecast',# 'mlp_forecast',
                                 'rfr_forecast']].mean(axis=1)
    if(plot): 
        dfreg['volume'].tail(50).plot(lw=2, figsize=(20,12))
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
        #dfreg['volume'].tail(200).plot()
        #plt.title(stock)
        #plt.show()

    #if(not(best_combination==None)):
    #    dfreg['best_mean_forecast'] = dfreg[[*list(best_combination)]].mean(axis=1)
    #    fit = np.polyfit([i for i in range(forecast_out)],
    #                      dfreg['best_mean_forecast'].values[-forecast_out:], deg=1) 
    #else:
    try:
        fit = np.polyfit([i for i in range(forecast_out)],
                         dfreg['mean_forecast'].values[-forecast_out:], deg=1)
    except:
        print("FORECASTING {} DAY OUT".format(forecast_out))
        fit = [dfreg['mean_forecast'].values[-1] - dfreg['close'].values[-1], 2]

    string = "VOLUME SHOULD GO UP" if(fit[0] > 0) else "VOlUME SHOULD GO DOWN"
    #print("{} {}".format(stock, string))
    #print("VOLUME HAS BEEN FIT: {}".format(fit[0]))
    return fit[0], dfreg['volume'].values[-forecast_out-1]



def forecast_out_sweep(stocks, forecasts, plot=False):
    # Averages To look for
    best_average = 0
    best_forecast = 0
    method_means = [0]*10
    stock_corr_averages = {ps: 0 for ps in penny_stocks}
    best = {}

    for fo in forecasts:
        start = time.time()
        print("\n\nFORECASTING {} DAYS OUT\n\n".format(fo))
        correlations = np.zeros((len(penny_stocks), 10))
        cor = []

        # Run test for forecast_out
        for idx, ps in enumerate(penny_stocks):
            print(ps)
            try:
                vals, good_combs = test_ml(ps, forecast_out=fo, plot=plot, day=18, month=6)
            except:
                print("HAD AN ERROR")
                vals = np.zeros(10)
                good_combs = {}

            cor.append((ps, vals))
            correlations[idx] = vals

            # Add combination to dict
            best = add_combs(best, good_combs)

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

    # Method list in order
    method = ['Linear Regression', 'Poly2 Regression', 'Poly3 Regression', 'Poly4 Regression',
              'Poly5 Regression', 'K Nearest Neighbors', 'Bayesian Ridge',
              'Multi Layer Perceptron', 'Random Forest Regression', 'Support Vector Regression']
    print("\nAVERAGE METHOD CORRELATION MEANS")#: {}".format(method_means/len(penny_stocks)))
    for k, m in enumerate(method_means):
        print(method[k], m)
        
    print("\nSTOCK AVERAGE CORRELATION OVER DAYS:")
    for key, val in stock_corr_averages.items():
        print(key, val)

    d = {k : v for k,v in filter(lambda t: t[1]>0.15, stock_corr_averages.items())}

    good_stocks = list(d.keys())
    print("STOCKS WITH HIGH CORRELATION: {}".format(good_stocks))
    print("TOP 10 BEST MEAN COMBINATIONS:")#.format(best))
    for i in range(10):
        best_combination = max(best, key=best.get)
        print(best_combination, best[best_combination])
        del best[best_combination]
    print("Histogram of best values:")
    fig, ax = plt.subplots()
    ax.hist(best.values(), bins=20)
    plt.show()
    fig.savefig("./hist_of_number_of_good_corrs.png")
    exit(1)
    #import operator
    #print(sorted(best.iteritems(), key=operator.itemgetter(1)))
    #for key, val in best.items():
        #if(val > len(stocks)/2):
        #    print(key, val)

    return good_stocks


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
    #penny_stocks = penny_stocks[:3]
    stocks.extend(penny_stocks)

    simulate(penny_stocks)
    exit(1)

    forecasts = [2,3,4,5,6,7,8,9,10]
    #forecasts = [4,5,6,7,8,9]
    #forecasts = [3,4]
    good_stocks = forecast_out_sweep(penny_stocks, forecasts, plot=True)
    print(good_stocks)

    exit(1)

    start = time.time()
    stock_pred = pd.DataFrame(columns=["Stock", "Price Slope", "volume Slope", "Good Buy",
                                       "Good Put"])
    for s in good_stocks:
        print(s)
        price_slope = buy_ml(s, best_forecast, plot=True)
        vol_slope = buy_ml_vol(s, best_forecast, plot=True)
        stock_pred = stock_pred.append({"Stock":s, "Price Slope":price_slope,
                         "volume Slope":vol_slope}, ignore_index=True)

    end = time.time()
    print("\nELAPSED TIME: {}\n".format(end-start))
    stock_pred["Good Buy"] = (stock_pred["Price Slope"]>0) & (stock_pred["volume Slope"]>0)
    stock_pred["Good Put"] = (stock_pred["Price Slope"]<0) & (stock_pred["volume Slope"]>0)
    print("Good buys:")
    print(stock_pred[stock_pred['Good Buy']])
    print("\nGood puts:")
    print(stock_pred[stock_pred['Good Put']])
    name="/home/dr/Projects/learn_stocks/towards_data_science/pred_plots/{}_{}/predictions.csv"
    stock_pred.to_csv(name.format(today.day, today.month))


