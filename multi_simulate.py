import numpy as np
import pandas as pd
import pandas_datareader.data as web
import pandas_datareader
import os
import dateutil
from shutil import copy2

from datetime import datetime as dt
from model import test_ml, buy_ml, buy_ml_vol, add_combs, read_data, get_trading_days
from matplotlib import pyplot as plt
import sys
import time
from contextlib import redirect_stderr
from multiprocessing import Process


def rename(file_name):
    '''
      Renames most recent simulation.out to proper number
    '''
    dir_name = file_name[:file_name.rfind('/')]
    num_restarts = -1
    for fi in os.listdir(dir_name):
        num_restarts += int(".out" in fi)

    copy2(file_name, dir_name+"/{}.simulation.out".format(num_restarts))


def gather_data(file_name):
    '''
        Gathers past data from old simulation
    '''
    with open(file_name) as fi:
        lines = fi.readlines()
        last_idx = 0
        for idx, l in enumerate(lines):
            if("CYCLE TOOK" in l):
                last_idx = idx
                #print(idx, l)
        if(last_idx == 0):
            # No cycles complete, just restart from beginning
            return [None]*7

        # Doesn't change based on restarted or not
        lines = lines[:last_idx]
        state_line = lines[last_idx-11]
        if("NOT PURCHASING NEXT CYCLE" in state_line):
            last_bought = False
        else:
            last_bought = True

        cyc_number = int(lines[last_idx-8].split(" ")[1])
        cyc_end = dt(*[int(x) for x in lines[last_idx-6].split(" ")[3].split('-')]) 
        pred_pcts = [0]
        mkt_pcts = [0]
        end_dates = [dt.strptime(lines[2].split(" ")[2], '%Y-%m-%d')]

        for l in lines:
            if("STARTING MONEY" in l):
                start_money = float(l.split(":")[1])
            elif("NEW TOTAL MONEY" in l):
                new_total_money = float(l.split(":")[1])
            elif("PREDICTION PERCENTAGE" in l):
                pred_pcts.append(float(l.split(":")[1]))
            elif("MARKET PERCENTAGE" in l):
                mkt_pcts.append(float(l.split(":")[1]))
            elif(("CYCLE END DATE" in l) and 
                 not(dt.strptime(l.split(" ")[3], '%Y-%m-%d') in end_dates)):
                end_dates.append(dt.strptime(l.split(" ")[3], '%Y-%m-%d'))

    return start_money, new_total_money, pred_pcts, mkt_pcts, last_bought, cyc_number, \
           cyc_end, end_dates


def write_restart(file_name, start_money, new_total_money, pred_pcts, mkt_pcts, last_bought,
                  cyc_number, cyc_end, end_dates):
    '''
      Takes gathered data, writes to new simulation out file
    '''
    if(start_money == None):
        with open(file_name, 'a+') as fout:
            fout.write("NOWHERE TO RESTART, PLEASE DELETE DIRECTORY AND RUN AGAIN")
            exit(1)
        fout.close()
    else: 
        new_line = str('X'*80)
        with open(file_name, "a+") as fout:
            fout.write("\n\n{}\n".format(new_line))
            fout.write("{}\n".format(new_line))
            fout.write("\n\nRESTARTING FROM CYCLE {}\n".format(cyc_number))
            fout.write("\nORIGINAL STARTING MONEY: {}".format(start_money))
            fout.write("\nCURRENT MONEY: {}".format(new_total_money))
            fout.write("\nBUYING NEXT CYCLE: {}".format(last_bought))
            fout.write("\n\nDATE\tPREDICTION\tMARKET SO FAR\n")
            print(len(mkt_pcts))
            for p in mkt_pcts:
                print(p)
            print()
            print(len(pred_pcts))
            print(len(end_dates))
            pred_pcts = pred_pcts[:len(end_dates)]
            mkt_pcts = mkt_pcts[:len(end_dates)]
            for i in range(len(pred_pcts)):
                fout.write("{0}\t{1:.5f}\t{2:.5f}\n".format(end_dates[i], pred_pcts[i],
                                                           mkt_pcts[i]))
            fout.write("\nSTARTING NEXT CYCLE AT: {}\n\n".format(cyc_end))
        fout.close()


def initialize_directory(stocks, start_date, end_date, start_money, forecast_out,
                         loss_threshold, buy_threshold, verbose):
    '''
      Initializes the directory to store data
    '''
    dir_name = 's_'+str(start_date.month)+'_'+str(start_date.day)+'_'+str(start_date.year)+'_'+\
               'e_'+str(end_date.month)+'_'+str(end_date.day)+'_'+str(end_date.year)+'_'+\
               str(len(stocks))+'_'+str(int(start_money))+'_'+str(int(forecast_out))
    try:
        hold_name = '/long_threshold_sweep_sims/multiprocess_sim_l{}_b{}'.format(loss_threshold,
                    buy_threshold)
        path = os.path.dirname(os.path.realpath(__file__))+hold_name
        file_name = path+dir_name+'/simulation.out'
        os.mkdir(path + dir_name)
        sys.stderr = open(path+dir_name+'/simulation.err', 'w+')

    except FileExistsError:
        sys.stderr = open(path+dir_name+'/simulation.err', 'w+')
        if(verbose):
            print("DIRECTORY ALREADY EXISTS, RESTARTING")

        if("simulation.out.png" in os.listdir(path+dir_name)):
            print("SIMULATION DONE")
            return None
            
        rename(file_name)
        gathered_data = gather_data(file_name)
        write_restart(file_name, *gathered_data)
        #print(type(gathered_data))
        print(*gathered_data)
        return file_name, gathered_data

    with open(file_name, 'w+') as fout:
        fout.write("SIMULATION OF STOCK FORECASTING\n\n")
        fout.write("START DATE: {}\n".format(start_date))
        fout.write("END DATE: {}\n".format(end_date))
        fout.write("STARTING MONEY: {}\n".format(start_money))
        fout.write("FORECASTING {} DAYS OUT\n".format(forecast_out))
        fout.write("LOSS THRESHOLD: {}\n".format(loss_threshold))
        fout.write("BUY THRESHOLD: {}\n\n".format(buy_threshold))
        fout.write("USING STOCKS:\n")
        try:
            for i in range(10, len(stocks), 10):
                fout.write(str(stocks[i-10:i])+'\n')
            fout.write(str(stocks[i:])+'\n\n\n')
        except UnboundLocalError:
            fout.write(str(stocks)+'\n\n\n')
        new_line = str('X'*80)
        fout.write(new_line+'\n')
        fout.write(new_line)
        fout.write('\n\n')

    fout.close()
    return file_name, (start_money, start_money, [0], [0], True, 0, \
           end_hold_date(forecast_out, start_date.day, start_date.month, start_date.year), \
           [start_date])


def write_predictions(file_name, cycle, df):
    with open(file_name, 'a+') as fout:
        fout.write("\nCYCLE {} PREDICTIONS\n".format(cycle))
        if(df.empty):
            fout.write("\nNO GOOD BUYS\n\n")
        else:
            fout.write(df[["Stock", "Last Price", "Price Slope", "Price Ratio",
                               "Volume Ratio", "Good Buy"]].to_string())
            fout.write("\n\nCYCLE {} BUYS\n".format(cycle))
    fout.close()


def write_buys(file_name, bought_stocks):
    with open(file_name, 'a+') as fout:
        for key, vals in bought_stocks.items():
            money = float(vals[0])*(float(vals[2]) - float(vals[1]))
            fout.write("\nBOUGHT {0} OF {1} AT {2:.4f}. SOLD AT {3:.4f}. MADE {4:.4f}".format(
                       vals[0], key, vals[1], vals[2], money))
    fout.close()


def write_cycle(file_name, cycle, new_money, total_sold, total_spent, denominator, start_date,
                end_hold, start, pred_percent, mkt_percent):
    with open(file_name, 'a+') as fout:
        fout.write("\n\nCYCLE {} RESULTS\n".format(cycle))
        fout.write("CYCLE START DATE: {}\n".format(start_date))
        fout.write("CYCLE END DATE: {}\n".format(end_hold))
        fout.write("NEW TOTAL MONEY: {0:.4f}\n".format(new_money))
        fout.write("MONEY MADE THIS CYCLE: {0:.4f}\n".format(total_sold-total_spent))
        fout.write("PERCENTAGE MADE ON MONEY SPENT THIS CYCLE: {0:.4f}\n".format(
                   100*(total_sold/denominator-1)))
        fout.write("CUMULATIVE PREDICTION PERCENTAGE: {}\n".format(pred_percent))
        fout.write("CUMULATIVE MARKET PERCENTAGE: {}\n".format(mkt_percent))
        fout.write("CYCLE TOOK {0:.4f} SECONDS\n\n\n".format(time.time()-start))
        fout.write("")
        new_line = str('X'*80)
        fout.write(new_line+'\n')
        fout.write(new_line)
        fout.write("\n\n\nCYCLE {} BUYS".format(cycle+1))

    fout.close()


def write_final(file_name, cycle, new_money, start_money, start_time, end_time, pred, mkt,
                end_hold_dates):
    with open(file_name, 'a+') as fout:
        fout.write("\n\nSIMULATION COMPLETE AFTER {} CYCLES\n\n".format(cycle))
        fout.write("PREDICTION\tMARKET\n\n")
        pred = np.unique(pred)
        for i in range(len(pred)):
            fout.write("{0}\t{1:.5f}\t{2:.5f}\n".format(end_hold_dates[i], pred[i], mkt[i]))

        fout.write("\nSTART MONEY: {}\n".format(start_money))
        fout.write("NEW TOTAL MONEY: {}\n".format(new_money))
        fout.write("PERCENT INCREASE: {}\n".format(100*(new_money/start_money-1)))
    fout.close()


def progress_bar(current_num, total_num):
    '''
      Progress bar
    '''
    start = "[" + "="*int(current_num/total_num*10)
    end = " "*(11-len(start)) + "]"
    sys.stdout.write(start + end + ' %.2f%% \r' % (current_num/total_num*100))
    sys.stdout.flush()
    if(current_num == total_num-1):
        print("[==========] 100.00%")


def new_deposit(deposits, start_date, file_name):
    # three deposits on 10-6-2019, two on 28-2-2019
    deposit_dates = [dt(2019,1,16), dt(2019,1,28), dt(2019,2,15), dt(2019,2,25),
                     dt(2019,2,28), dt(2019,4,26), dt(2019,6,10)]
    deposit_amounts = [150., 20., 60.45, 90., 530.+10., 150., 1000.+5000.+1000.]
    deposit_money = 0
    for i in range(len(deposits)):
        if(not(deposits[i]) and (start_date >= deposit_dates[i])):
            #print("\n\nDEPOSITING {} FOR START DATE OF: {}\n\n".format(deposit_amounts[i],
            #           start_date))
            deposits[i] = True
            deposit_money += deposit_amounts[i]

            with open(file_name, 'a+') as fout:
                fout.write("\n\nDEPOSITING {} FOR START DATE OF: {}\n".format(
                                    deposit_amounts[i], start_date))
            fout.close()
    return deposit_money, deposits


def holiday(date):
    '''
      Returns True if date is a holiday according to NYSE or special closure, else False
    '''
    holidays = [dt(2019,1,1), dt(2019,1,21), dt(2019,2,18), dt(2019,4,19), dt(2019,5,27),
                dt(2019,7,4), dt(2019,9,2), dt(2019,11,28), dt(2019,12,25), 

                dt(2018,1,1), dt(2018,1,15), dt(2018,2,19), dt(2019,3,30), dt(2018,5,28),
                dt(2018,7,4), dt(2018,9,3), dt(2018,11,22), dt(2018,12,25), dt(2018, 12, 5),
                dt(2018,3,30),

                dt(2017,1,1), dt(2017,1,16), dt(2017,2,20), dt(2017,4,14), dt(2017,5,29),
                dt(2017,7,4), dt(2017,9,4), dt(2017,11,23), dt(2017,12,25)]
    return date in holidays


def end_hold_date(forecast_out, day, month, year):
    '''
      Calculates valid end date for holding period. forecast_out open trading days between
      input and output dates
    '''

    days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    trading_days = 0
    trial_year = year
    trial_day = day
    trial_month = month
    while(trading_days <= forecast_out):
        trial_year = year+1 if(trial_day+1 > 31 and trial_month==12) else trial_year
        trial_month = 1 if(trial_day+1 > days[trial_month-1] and trial_month==12) \
                      else trial_month+1 if(trial_day+1 > days[trial_month-1]) \
                      else trial_month
        trial_day = 2 if(trial_day+1 > days[trial_month-1] and trial_month==12) \
                      else trial_day+1 if(trial_day+1 <= days[month-1]) else \
                      trial_day+1-days[month-1]

        trial_date = dt(trial_year, trial_month, trial_day)
        if(trial_date.weekday()<5 and not(holiday(trial_date))):
            trading_days += 1
            if(trading_days == forecast_out):
                pred_year = trial_year
                pred_month = trial_month
                pred_day = trial_day

                break

    return dt(pred_year, pred_month, pred_day)


def best_combinations(best, vol_best):
    if(not(bool(best))):
        #print("NO BEST MEAN, USING TOTAL MEAN FOR PRICE")
        #print(best)
        best = ('reg_forecast', 'poly2_forecast', 'poly3_forecast', 'knn_forecast',
                'bayr_forecast','rfr_forecast', 'svr_forecast')
    else:
        best = max(best, key=best.get)

    if(not(bool(vol_best))):
        #print("NO BEST MEAN, USING TOTAL MEAN FOR VOLUME")
        #print(vol_best)
        vol_best = ('reg_forecast', 'poly2_forecast', 'poly3_forecast', 'knn_forecast',
                'bayr_forecast','rfr_forecast', 'svr_forecast')
    else:
        vol_best = max(vol_best, key=vol_best.get)

    return [best, vol_best]


def test(stocks, forecast_out, start_date, verbose=False):
    '''
        Takes in stocks, forecast length, start date, runs tests from model.py to return 
        best correlation averages, high correlation mean combinations for volume and price
    '''

    # To keep track of data
    stock_corr_averages = {s: 0 for s in stocks}
    correlations = np.zeros((len(stocks), 10))
    best = {}
    vol_best = {}
    cor = []

    # Test for stock price and volume
    for idx, s in enumerate(stocks):
        try:
            vals, good_combs = test_ml(s, forecast_out, year=start_date.year,
                            month=start_date.month, day=start_date.day)

            if(good_combs=="ERROR"):
                #print("DIDN'T WORK FOR: {}".format(s))
                continue

            vol_vals, vol_good_combs = test_ml(s, forecast_out, year=start_date.year,
                            month=start_date.month, day=start_date.day, volume=True)
        except KeyError:
            #print("NO DATA FOR {}".format(s))
            vals = [0]*10
            good_combs = {}

        if(good_combs=="ERROR" or vol_good_combs=="ERROR"):
            continue

        cor.append((s, vals))
        correlations[idx] = vals

        best = add_combs(best, good_combs)
        vol_best = add_combs(vol_best, vol_good_combs)
        if(verbose):
            progress_bar(idx, len(stocks))

    correlations = correlations[~np.isnan(correlations).any(axis=1)]
    for c in cor:
        stock_corr_averages[c[0]] += np.mean(c[1])
    return best, vol_best, correlations, stock_corr_averages


def forecast(good_stocks, forecast_out, start_date, best_combination, buy_threshold=0,
             verbose=False):
    '''
      Takes in stocks, forecast length, start date, best mean combination for price and volume
    '''
    # Dataframe for forecasting
    stock_pred = pd.DataFrame(columns=['Stock', 'Last Price', 'Price Slope', 'Last Volume',
                                       'Volume Slope', 'Good Buy', 'Good Put'])
    
    #print("FORECASTING")
    for idx, s in enumerate(good_stocks):
        price_slope, last_price = buy_ml(s, forecast_out, month=start_date.month,
              day=start_date.day, year=start_date.year, best_combination=best_combination[0])

        vol_slope, last_vol = buy_ml_vol(s, forecast_out, month=start_date.month,
           day=start_date.day, year=start_date.year, best_combination=best_combination[1])

        stock_pred = stock_pred.append({"Stock":s, "Price Slope":price_slope,
                         "Volume Slope":vol_slope, "Last Price":last_price,
                         "Last Volume":last_vol}, ignore_index=True)

        if(verbose):
            progress_bar(idx, len(good_stocks))
    #print("\n\nIN FORECAST")
    #print(stock_pred)
    #print()

    # Add columns for more relevant data
    stock_pred['Price Slope'] = stock_pred['Price Slope'].astype(np.float64)
    stock_pred['Volume Slope'] = stock_pred['Volume Slope'].astype(np.float64)
    stock_pred['Good Buy'] = pd.Series((stock_pred['Price Slope']>0) & 
                                       (stock_pred['Volume Slope']>0))
    stock_pred['Good Put'] = pd.Series((stock_pred['Price Slope']<0) & 
                                       (stock_pred['Volume Slope']>0))
    stock_pred['Price Ratio'] = stock_pred['Price Slope']/stock_pred['Last Price']
    stock_pred['Volume Ratio'] = stock_pred['Volume Slope']/stock_pred['Last Volume']
    #print(stock_pred)

    # EMPTY IF NO GOOD BUYS
    stock_pred = stock_pred[(stock_pred['Good Buy']) &
                            (stock_pred['Price Ratio']>buy_threshold)].sort_values(
                             by=['Price Ratio', 'Volume Ratio'], ascending=False)
    #print("\n\nSTOCK PREDICTION JUST AFTER")
    #print(stock_pred)
    #print()
    return stock_pred


def buy(buy_stocks, start_date, end_hold, new_money, file_name, last_bought, threshold=0,
        verbose=False):
    bought_stocks = {}
    num_no_buys = 0
    total_spent = 0
    total_sold = 0

    old_money = np.copy(new_money)
    old_total_spent = np.copy(total_spent)
    old_total_sold = np.copy(total_sold)
    
    #print("STOCKS TO BUY: {}".format(buy_stocks))
    while(num_no_buys < len(buy_stocks)):
        for bs in buy_stocks:
            bs_dat = read_data(bs, start_date, dt.now())['adjusted close']
            #if(verbose):
            #    print("Before hold: {} at {}".format(bs, bs_dat[0]))

            if(bs_dat.values[0] < new_money):
                 total_spent += bs_dat.values[0]
                 new_money = new_money - bs_dat.values[0]
                 if(bs not in bought_stocks):
                     bought_stocks[bs] = [1, bs_dat.values[0], 0]
                 else:
                     bought_stocks[bs][0] += 1
                 num_no_buys = 0
            else:
                num_no_buys += 1

    if(verbose):
        print("STOCKS PURCHASED")
        for key, val in bought_stocks.items():
            print(key, val)

    for bs, vals in bought_stocks.items():
        hold_dat = read_data(bs, end_hold, dt.now())['adjusted close']
        bought_stocks[bs][2] = hold_dat.values[0]
        total_sold += hold_dat.values[0]*vals[0]
        new_money += hold_dat.values[0]*vals[0]
        money = (hold_dat.values[0] - vals[1])*vals[0]

        # Prints relevant details
        if(verbose):
            print("BOUGHT {} OF {} AT {}".format(vals[0], bs, vals[1]))
            print("AFTER HOLD: {}".format(hold_dat.values[0]))
            print("MONEY MADE FROM PURCHASE OF {} IS: {} WAS {}".format(bs, money,
              "GOOD" if money>0 else "BAD"))
            print("LAST BOUGHT: {}".format(last_bought))
            print("OLD MONEY: {} NEW MONEY: {}".format(old_money, new_money))

    # Doesn't buy if last buy was bad
    # This buy good and last buy good
    if((new_money >= (1-threshold)*old_money) and (last_bought)):
        #print("GOOD GOOD")
        #print("SHOULD BE HERE")
        last_bought = True
        write_buys(file_name, bought_stocks)
        return total_spent, total_sold, new_money, last_bought

    # This buy good and last buy bad
    elif((new_money > (1-threshold)*old_money) and not(last_bought)):
        #print("GOOD BAD")
        last_bought = True
        with open(file_name, 'a+') as fout:
            fout.write("\nLAST CYCLE LOST MONEY. THIS CYCLE MADE MONEY")
            fout.write("\nMAKING PURCHASES NEXT CYCLE\n")
        fout.close()
        return old_total_spent, old_total_sold, old_money, last_bought

    # This buy bad and last buy good
    elif((new_money <= (1-threshold)*old_money) and (last_bought)):
        #print("BAD GOOD")
        last_bought = False
        write_buys(file_name, bought_stocks)
        with open(file_name, 'a+') as fout:
            fout.write("\n\nTHIS CYCLE LOST MONEY")
            fout.write("\nNOT PURCHASING NEXT CYCLE\n")
        fout.close()
        return total_spent, total_sold, new_money, last_bought

    # This buy bad and last buy bad
    elif((new_money <= (1-threshold)*old_money) and not(last_bought)):
        #print("BAD BAD")
        last_bought = False
        with open(file_name, 'a+') as fout:
            fout.write("\nLAST CYCLE LOST MONEY")
            fout.write("\nNO PURCHASES MADE THIS CYCLE")
            fout.write("\nNOT PURCHASING NEXT CYCLE\n")
        fout.close()
        return old_total_spent, old_total_sold, old_money, last_bought

    # Undefined case
    else:
        raise Exception("UNDEFINED BEHAVIOR IN STOCK PURCHASING")
        exit(1)
            
    #return total_spent, total_sold, new_money


def simulate(stocks, forecast_out=7, start_day=2, start_month=1, start_year=2017,
             end_day=None, end_month=None, end_year=None, start_money=1000, plot=False,
             deposits=None, loss_threshold=0, buy_threshold=0, verbose=False):
    '''
      stocks - A list of stocks to test and forecast with
      forecast_out - the number of days to forecast over
      start_day - day of month to start simulation
      start_month - month to start simulation
      start_year - year to start simulation
      end_day - day of month to end simulation
      end_month - month to end simulation
      end_year - year to end simulation
      start_money - the amount of money to start the simulation with
      plot - if True, plots percentage increase compared to market
    '''

    first_start = time.time()
    if(verbose):
        print("STOCKS BEING LOOKED AT: {}\n".format(stocks))
        print("STARTING WITH: ${}".format(start_money))

    # Make start date if none given
    if(start_day==2 and start_month==1 or start_year==2017):
        if(verbose):
            print("DEFAULT START DATE")

    start_date = dt(start_year, start_month, start_day)

    if(end_day==None or end_month==None or start_year==None):
        if(verbose):
            print("DEFAULT END DATE OF TODAY")
        end_date = dt.now()
    else:
        end_date = dt(end_year, end_month, end_day) 

    if(deposits is not None):
        start_money = 50

    # Initialize directory for plotting, log files, etc.
    file_name, gathered_data = \
         initialize_directory(stocks, start_date, end_date, start_money, forecast_out,\
                                     loss_threshold, buy_threshold, verbose)
    _, new_money, pred_percentages, mkt_percentages, last_bought, dates, end_hold, \
    end_hold_dates = (x for x in gathered_data)

    if(file_name == None):
        return None

    # Initializes plotting if plot
    if(plot):
        fig, ax = plt.subplots()
        start_market = read_data("^GSPC", start_date, dt.now())["adjusted close"].values[0]

    # To keep track of total earnings
    total_spent = 0 # Do these even matter?
    total_sold = 0
    if(verbose):
        print("START DATE: {}".format(start_date))
    #end_hold = end_hold_date(forecast_out, start_day, start_month, start_year)
    #new_money = np.copy(start_money)
    #pred_percentages, mkt_percentages = ([] for i in range(2))
    #last_bought = True
    while(end_hold.date() < end_date.date()):

        if(verbose):
            print("START DATE: \t{}".format(start_date))
            print("END HOLD DATE: \t{}".format(end_hold))
            print()

        start = time.time()
        
        # Test each stock, hold on to relevant data
        if(verbose):
            print("\nTESTING")
        best, vol_best, correlations, stock_corr_averages = test(stocks, forecast_out,
                                                                  start_date, verbose)

        # Grab data from testing results
        best_combination = best_combinations(best, vol_best)
        try:
            num = best[best_combination[0]]
            vnum = vol_best[best_combination[1]]
        except:
            num = 0
            vnum = 0

        if(verbose):
            print(best_combination)
            print("BEST MEAN PRICE COMBINATION: {} WITH {} BESTS".format(best_combination[0],
                                                                         num))
            print("BEST MEAN VOLUME COMBINATION: {} WITH {} BESTS".format(best_combination[1],
                                                                          vnum))

        # Get list of stocks that had high correlation on average
        d = {k:v for k,v in filter(lambda t: t[1]>0.15, stock_corr_averages.items())}
        good_stocks = list(d.keys())

        # Forecast stocks and get good buys
        stock_pred = forecast(good_stocks, forecast_out, start_date, best_combination, 
                              buy_threshold, verbose)
        write_predictions(file_name, dates, stock_pred)
        buy_stocks = stock_pred[stock_pred['Good Buy']]['Stock'].values

        # Do buying
        total_spent, total_sold, new_money, last_bought = buy(buy_stocks, start_date, end_hold,
                                  new_money, file_name, last_bought, loss_threshold, verbose)

        # Print out results from trading cycle
        denominator = 999999999 if (total_spent==0) else total_spent
        if(verbose):
            print("AFTER CYCLE {}".format(dates))
            print("DIFFERENCE: {}\nPERCENTAGE: {}".format(total_sold-total_spent,
                                                          100*(total_sold/denominator-1)))
            print("NEW TOTAL MONEY: {}\n\n".format(new_money))

        # Plot agains market
        if(plot):
            end_market = read_data("^GSPC", end_hold, dt.now())["adjusted close"].values[0]
            if(verbose):
                print("END HOLD PERIOD: {}".format(end_hold))
                print("\nEND MARKET HOLD: {}".format(end_market))
                print("START MARKET: {}".format(start_market))
                print()
            mkt_percentages.append(100*(end_market/start_market - 1))
            
            # Plot percent from model
            pred_percentages.append(100*(new_money/start_money-1))

        # Write data
        write_cycle(file_name, dates, new_money, total_sold, total_spent, denominator,
                    start_date, end_hold, start, pred_percentages[-1], mkt_percentages[-1])

        # Update days for next trading cycle
        start_date = end_hold
        end_hold_dates.append(end_hold)
        end_hold = end_hold_date(forecast_out, end_hold.day, end_hold.month, end_hold.year)
        if(deposits is not None):
            deposit_money, deposits = new_deposit(deposits, start_date, file_name)
            new_money = new_money + float(deposit_money)
            start_money = start_money + float(deposit_money)
        dates += 1

    # Print out final results
    if(verbose):
        print("START DATE: {}, END DATE: {}".format(start_date, end_hold))
        print("TOTAL SPEND: {}\nTOTAL SOLD: {}\nDIFFERENCE: {}".format(
              total_spent, total_sold, total_sold-total_spent))
        print("INITIAL INVESTMENT: {}, MONEY NOW: {}, PERCENT CHANGE: {}".format(start_money,
               new_money, 100*(new_money/start_money - 1)))
    percent = 100*(new_money/start_money - 1)

    if(plot):
        end_hold_dates = np.unique(end_hold_dates)
        mkt_percentages = mkt_percentages[:len(end_hold_dates)]
        pred_percentages = pred_percentages[:len(end_hold_dates)]

        ax.plot(mkt_percentages, marker='s', color='k', label='Market')
        ax.plot(pred_percentages, marker='p', color='g', label='Prediction')
        ax.set(title='Predicition vs. Market for {} Day Forecasting'.format(forecast_out),
               xlabel='Trade Cycle', ylabel='Percent Growth')
        ax.legend(loc='best')
        #plt.show()
        path = "/home/dr/Projects/multi_model_stock_forecasting/results/"
        path += "simulation_result_{}_{}_stocks.png".format(forecast_out, len(stocks))
        #plt.savefig(path)
        ticks = ax.get_xticks()
        vals = [int(t) for t in ticks if((t==int(t)) and (t>=0) and (t<len(end_hold_dates)))]
        #vals = [v for idx, v in enumerate(vals) if(idx < len(end_hold_dates))]
        print("TICKS")
        print(vals)
        print("END HOLD DATES")
        print(end_hold_dates)
        ax.set_xticks(vals)
        tick_dates = [end_hold_dates[v] for v in vals]
        ax.set_xticklabels(("{}-{}-{}".format(t.month,t.day,t.year) for t in tick_dates),
                           rotation=30)

        plt.savefig(file_name[:file_name.rfind('/')]+"/final_graph.png")
        #plt.savefig(file_name+".png")
        plt.close()

    # Final write to file
    write_final(file_name, dates, new_money, start_money, first_start, time.time(),
              pred_percentages, mkt_percentages, end_hold_dates)
    return percent


if __name__ == '__main__':
    penny_stocks = ['NUGT', 'REKR', 'INSG', 'UAN', 'NVTR', 'CETX', 'SSKN', 'INSY', 'UAMY',
                    'ZIXI', 'GLUU', 'ARTX', 'PLUG', 'MCI', 'INWK', 'TRPX', 'BIOC', 'OTLK',
                    'SESN', 'BCRX', 'CANF', 'ESTR', 'TYME', 'AAU', 'CHKE', 'CTIC', 'EFOI',
                    'SYN', 'MCEP', 'TGB', 'LGCY', 'YOGA', 'IGLD', 'CLIR', 'AMPE', 'AFH',
                    'HEBT', 'PETZ', 'XRF', 'IGC', 'UPL', 'AKG', 'HLTH', 'ENG', 'MYT', 'TWMC',
                    'OPGN', 'XBIO', 'SFUN', 'NGD', 'RVLT', 'HK', 'PES', 'APRN', 'OBLN', 'ADOM',
                    'FCEL', 'NAVB', 'SLS', 'DPW', 'PES', 'AVEO', 'NVCN', 'APVO', 'SFUN',
                    'LIFE', 'YUMA', 'FCEL', 'AMR', 'SNSS', 'PIXY', 'HUSA', 'NAKD',
                    'CYTX', 'TGB', 'LGCY', 'IGC', 'ZNGA', 'TOPS', 'TROV', 'SPXCY',
                    'JAGX', 'CEI', 'AKR', 'BPMX', 'MYSZ','GNC', 'CHK', 'BLNK', 'SLNO', 'ZIOP',
                    'AUGR', 'CTL', 'FDNI']
    penny_stocks = np.unique(penny_stocks)
    #penny_stocks = penny_stocks[:20]

    # Regular stocks
    stocks = ["AMZN", "VTI", "VOO", "QQQ", "MSFT", "AAPL", "VYM", "F", "GE", "AMD",
              "ACB", "APHA", "ZNGA", "NFLX", "TSLA", "BABA", "NVDA", "XRX", "SBUX",
              "TWTR", "GOOG", "FB", "FDX", "DIS", "K", "MNST", "SPY", "IEFA", "SPXCY",
              "XLK", "VGT", "RYT", "QTEC", "FDN", "IGV", "HACK", "SKYY", "SOXX",
              "ROBO", "PSJ", "IGV", "XSW", "XITK", "TECL", "IPAY", "XSD", "SMH", "TECS",
              "FNG", "SOXS", "VGT", "KWEB", "PXQ", "SSG", "IGV", "AIQ", "ARKQ", "ARKW",
              "CIBR", "CWEB", "DTEC", "EMQQ", "FDN", 
              "FINX", "FNG", "FTEC", "FTXL", "FXL", "GDAT", "HACK", "IGM", "IGN", "IGV",
              "IPAY", "ITEQ", "IXN", "IYW", "IZRL", "JHMT", "KEMQ", "KWEB", "LOUP",
              "OGIG", "PLAT", "PNQI", "PRNT", "PSCT", "PSI", "PSJ", "PTF", "PXQ", "QTEC",
              "QTUM", "REW", "ROM", "RYT", "SMH", "SOCL", "SOXL", "SSG", "TCLD", "TDIV", "TECL",
              "TECS", "TPAY", "TTTN", "USD", "VGT", "XITK", "XLK", "XNTK", "XSD", "XSW", "XT",
              "XTH", "XWEB", "NWL", "MO", "IVZ", "M", "KIM", "T", "IRM", "MAC", 
              'AUGR', 'CTL', 'FDNI', 'CLOU', 'CQQQ']
    stocks = np.unique(stocks)
    #stocks = stocks[:10]

    # Initial Date must be on day markets were open
    best_percentage = 0
    best_forecast = ''
    verbose = True
    #loss_threshold = 0.02 # Loss threshold to skip buying next cycle
    #buy_threshold = 0.005 # Threshold for Price Ratio. Purchases stocks above this
    #deposits = [False]*7
    deposits = None
    procs = []
    #for i in range(2, 15):
    #for i in range(10, 100, 10):
    for l in [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, -0.005]:
        for b in [0.0, 0.005, 0.01, 0.015, -0.005, -0.001, -0.01]:
    #        if((l==0.0) and (b==0.0)):
    #            continue
    #for l in [0.0]:
    #    for b in [-0.005]:
    #for l in [0.01, 0.005]:
    #    for b in [0.01, 0.005]:
            #print("FORECASTING: {}".format(i))
            p = Process(target=simulate, args=(stocks, 8, 2, 2, 2017, 10, 7, 2019,
                        2000.0, True, deposits, l, b, verbose))
            procs.append(p)
            p.start()

        for p in procs:
            p.join()
            #, 1, 6, 2019)
            #if(percentage > best_percentage):
            #    #print("NEW BEST")
            #    best_percentage = percentage
            #    best_forecast = i
            #if(verbose):
            #    print("PERCENTAGE: {} FOR FORECAST OUT: {}\n\n\n".format(percentage, i))
            
        if(verbose):
            print("\nBEST PERCENTAGE: {} BEST FORECAST: {}".format(best_percentage, best_forecast))

# Variable number of forecast days? Choose best one for each start_date?
