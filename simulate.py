import datetime
import numpy as np
import model
import pandas as pd

from datetime import datetime as dt
from model import test_ml, buy_ml, buy_ml_vol
import pandas_datareader.data as web
import sys


def progress_bar(current_num, total_num):
    start = "[" + "="*int(current_num/total_num*10)
    end = " "*(11-len(start)) + "]"
    #print(len(start))
    sys.stdout.write(start + end + ' %.2f%% \r' % (current_num/total_num*100))
    sys.stdout.flush()
    if(current_num == total_num-1):
        print("[==========] 100.00%")
    #print(start, end, '\r')


def holiday(date):
    '''
      Returns True if date is a holiday according to NYSE, else False
    '''
    holidays = [dt(2019,1,1), dt(2019,1,21), dt(2019,2,18), dt(2019,4,19), dt(2019,5,27),
                dt(2019,7,4), dt(2019,9,2), dt(2019,11,28), dt(2019,12,25), 
                dt(2018,1,1), dt(2018,1,15), dt(2018,2,19), dt(2019,3,30), dt(2018,5,28),
                dt(2018,7,4), dt(2019,9,3), dt(2018,11,22), dt(2018,12,25),
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
    for i in range(2*forecast_out):
        trial_year = year+1 if(trial_day+1 > 31 and trial_month==12) else trial_year
        trial_month = 1 if(trial_day+1 > days[trial_month-1] and trial_month==12) \
                      else trial_month+1 if(trial_day+1 > days[trial_month-1]) \
                      else trial_month
        trial_day = 2 if(trial_day+1 > days[trial_month-1] and trial_month==12) \
                      else trial_day+1 if(trial_day+1 <= days[month-1]) else \
                      trial_day+1-days[month-1]

        trial_date = datetime.datetime(trial_year, trial_month, trial_day)
        if(trial_date.weekday()<5 and not(holiday(trial_date))):
            trading_days += 1
            if(trading_days == forecast_out):
                pred_year = trial_year
                pred_month = trial_month
                pred_day = trial_day
                break

    return dt(pred_year, pred_month, pred_day)


def simulate(stocks, forecast_out=7, start_day=None, start_month=None, start_year=None,
             end_day=None, end_month=None, end_year=None):

    print("STOCKS BEING LOOKED AT: {}\n".format(stocks))

    # Make start date if none given
    if(start_day==None or start_month==None or start_year==None):
        print("DEFAULT START DATE")
        start_day = 2
        start_month = 1
        start_year = 2017

    start_date = dt(start_year, start_month, start_day)

    if(end_day==None or end_month==None or start_year==None):
        print("DEFAULT END DATE OF TODAY")
        end_date = dt.now()
    else:
        end_date = dt(end_year, end_month, end_day)

    # To keep track of total earnings
    total_spent = 0
    total_sold = 0
    end_hold = end_hold_date(forecast_out, start_day, start_month, start_year)
    print(end_hold)
    while(end_hold.date() < end_date.date()):

        print("START DATE: {}".format(start_date))
        print("END HOLD DATE: {}".format(end_hold))
        
        # To keep track of data
        stock_corr_averages = {s: 0 for s in stocks}
        correlations = np.zeros((len(stocks), 10))
        best = {}
        cor = []

        # Test each stock, hold on to relevant data
        print("\nTRAINING")
        for idx, s in enumerate(stocks):
            vals, good_combs = test_ml(s, forecast_out, year=start_date.year,
                            month=start_date.month, day=start_date.day)
            cor.append((s, vals))
            correlations[idx] = vals

            best = model.add_combs(best, good_combs)
            progress_bar(idx, len(stocks))

        correlations = correlations[~np.isnan(correlations).any(axis=1)]
        #print("INDIVIDUAL CORRELATION MEAN")
        #print(np.mean(correlations, axis=0))
        for c in cor:
            stock_corr_averages[c[0]] += np.mean(c[1])
        
        #print(stock_corr_averages)
        d = {k:v for k,v in filter(lambda t: t[1]>0.15, stock_corr_averages.items())}
        good_stocks = list(d.keys())
        #print("GOOD STOCKS: {}".format(good_stocks))

        # Forecast stocks
        stock_pred = pd.DataFrame(columns=['Stock', 'Price Slope', 'Volume Slope', 'Good Buy',
                                           'Good Put'])
        print("FORECASTING")
        for idx, s in enumerate(good_stocks):
            price_slope = buy_ml(s, forecast_out, month=start_date.month, day=start_date.day,
                                 year=start_date.year)
            vol_slope = buy_ml_vol(s, forecast_out, month=start_date.month, day=start_date.day,
                                 year=start_date.year)
            stock_pred = stock_pred.append({"Stock":s, "Price Slope":price_slope,
                             "Volume Slope":vol_slope}, ignore_index=True)
            progress_bar(idx, len(good_stocks))

        stock_pred['Good Buy'] = (stock_pred['Price Slope']>0) & (stock_pred['Volume Slope']>0)
        stock_pred['Good Put'] = (stock_pred['Price Slope']<0) & (stock_pred['Volume Slope']>0)
        print("\nGood Buys:")
        print(stock_pred[stock_pred['Good Buy']])
        print("\n")
        print(stock_pred[stock_pred['Good Buy']]['Stock'].values)
        buy_stocks = stock_pred[stock_pred['Good Buy']]['Stock'].values
        for bs in buy_stocks:
            #print("Before hold")
            bs_dat = web.DataReader(bs, 'yahoo', start_date, dt.now())['Close']
            total_spent += bs_dat.values[0]
            #print("BEFORE HOLD: {}".format(bs_dat))
            hold_dat = web.DataReader(bs, 'yahoo', end_hold, dt.now())['Close']
            #print("AFTER HOLD: {}".format(hold_dat))
            total_sold += hold_dat.values[0]
            money = hold_dat.values[0] - bs_dat.values[0]
            print("MONEY MADE FROM PURCHASE OF {} IS: {} WAS {}".format(bs, money,
                  "GOOD" if money>0 else "BAD"))
                   
            
        start_date = end_hold
        end_hold = end_hold_date(forecast_out, end_hold.day, end_hold.month, end_hold.year)
        #for bs in buy_stocks:
        #    print("After Hold")
        #    bs_dat = web.DataReader(bs, 'yahoo', end_hold, end_hold)['Close']
        #    print(bs_dat)

        #exit(1)

    print("TOTAL SPEND: {}\nTOTAL SOLD: {}\nDIFFERENCE: {}\nPERCENTAGE: {}".format(
          total_spent, total_sold, total_sold-total_spent, 100*(total_sold/total_spent-1)))


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
                    'JAGX', 'CEI', 'AKR', 'BPMX', 'MYSZ','GNC', 'CHK', 'BLNK', 'SLNO', 'ZIOP']
    penny_stocks = ['SPXCY']
    #penny_stocks = np.unique(penny_stocks)
    #penny_stocks = penny_stocks[:15]

    # Initial Date must be on day markets were open
    # ISSUES WHEN YOU HAVE START OR END ON A DAY MARKETS AREN'T OPEN
    simulate(penny_stocks, 5, 3, 1, 2019)#, 1, 6, 2019)
