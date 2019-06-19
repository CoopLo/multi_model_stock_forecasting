import numpy as np
from datetime import datetime
import smtplib
import time
from selenium import webdriver

from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, svm
from sklearn.model_selection import cross_validate

from iexfinance.stocks import Stock, get_historical_data


def get_stocks(n):
    driver = webdriver.Chrome('/usr/bin/google-chrome')
    url="https://finance.yahoo.com/screener/predefined/aggressive_small_caps?offset=0&count=202"
    driver.get(url)

if __name__ == '__main__':
    #get_stocks(2)
    pass
