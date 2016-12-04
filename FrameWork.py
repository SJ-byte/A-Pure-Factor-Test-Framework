import pandas as pd
import talib
import datetime as datetime
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import math as math
from sklearn import datasets, linear_model
from scipy.optimize import minimize 

#In order to make the program faster, we can try to calculate the Beta first and write to CSV, and then read the csv in

############################Define our factors####################################### 
def RSIIndividual(stock,end):
    window_length = 14
    start = "{:%Y-%m-%d}".format(datetime.datetime.strptime(end, '%Y-%m-%d') - datetime.timedelta(days=window_length))
    data = get_price(list(stock), start, end_date=end, frequency='1d', fields=None)['OpeningPx']
    close = data
    delta = close.diff()
    delta = delta[1:]
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    roll_up1 = pd.stats.moments.ewma(up, window_length)
    roll_down1 = pd.stats.moments.ewma(down.abs(), window_length)
    RS1 = roll_up1 / roll_down1
    RSI1 = 100.0 - (100.0 / (1.0 + RS1))
    FactorValue = RSI1.iloc[-1]
    FactorValue = (FactorValue - np.mean(FactorValue))/np.std(FactorValue)
    return FactorValue

def Min130Day(stock,enddate): 
    startdate = "{:%Y-%m-%d}".format(datetime.datetime.strptime(enddate, '%Y-%m-%d') - datetime.timedelta(days=130))
    if len(stock) != 0 :
        prices = get_price(list(stock), startdate, end_date=enddate, frequency='1d', fields=None)['OpeningPx']
        returns = np.log(prices/prices.shift(1)).iloc[1:-1]
        MinReturn = returns.min()
        FactorValue = MinReturn
        FactorValue = (FactorValue - np.mean(FactorValue))/np.std(FactorValue)
        return FactorValue
    else:
        return pd.Series(1000, index = ['000001.XSHE'])

    
#Thing is that sometimes the date that we input is not working day, if so, we select the most recent working day's value as the value for this
def EquitySize(stock,enddate):
    date = enddate
    fundamental_df = get_fundamentals(
        query(
            fundamentals.eod_derivative_indicator.market_cap 
        ).filter(
            fundamentals.income_statement.stockcode.in_(stock)
        )
    )
    #print(fundamental_df)
    FactorValue = fundamental_df.T['market_cap']
    FactorValue = (FactorValue - np.mean(FactorValue))/np.std(FactorValue)
    #print('R',FactorValue)
    return FactorValue
    
def EquityOCFP(stock,enddate):
    date = enddate
    fundamental_df = get_fundamentals(
        query(
            fundamentals.financial_indicator.operating_cash_flow_per_share 
        ).filter(
            fundamentals.income_statement.stockcode.in_(stock)
        )
    )
    prices = get_price(list(stock), "{:%Y-%m-%d}".format(datetime.datetime.strptime(date, '%Y-%m-%d') - datetime.timedelta(days=10)), date, frequency='1d', fields=None)['OpeningPx']
    FactorValue = fundamental_df.T['operating_cash_flow_per_share']/prices.iloc[-1]
    FactorValue = (FactorValue - np.mean(FactorValue))/np.std(FactorValue)
    return FactorValue

#Try some new factors
def PricetoLowest(stock,end):
    start = "{:%Y-%m-%d}".format(datetime.datetime.strptime(end, '%Y-%m-%d') - datetime.timedelta(days=260))
    if len(stock) != 0 :
        prices = get_price(list(stock), start, end, frequency='1d', fields=None)['OpeningPx']
        RatiotoLowest = pd.Series()
        for i in list(range(1,len(stock)+1)):
            RatiotoLowest = RatiotoLowest.append(pd.Series((prices.iloc[np.shape(prices)[0]-1][[stock[i-1]]]/min(prices[stock[i-1]]))))
        returnv = (RatiotoLowest)
        FactorValue = returnv
        FactorValue = (FactorValue - np.mean(FactorValue))/np.std(FactorValue)
        return FactorValue
    else:
        return pd.Series(-1000, index = ['000001.XSHE'])

def Volatility(stock,end): 
    start = "{:%Y-%m-%d}".format(datetime.datetime.strptime(end, '%Y-%m-%d') - datetime.timedelta(days=30))
    if len(stock) != 0 :
        #indexprice = get_price('000001.XSHG', startdate, end_date=enddate, frequency='1d', fields=None, adjusted=True)['OpeningPx']
        prices = get_price(list(stock), start, end, frequency='1d', fields=None)['OpeningPx']
        returns = np.log(prices/prices.shift(1)).iloc[1:-1]
        Volatility = pd.Series()
        for i in list(range(1,len(stock)+1)):
            covmat = np.cov(returns[stock[i-1]])
            Volatility = Volatility.set_value(stock[i-1], covmat)
        FactorValue = Volatility
        #print('fv')#,FactorValue.iloc[0:5])
        FactorValue = (FactorValue - np.mean(FactorValue))/np.std(FactorValue)
        return FactorValue
    else:
        return pd.Series(-1000, index = ['000001.XSHE'])

def sharpe(stock,end):
    start = "{:%Y-%m-%d}".format(datetime.datetime.strptime(end, '%Y-%m-%d') - datetime.timedelta(days=360))
    if len(stock) != 0 :
        prices = get_price(list(stock), start, end, frequency='1d', fields=None)['OpeningPx']
        returns = pd.DataFrame(columns=(stock))
        for i in list(range(1,np.shape(prices)[0]-1)):
            returns.loc[i] = ((prices.iloc[i]/prices.iloc[i-1]-1))
        ave = np.mean(returns)
        std = np.std(returns)
        sharpe = (ave/std)*np.sqrt(252)
        returnv = pd.Series(sharpe,index = [stock])
        FactorValue = returnv
        FactorValue = (FactorValue - np.mean(FactorValue))/np.std(FactorValue)
        return FactorValue
    else:
        return pd.Series(-1000, index = ['000001.XSHE'])
        
############################Define functions to get IC and Beta######################
def GetIC(f,*args):
    FactorValue = f(*args)
    stock = args[0]
    date = args[1]
    tempprice = get_price(list(stock), date, "{:%Y-%m-%d}".format(datetime.datetime.strptime(date, '%Y-%m-%d') + datetime.timedelta(days=30)), frequency='1d', fields=None)['OpeningPx']
    tempreturn = np.log(tempprice.iloc[-1]/tempprice.iloc[0])
    DataAll = pd.concat([FactorValue,tempreturn],axis = 1)
    DataAll = DataAll.dropna()
    return np.corrcoef(np.array(DataAll.ix[:,0].T.rank().T),np.array(DataAll.ix[:,1].T.rank().T))[0,1]

##################Get Historical Beta and Residuals to calculate Risk##############
#In order to get daily Beta and Residuals, we can change the enddate from monthly to daily
def GetBeta(f,*args):
    FactorValue = f(*args)
    stock = args[0]
    date = args[1]
    #Get 20 Business day's data
    tempprice = get_price(list(stock), date, "{:%Y-%m-%d}".format(datetime.datetime.strptime(date, '%Y-%m-%d') + datetime.timedelta(days=30)), frequency='1d', fields=None)['OpeningPx']
    tempreturn = np.log(tempprice.iloc[-1]/tempprice.iloc[0])
    #print('FV',FactorValue)
    FactorValue = pd.DataFrame(FactorValue)
    DataAll = pd.concat([FactorValue,tempreturn],axis = 1)
    DataAll = DataAll.dropna()
    DataAll.columns = ['f','p']
    #print('fs',FactorValue.shape)    
    #print('ts',tempreturn.shape)
    #print(DataAll)
    #print(DataAll.shape)
    #print(np.matrix(DataAll.ix[:,0]).shape)
    #print(np.matrix(DataAll.ix[:,1]).shape)
    #print('In side GetBeta',DataAll['f'].iloc[0:5])
    regr = linear_model.LinearRegression()
    regr.fit(np.transpose(np.matrix(DataAll['f'])), np.transpose(np.matrix(DataAll['p'])))
    return regr.coef_

def GetResiduals(stock,enddate,Xinput):
    X = Xinput
    dim = X.shape
    length = dim[0]
    nfactors = dim[1]
    date = enddate
    tempprice = get_price(list(stock), date, "{:%Y-%m-%d}".format(datetime.datetime.strptime(date, '%Y-%m-%d') + datetime.timedelta(days=30)), frequency='1d', fields=None)['OpeningPx']
    y = np.log(tempprice.iloc[-1]/tempprice.iloc[0])
    DataAll = pd.concat([X,y],axis = 1)
    DataAll = DataAll.dropna()
    DataAll.columns = list(range(0,nfactors+1))
    #print(DataAll.iloc[0:5])
    #print(np.matrix(DataAll.ix[:,0:nfactors]))
    #print(np.transpose(np.matrix(DataAll.ix[:,nfactors])))
    regr = linear_model.LinearRegression()
    regr.fit(np.matrix(DataAll.ix[:,0:nfactors]), np.transpose(np.matrix(DataAll.ix[:,nfactors])))
    residuals = regr.predict(np.matrix(DataAll.ix[:,0:nfactors])) - np.transpose(np.matrix(DataAll.ix[:,nfactors]))
    residuals = pd.DataFrame(data = residuals, index = np.transpose(np.matrix(DataAll.index.values)))
    residuals.index = DataAll.index.values
    residuals.columns = [enddate]
    return residuals

def GetBetaCovEsti(i,lam,h,BetaAll):
    dim = BetaAll.T.shape
    #print(dim)
    length = dim[0]
    width = dim[1]
    tempAll = BetaAll.T.ix[0:length - i,:]
    #print(tempAll)
    #print(tempAll.shape)
    #print(type(tempAll))
    #print(tempAll.iloc[0])
    tempAll = tempAll.T
    CovEsti = pd.DataFrame(np.random.randn(width, width))
    for i in list(range(0,width)):
        for j in list(range(0,width)):
            #print(i)
            #print(j)
            #print(tempAll.ix[:,j])
            tempSeries1 = tempAll.iloc[i].T
            tempSeries2 = tempAll.iloc[j].T
            tempresult = EWMA(tempSeries1,tempSeries2,lam,h)
            CovEsti.ix[i,j] = tempresult
    #print(CovEsti)
    return CovEsti

def GetIndRiskCovEsti(i,lam,h,ResidualAll):
    dim = ResidualAll.T.shape
    length = dim[0]
    width = dim[1]
    tempAll = ResidualAll.T.iloc[0:length - i]
    CovEsti = pd.DataFrame(np.zeros((width, width)),columns = ResidualAll.T.columns.values)
    for i in list(range(1,width+1)):
        tempSeries1 = tempAll.ix[:,i-1]
        tempSeries2 = tempAll.ix[:,i-1]
        tempresult = EWMA(tempSeries1,tempSeries2,lam,h)
        CovEsti.ix[i-1,i-1] = tempresult
    return CovEsti

#######################Some tools that is used in the previous functions#############
#This function is used in the later function
def lamtotal(lam,h):
    total = 0
    for i in list(range(h,-1,-1)):
        total = total + lam ** (i)
    return total

#input as Series, date of series should be increasing
#We also need to deal with isnan here, modify this part later
def EWMA(Series1,Series2,lam,h):
    Sm1 = np.mean(Series1)
    Sm2 = np.mean(Series2)
    l1 = len(Series1)
    l2 = len(Series2)
    total = 0
    for i in list(range(h,0,-1)):
        #print(Series1.iloc[l1-i])
        #if ~np.isnan(Series1.iloc[l1-i])&~np.isnan(Series1.iloc[l1-i]):
        total = total + ((lam ** i) *(Series1.iloc[l1-i] - Sm1)*(Series2.iloc[l1-i] - Sm2))
    total = total/lamtotal(lam,i)
    return total
    
##########################Get the exposure on factors of indices####################
#######For Hedging purpose, get the number of the factors in our input
def NumberofArgs(*args):
    for i in list(range(0,1000)):
        try:
            args[i]
        except:
            break
    return i

############################Get the exposre on a factor########################
def GetIndSinExp(f,*args):
    return f(*args)

#Get the exposure of all factors
#Here the stock should be the compoment of the index that we want to use to hedge
def GetIndAllExp(stock,enddate,*args):
    OutValue = []
    for arg in args[0]:
        temp = arg(stock,enddate)
        OutValue.append(np.mean(temp.values))
    return OutValue

##########################Get the Beta in the history
def GetAllBeta(stock,enddate,*args):
    OutValue = []
    for arg in args[0]:
        OutValue.extend(GetBeta(arg,stock,enddate).tolist()[0])
    return OutValue

########################Get Factor Exposure at a specific date
def GetAllFactorExposure(stock,enddate,*args):  #Output is directly dataframe
    OutValue = []
    flag = 1
    for arg in args[0]:
        if flag == 1:
            OutValue = pd.DataFrame(arg(stock,enddate))
            flag = 0
        else:
            OutValue = pd.concat([OutValue,arg(stock,enddate)],axis = 1)
    OutValue.columns = list(range(1,len(args[0])+1))
    return OutValue

def choose_stocks(context,bar_dict):
    #########################Get the exposure of factors on this stock#######
    end = "{:%Y-%m-%d}".format(get_previous_trading_date(context.now))    
    index = '000016.XSHG'  #this is the index for hedging
    hedgingstocks = index_components(index)  #this is the stocks we use for hedging purpose
    
    MyFactors = [RSIIndividual,Min130Day,EquitySize,EquityOCFP,PricetoLowest,sharpe]
    Exposure = GetIndAllExp(hedgingstocks,end,MyFactors) #this is used for later hedging purpose
    
    AllStock = index_components('000300.XSHG') #this is the stock pool for selection
    stock = AllStock
    #########################Get the list of Beta to calculate new BetaCov##########
    enddatel = []
    for i in list(range(1,10)):
        enddatel.append("{:%Y-%m-%d}".format(context.now - datetime.timedelta(days=(31*i))))
    #Here to change the format, same column belong to a factor, same row belong to a time
    
    #we get the historical beta for all
    AllHisBeta = pd.DataFrame(np.nan * np.ones([len(enddatel),len(MyFactors)]))
    AllHisBeta.index = enddatel
    count = 0
    for enddate in enddatel:
        AllHisBeta.ix[count,:] = GetAllBeta(stock,enddate,MyFactors)
        count = count + 1

    #########################Get All Residuals to calculate ResCov##############
    #Here we only need to change the Xinput can be enough
    ResidualAll = pd.DataFrame()
    for enddate in enddatel:
        Xinput = GetAllFactorExposure(stock,enddate,MyFactors)
        #print(Xinput.iloc[0:5])
        tempresidual = GetResiduals(stock,enddate,Xinput)
        ResidualAll = pd.concat([ResidualAll,tempresidual],axis = 1,join='outer')
    ResidualAll.columns = enddatel

    #########################Get Historical Cov for test purpose################
    #Get the Covariance Matrix of the Residuals
    ResidualCovh = ResidualAll.T.cov()
    #Get the Covariance Matrix of the Factor Earnings
    BetaAll = AllHisBeta
    BetaCovh = BetaAll.cov()

    ############################optimize our portfolio##########################
    lam = 0.5
    h = 5
    endv = "{:%Y-%m-%d}".format(get_previous_trading_date(context.now))
    endb = "{:%Y-%m-%d}".format(context.now - datetime.timedelta(days=31))
    
    #If we use the EWMA Method
    BetaCov = GetBetaCovEsti(0,lam,h,BetaAll)
    IndCov = GetIndRiskCovEsti(0,lam,h,ResidualAll) 
    
    #If we use the historical Method
    BetaCov = BetaCovh
    IndCov = ResidualCovh
    
    print(IndCov)
    print(IndCov.shape)
    stock = IndCov.columns

    StockNumber = IndCov.shape[1]
    #print(StockNumber)
    w = np.full((1, StockNumber), 1/StockNumber)

    f = np.asarray(GetAllBeta(stock,endb,MyFactors))
    x = GetAllFactorExposure(stock,endv,MyFactors)
    x = x.fillna(0).as_matrix()
    StockReturn = np.matmul(x,f)
    #print(x.shape)
    FactorLoading = x

    #Tool Function
    def MyVar(w,BetaCov,ResidualCov,FactorLoading):
        part1 = np.matmul(FactorLoading,BetaCov)
        part2 = np.matmul(part1,FactorLoading.T) + ResidualCov
        part3 = np.matmul(w,part2)
        pos = ~np.isnan(part3)
        part3 = part3[pos]
        w = w[pos]
        part4 = np.matmul(part3,w.T)
        return math.sqrt(part4)
    
    #objective function:
    def RiskAdjReturn(w,StockReturn,BetaCov,IndCov,FactorLoading):
        ExpReturn = np.matmul(w,StockReturn)
        Std = MyVar(w,BetaCov,IndCov,FactorLoading)
        return -(ExpReturn-0.5*(Std**2))
    
    #Constrains for sum of weight equeals to 1 and Factor Loading of a certain factor as 0
    cons = ({'type': 'eq',
             'fun' : lambda w: np.array(sum(w.T) - 1)},
           {'type': 'eq',
             'fun' : lambda w: np.array(np.matmul(w,FactorLoading[:,-1])-Exposure[-1])})
    
    #bands that make sure we don't have weights too extrem
    bnds = ((0, 0.15),) * StockNumber
    
    #Train the optimizer
    res = minimize(RiskAdjReturn, np.full((1, StockNumber), 1/StockNumber), args=(StockReturn,BetaCov,IndCov,FactorLoading,), bounds=bnds,
                   constraints=cons, method='SLSQP', options={'disp': True})
    
    w = res.x
    context.stocks = stock[w>np.percentile(w,95)].values
    context.weight = w[w>np.percentile(w,95)]
    update_universe(context.stocks)

#Get stocks that are currently trading to refine the pool of stock selection
def get_trading_stocks(raw_stocks, bar_dict):
    trading_stocks = []
    for stock in raw_stocks:
        if bar_dict[stock].is_trading:
            trading_stocks.append(stock)
    return trading_stocks
    
def adjust_positions(context, bar_dict):

    for last_stock in context.portfolio.positions:
        if bar_dict[last_stock].is_trading:
            order_target_percent(last_stock,0)
    #firstly we sell out all the stocks in our portfolio to make sure we can get the ideal weight that we need.
    
    to_buy_stocks = context.stocks
    #here in fact we set a limit to how much money we can use
    avail_cash = context.portfolio.cash*context.position_limit
    each_cash = avail_cash/len(to_buy_stocks)
    #logger.info("avail cash is %f, stock num is %d, each stock cash is %f.",avail_cash,len(to_buy_stocks),each_cash )
    for current_stock in to_buy_stocks:
        order_target_value(current_stock, each_cash)    
    portfolio_value = context.portfolio.portfolio_value* context.position_limit 
    order_target_value("510300.XSHG", -(portfolio_value))
    
def init(context):
    context.benchmark = '000300.XSHG'    
    context.stocks = '000300.XSHG'
    context.short_selling_allowed = True
    context.stoplossmultipler= 0.85 #止损 乘数 
    context.stoppofitmultipler= 1000.8 #止盈 乘数
    context.Traded = 0
    context.position_limit  = 0.9
    context.position_num = 10
    context.countdate = 0
    context.holdingperiod = 15
    context.weight = []
    scheduler.run_monthly(choose_stocks, monthday=1, time_rule='before_trading')
    #scheduler.run_monthly(adjust_positions, monthday=1)
    #scheduler.run_weekly(adjust_future, weekday=1)
    
    update_universe([context.stocks])
    
def before_trading(context, bar_dict):
    pass
    #context.countdate = context.countdate + 1
    #if context.countdate%context.holdingperiod == 1:
    #    choose_stocks(context, bar_dict)

def handle_bar(context, bar_dict):
    #if context.Traded == 1:
        #stoploss(context,bar_dict)
    #if context.countdate%context.holdingperiod == 1:
    context.average_percent = context.weight
        #print(context.stocks[0:5])
        #print(len(context.stocks))
        #print(len(context.weight))
    for i in list(range(0,len(context.weight))):
        order_target_percent(context.stocks[i], context.average_percent[i])
    context.Traded = 1    

