"The Interaction and Literature Exploration of the U.S. Philadelphia Semiconductor Market, Nasdaq Index, and TSMC"
# import packages
import numpy as np
import pandas as pd
import math
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import coint_johansen

# 資料收集 費城半導體 ^SOX, 那斯達克指數 ^IXIC, 台積電 2330.TW
# 1:疫情前 2:疫情後
start_date1 = '2016-08-01'
end_date1 = '2019-12-31'
start_date2 = '2020-01-01'
end_date2 = '2023-05-30'

pSOX1 = yf.download('^SOX', start = start_date1, end = end_date1)['Adj Close'].dropna()
pSOX2 = yf.download('^SOX', start = start_date2, end = end_date2)['Adj Close'].dropna()
pIXIC1 = yf.download('^IXIC', start = start_date1, end = end_date1)['Adj Close'].dropna()
pIXIC2 = yf.download('^IXIC', start = start_date2, end = end_date2)['Adj Close'].dropna()
pTSMC1 = yf.download('2330.TW', start = start_date1, end = end_date1)['Adj Close'].dropna()
pTSMC2 = yf.download('2330.TW', start = start_date2, end = end_date2)['Adj Close'].dropna()

# 整合所有歷史股價成一個DataFrame: Dataset1 是疫情前, Dataset2 是疫情後
Dataset1 = pd.DataFrame(
    {
        'pSOX1':pSOX1,
        'pIXIC1': pIXIC1,
        'pTSMC1': pTSMC1,
    }).dropna()
Dataset2 = pd.DataFrame(
    {
        'pSOX2':pSOX2,
        'pIXIC2': pIXIC2,
        'pTSMC2': pTSMC2,
    }).dropna()

print(Dataset1.head())
print(Dataset2.head())

# 研究目標: 欲求台積電股價與費城半導體、那斯達克的關係
'''
確保時間序列的數據平穩(Stationary),不平穩可能產生的後果
1. 誤導性趨勢分析： 如果時間序列數據不平稳，可能會出現錯誤的趨勢分析结果。例如，在不平穩的序列上進行的統計檢驗可能不具有可靠性，因為結果可能會受到序列的變異性影響。
2. 模型不穩定： 如果在不平穩的序列上建立模型，模型可能會無法捕捉到真實的趨勢和關係。這可能導致預測不準確，因為模型無法反映出序列的真實行為。
3. 自相關性： 不平穩的序列可能會表現出強烈的自相關性，即過去時間點的值對未來值有影響。這可能使得分析和預測變得困難，因為序列的變化模式不是固定的。
4. 虛假相關性： 不平穩的序列可能會產生虛假的相關性，即兩個變數之間的相關性是由於共同的趨勢或變化而不是真正的因果關係。這可能導致錯誤的分析和錯誤的決策。
5. 無效的預測： 在不平穩的序列上進行預測可能會產生不穩定和不可靠的結果。預測模型無法捕捉到序列的真實變化，因此無法提供準確的預測結果。
'''
# 檢驗時間序列是否平穩 --> 單位根檢驗 Augmented Dickey-Fuller Test(ADF)
def adf_test(timeseries):
    result = adfuller(timeseries, autolag = 'AIC')
    # 視情況列印出來
    '''
    print("ADF Test Results:")
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print('Critical Values:')
    '''
    '''
    for key, value in result[4].items():
        print(f'   {key}: {value}')
    '''
    # 如果p值小於0.05就是平穩,回傳1; 若大於0.05為不平穩, 回傳0
    if result[1] <= 0.05:
        # print("Reject the null hypothesis (data is stationary)")
        return 1
    else:
        # print("Fail to reject the null hypothesis (data is non-stationary)")
        return 0

adftest_result1 = []
for i in range(3):
    adftest_result1.append(adf_test(Dataset1.iloc[:,i]))

adftest_result2 = []
for i in range(3):
    adftest_result2.append(adf_test(Dataset2.iloc[:,i]))

print(adftest_result1, "疫情前3組時間序列數據p值皆>0.05, 因此回傳0 -> 不平穩")
print(adftest_result2, "疫情後3組時間序列數據p值皆>0.05, 因此回傳0 -> 不平穩")

# 將不平穩數據轉換為平穩 差分法:若數據為daily,報酬率為diff(log(price))
rDataset1 = pd.DataFrame(
    {
        'rSOX1':np.log(pSOX1).diff(),
        'rIXIC1': np.log(pIXIC1).diff(),
        'rTSMC1': np.log(pTSMC1).diff(),
    }).dropna()

rDataset2 = pd.DataFrame(
    {
        'rSOX2':np.log(pSOX2).diff(),
        'rIXIC2': np.log(pIXIC2).diff(),
        'rTSMC2': np.log(pTSMC2).diff(),
    }).dropna()

R_adftest_result1 = []
for i in range(3):
    R_adftest_result1.append(adf_test(rDataset1.iloc[:,i]))

R_adftest_result2 = []
for i in range(3):
    R_adftest_result2.append(adf_test(rDataset2.iloc[:,i]))

print(R_adftest_result1, "疫情前3組時間序列數據p值皆<0.05, 因此回傳1 -> 平穩")
print(R_adftest_result2, "疫情後3組時間序列數據p值皆<0.05, 因此回傳1 -> 平穩")

# 成功用差分法將時間序列數據從不平穩變為平穩，往後分析與模型皆使用此平穩數據進行
    
# 研究目標: 欲求台積電股價與費城半導體、那斯達克的關係
'''
確保時間序列的數據平穩(Stationary),不平穩可能產生的後果
1. 誤導性趨勢分析： 如果時間序列數據不平稳，可能會出現錯誤的趨勢分析结果。例如，在不平穩的序列上進行的統計檢驗可能不具有可靠性，因為結果可能會受到序列的變異性影響。
2. 模型不穩定： 如果在不平穩的序列上建立模型，模型可能會無法捕捉到真實的趨勢和關係。這可能導致預測不準確，因為模型無法反映出序列的真實行為。
3. 自相關性： 不平穩的序列可能會表現出強烈的自相關性，即過去時間點的值對未來值有影響。這可能使得分析和預測變得困難，因為序列的變化模式不是固定的。
4. 虛假相關性： 不平穩的序列可能會產生虛假的相關性，即兩個變數之間的相關性是由於共同的趨勢或變化而不是真正的因果關係。這可能導致錯誤的分析和錯誤的決策。
5. 無效的預測： 在不平穩的序列上進行預測可能會產生不穩定和不可靠的結果。預測模型無法捕捉到序列的真實變化，因此無法提供準確的預測結果。
'''
# 檢驗時間序列是否平穩 --> 單位根檢驗 Augmented Dickey-Fuller Test(ADF)
def adf_test(timeseries):
    result = adfuller(timeseries, autolag = 'AIC')
    # 視情況列印出來
    '''
    print("ADF Test Results:")
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print('Critical Values:')
    '''
    '''
    for key, value in result[4].items():
        print(f'   {key}: {value}')
    '''
    # 如果p值小於0.05就是平穩,回傳1; 若大於0.05為不平穩, 回傳0
    if result[1] <= 0.05:
        # print("Reject the null hypothesis (data is stationary)")
        return 1
    else:
        # print("Fail to reject the null hypothesis (data is non-stationary)")
        return 0

adftest_result1 = []
for i in range(3):
    adftest_result1.append(adf_test(Dataset1.iloc[:,i]))

adftest_result2 = []
for i in range(3):
    adftest_result2.append(adf_test(Dataset2.iloc[:,i]))

print(adftest_result1, "疫情前3組時間序列數據p值皆>0.05, 因此回傳0 -> 不平穩")
print(adftest_result2, "疫情後3組時間序列數據p值皆>0.05, 因此回傳0 -> 不平穩")

# 將不平穩數據轉換為平穩 差分法:若數據為daily,報酬率為diff(log(price))
rDataset1 = pd.DataFrame(
    {
        'rSOX1':np.log(pSOX1).diff(),
        'rIXIC1': np.log(pIXIC1).diff(),
        'rTSMC1': np.log(pTSMC1).diff(),
    }).dropna()

rDataset2 = pd.DataFrame(
    {
        'rSOX2':np.log(pSOX2).diff(),
        'rIXIC2': np.log(pIXIC2).diff(),
        'rTSMC2': np.log(pTSMC2).diff(),
    }).dropna()

R_adftest_result1 = []
for i in range(3):
    R_adftest_result1.append(adf_test(rDataset1.iloc[:,i]))

R_adftest_result2 = []
for i in range(3):
    R_adftest_result2.append(adf_test(rDataset2.iloc[:,i]))

print(R_adftest_result1, "疫情前3組時間序列數據p值皆<0.05, 因此回傳1 -> 平穩")
print(R_adftest_result2, "疫情後3組時間序列數據p值皆<0.05, 因此回傳1 -> 平穩")

# 成功用差分法將時間序列數據從不平穩變為平穩，往後分析與模型皆使用此平穩數據進行
    
'''
疫情後費半、那斯達克和台積電的相互關係: test1(費半是否影響台積電) test2(那斯達克是否影響台積電)
'''
test3 = pd.DataFrame({'X': rDataset2['rSOX2'], 'Y': rDataset2['rTSMC2']})

test3_result = grangercausalitytests(test3, max_lag, verbose = True)

test4 = pd.DataFrame({'X': rDataset2['rIXIC2'], 'Y': rDataset2['rTSMC2']})
test4_result = grangercausalitytests(test4, max_lag, verbose = True)

reverse_test3 = pd.DataFrame({'X': rDataset2['rTSMC2'], 'Y': rDataset2['rSOX2']})
reverse_test3_result = grangercausalitytests(reverse_test3, max_lag, verbose = True)

reverse_test4 = pd.DataFrame({'X': rDataset2['rTSMC2'], 'Y': rDataset2['rIXIC2']})
reverse_test4_result = grangercausalitytests(reverse_test4, max_lag, verbose = True)

# 共整合檢定(Cointegration Test)
CI_result1 = coint_johansen(rDataset1, det_order = 0, k_ar_diff = 1)
# Print results
print("Eigenvalues:")
print(CI_result1.eig)
print("\nTrace Statistics:")
print(CI_result1.lr1)
print("\nCritical Values (90%, 95%, 99%):")
print(CI_result1.cvt)
print("\nMaximum Eigenvalue Statistics:")
print(CI_result1.lr2)
print("\nCritical Values (90%, 95%, 99%):")
print(CI_result1.cvm)



























