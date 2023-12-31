#!/usr/bin/env python
# coding: utf-8

# In[3]:


# import packages 
import numpy as np
import pandas as pd
import ipywidgets as widgets
from IPython.display import display


# In[4]:


# functions

def run_cppi(risky_r, safe_r = None, m = 3, start = 1000, floor = 0.8, riskfree_rate = 0.03, drawdown = None):
    """
    Run a backtest of the CPPI strategy, given a set of returns for the risky asset
    Returns a dictionary containing: Asset Value History, Risk Budget History, Risky Weight History
    """
    # Set up the CPPI parameters
    dates = risky_r.index
    n_steps = len(dates)
    account_value = start
    floor_value = start * floor
    peak = start
    
    if isinstance(risky_r, pd.Series):
        risky_r = pd.DataFrame(risky_r, columns = ['R'])
    
    if safe_r is None:
        safe_r = pd.DataFrame().reindex_like(risky_r)
        safe_r.values[:] = riskfree_rate/12            # A fast way to set all values to a number
    
    # Set up some DataFrames for saving intermediate values
    account_history = pd.DataFrame().reindex_like(risky_r)
    risky_w_history = pd.DataFrame().reindex_like(risky_r)
    cushion_history = pd.DataFrame().reindex_like(risky_r)

    for i in range(n_steps):
        if drawdown is not None:
            peak = np.maximum(peak, account_value)
            floor_value = peak * (1 - drawdown)
        cushion = (account_value - floor_value)/account_value
        risky_w = m*cushion
        risky_w = np.maximum(0, risky_w)
        risky_w = np.minimum(1, risky_w)
        safe_w = 1 - risky_w
        risky_alloc = account_value * risky_w
        safe_alloc = account_value * safe_w
        # recompute the new account value at the end of this step
        account_value = risky_alloc * (1 + risky_r.iloc[i]) + safe_alloc * (1 + safe_r.iloc[i])
        # Save the histories for analysis and plotting
        cushion_history.iloc[i] = cushion
        account_history.iloc[i] = account_value
        risky_w_history.iloc[i] = risky_w
        
        risky_wealth = start * (1 + risky_r).cumprod()
        backtest_result = {
            "Wealth": account_history,
            "Risky Wealth": risky_wealth,
            "Risk Budget": cushion_history,
            "m": m,
            "start": start,
            "floor": floor,
            "risky_r": risky_r,
            "safe_r": safe_r
        }
    return backtest_result


# In[5]:


# Data Cleaning & Preparing - 4 major industries & risk-free asset (10 years t-bill)
r_port = pd.read_csv('CPPI Port.csv',
                  header = 0, index_col = 0, parse_dates = True)
rets = r_port['2010':][['Tech ', 'Consumer', 'Finance', 'Healthcare']]
rets.columns = ['Tech', 'Consumer', 'Finance', 'Healthcare']
rets.index = pd.to_datetime(rets.index, format = "%Y%m").to_period('M')
risky_r = rets.iloc[0:203]

safe_r = pd.DataFrame().reindex_like(risky_r)
safe_r[:] = 0.03/12
#safe_r = r_port['2010':][['TNX', 'TNX', 'TNX', 'TNX']]
#safe_r.index = pd.to_datetime(safe_r.index, format = "%Y%m").to_period('M')
#safe_r = safe_r.iloc[0:203]
risky_r


# In[6]:


dates = risky_r.index
n_steps = len(dates)
start = 1000
floor = 0.8
account_value = start # initial value
floor_value = start * floor
m = 3
account_history = pd.DataFrame().reindex_like(risky_r)
cushion_history = pd.DataFrame().reindex_like(risky_r)
risky_w_history = pd.DataFrame().reindex_like(risky_r)

account_history.shape
for i in range(n_steps):
    cushion = (account_value - floor_value)/account_value
    risky_w = m*cushion
    # The weights of the porfolio in cppi strategy should be between 0 and 1 
    risky_w = np.minimum(1, risky_w)
    risky_w = np.maximum(0, risky_w)
    safe_w = 1 - risky_w
    risky_alloc = account_value * risky_w
    safe_alloc = account_value * safe_w
    # Update the account value for this time step
    account_value = risky_alloc * (1 + risky_r.iloc[i]) + safe_alloc * (1 + safe_r.iloc[i])
    # Save the values so I can look at the history and plot it etc
    cushion_history.iloc[i] = cushion
    risky_w_history.iloc[i] = risky_w
    account_history.iloc[i] = account_value


# In[57]:


# Wealth Change over time without applying CPPI
risky_wealth = start*(1 + risky_r).cumprod()
risky_wealth.plot()


# In[7]:


# Wealth using CPPI strategy
w_cppi_tech = account_history['Tech'].plot()
risky_wealth['Tech'].plot(ax = w_cppi_tech, style = 'k')
w_cppi_tech.axhline(y = floor_value, color = 'r', linestyle = '--')


# In[64]:


risky_w_history.plot()


# In[73]:


w_cppi_finance = account_history['Finance'].plot()
risky_wealth['Finance'].plot(ax = w_cppi_finance, style = 'k')
w_cppi_finance.axhline(y = floor_value, color = 'r', linestyle = '--')


# In[77]:


# Constant Floor wasn't effective, Let's add in drawdown restraints
btr = run_cppi(risky_r, safe_r, drawdown = 0.25)
btr['Wealth'].plot(figsize = (12, 6))


# In[79]:


btr['Risky Wealth'].plot(figsize = (12, 6))


# In[ ]:


ax = btr['Wealth']['Healthcare'].plot(figsize = (12,6))
btr['Risky Wealth']['Healthcare'].plot(ax = ax, style = 'k')


# ## Interactive CPPI Simulation - Monte Carlo

# In[8]:


def gbm (n_years = 10, n_scenarios = 1000, mu = 0.07, sigma = 0.15, steps_per_year = 12, s_0 = 100, prices = True):
    ''' 
    Evolution of the stock price using Geometric Brownian Motion Model trajectories, such as for Stock Prices through Monte Carlo
    '''
    # Derive per-step model parameters from user specifications
    dt = 1/n_years
    n_steps = int(n_years*steps_per_year)
    
    # to prices
    if prices == True:
        rets_plus_1 = np.random.normal(loc = (1 + mu*dt), scale = (sigma*np.sqrt(dt)), size = (n_steps, n_scenarios))
        rets_plus_1[0] = 1
        prices = s_0*pd.DataFrame(rets_plus_1).cumprod()
        return prices
    else: 
        rets = np.random.normal(loc = (mu*dt), scale = (sigma*np.sqrt(dt)), size = (n_steps, n_scenarios))
        rets[0] = 0
        return rets
    
    


# ### Random Walk Generation
# $$ \frac{S_{t + dt} - S_{t}}{S_{t}} ={{\mu}dt + {\sigma}\sqrt{dt}\xi_{t}}$$

# In[9]:


def show_cppi(n_scenarios = 50, mu = 0.07, sigma = 0.3, m = 3, floor = 0., riskfree_rate = 0.03, y_max = 100):
    '''
    Plot the results of a Monte Carlo Simulation of CPPI
    '''
    start = 100
    sim_rets = gbm(n_scenarios = n_scenarios, mu = mu, sigma = sigma, prices = False, steps_per_year = 12)
    risky_r = pd.DataFrame(sim_rets)
    # run the 'back'-test
    btr = run_cppi(risky_r = pd.DataFrame(risky_r), riskfree_rate = riskfree_rate, m = m, start = start, floor = floor)
    wealth = btr['Wealth']
    y_max = wealth.values.max()*y_max/100
    ax = wealth.plot(legend = False, alpha = 0.3, color = 'indianred', figsize = (12, 6))
    ax.axhline(y = start, ls = ':', color = 'black')
    ax.axhline(y = start * floor, ls = '--', color = 'red')
    ax.set_ylim(top = y_max)
    
cppi_controls = widgets.interactive(show_cppi,
                                    n_scenarios = widgets.IntSlider(min = 1, max = 1000, step = 5, value = 50),
                                    mu = (0., +.2, .01),
                                    sigma = (0, 0.3, 0.05),
                                    floor = (0, 2, 0.1),
                                    m = (1, 5, 0.5),
                                    riskfree_rate = (0, 0.05, 0.01),
                                    y_max = widgets.IntSlider(min = 0, max = 100, step = 1, value = 100,
                                                             description = 'Zoom Y Axis')
)
display(cppi_controls)
                           


# In[11]:


import matplotlib.pyplot as plt


# In[12]:


def show_cppi(n_scenarios = 50, mu = 0.07, sigma = 0.15, m = 3, floor = 0., riskfree_rate = 0.03, y_max = 100):
    '''
    Plot the results of a Monte Carlo Simulation of CPPI
    '''
    start = 100
    sim_rets = gbm(n_scenarios = n_scenarios, mu = mu, sigma = sigma, prices = False, steps_per_year = 12)
    risky_r = pd.DataFrame(sim_rets)
    # run the back test
    btr = run_cppi(risky_r = pd.DataFrame(risky_r), riskfree_rate = riskfree_rate, start = start, floor = floor)
    wealth = btr['Wealth']
    
    # Calculate terminal wealth stats
    y_max = wealth.values.max()*y_max/100
    terminal_wealth = wealth.iloc[-1]
    
    tw_mean = terminal_wealth.mean()
    tw_median = terminal_wealth.median()
    failure_mask = np.less(terminal_wealth, start*floor)
    n_failures = failure_mask.sum()
    p_fails = n_failures/n_scenarios
    
    e_shortfall = np.dot(terminal_wealth - start*floor, failure_mask)/n_failures if n_failures > 0 else 0.0
    
    # Plot!
    fig, (wealth_ax, hist_ax) = plt.subplots(nrows = 1, ncols= 2, sharey = True, gridspec_kw = {'width_ratios':[3,2]}, figsize = (24, 9))
    plt.subplots_adjust(wspace = 0.0)
    
    wealth.plot(ax = wealth_ax, legend = False, alpha = 0.3, color = 'indianred')
    wealth_ax.axhline(y = start, ls= '--', color = 'b')
    wealth_ax.axhline(y = start*floor, ls = '--', color = 'r')
    wealth_ax.set_ylim(top = y_max)
    
    terminal_wealth.plot.hist(ax = hist_ax, bins = 50, ec = 'w', fc = 'indianred', orientation = 'horizontal')
    hist_ax.axhline(y = start, ls = ':', color = 'black')
    hist_ax.axhline(y = tw_mean, ls = ':', color = 'blue')
    hist_ax.axhline(y = tw_median, ls = ':', color = 'purple')
    hist_ax.annotate(f"Mean: ${int(tw_mean)}", xy = (0.7, 0.9), xycoords = 'axes fraction', fontsize = 24)
    hist_ax.annotate(f"Median: ${int(tw_median)}", xy = (0.7, 0.85), xycoords = 'axes fraction', fontsize = 24)
    if (floor > 0.01):
        hist_ax.axhline(y = start*floor, ls = '--', color = 'red', linewidth = 3)
        hist_ax.annotate(f"Violations: {n_failures} ({p_fails*100:2.2f}%)\nE(shortfall) = ${e_shortfall:2.2f}", xy = (0.7, 0.7), xycoords = 'axes fraction', fontsize = 24)
    
cppi_controls = widgets.interactive(show_cppi,
                                    n_scenarios = widgets.IntSlider(min = 1, max = 1000, step = 5, value = 50),
                                    mu = (0., +.2, .01),
                                    sigma = (0, 0.5, 0.05),
                                    floor = (0, 2, 0.1),
                                    m = (1, 5, 0.5),
                                    riskfree_rate = (0, 0.05, 0.01),
                                    steps_per_year = widgets.IntSlider(min = 1, max = 12, step = 1, value = 12,
                                                             description = 'Rebals/Year'),                                   
                                    y_max = widgets.IntSlider(min = 0, max = 100, step = 1, value = 100,
                                                             description = 'Zoom Y Axis')
)
display(cppi_controls)


# In[ ]:





# In[ ]:





# In[ ]:




