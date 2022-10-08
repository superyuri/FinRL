#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 1. This tutorial introduces *trade-based metrics* for hyperparameter optimization of FinRL models.
# 2. As the name implies, trade-based metrics are associated with the trade activity that FinRL captures in its actions tables. In general, a trade is represented by an entry in an actions file.
# 2. Such metrics include counts of winning and losing trades, total value of wins and losses and ratio of average market value of wins to losses.
# 1. In this tutorial, we will be tuning hyperparameters for Stable baselines3 models using Optuna.
# 2. The default model hyperparameters may not be adequate for your custom portfolio or custom state-space. Reinforcement learning algorithms are sensitive to hyperparameters, hence tuning is an important step.
# 3. Hyperparamters are tuned based on an objective, which needs to be maximized or minimized. ***In this tutorial, the ratio of average winning to losing trade value is used as the objective.*** This ratio is to be ***maximized***.
# 3. This tutorial incorporates a multi-stock framework based on the 30 stocks (aka tickers) in the DOW JONES Industrial Average. Trade metrics are calculated for each ticker and then aggregated.
# 7.**IMPORTANT**: While the DOW stocks represent a portfolio, portfolio optimization techniques, such as the classic Markowitz mean-variance model, are not applied in this analysis. Other FinRL tutorials and examples demonstrate portfolio optimization. 
# 
# 
# 

# In[ ]:


#Installing FinRL
# Set colab status to trigger installs
clb = False

print(f'Preparing for colab: {clb}')
pkgs = ['FinRL', 'optuna', 'plotly','ipywidgets']
if clb:
    print(f'Installing packages: {pkgs}')


# In[ ]:


# Set Variables
## Fixed
tpm_hist = {}  # record tp metric values for trials
tp_metric = 'avgwl'  # specified trade_param_metric: ratio avg value win/loss
## Settable by User
n_trials = 5  # number of HP optimization runs
total_timesteps = 2000 # per HP optimization run
## Logging callback params
lc_threshold=1e-5
lc_patience=15
lc_trial_number=5


# In[ ]:


#get_ipython().run_cell_magic('capture', '', 'if clb:\n    # installing packages\n    !pip install pyfolio-reloaded  #original pyfolio no longer maintained\n    !pip install git+https://github.com/AI4Finance-LLC/FinRL-Library.git\n    !pip install optuna\n    !pip install -U "ray[rllib]"\n    !pip install plotly\n    !pip install ipywidgets\n    !pip install -U kaleido   # enables saving plots to file')


# In[ ]:


#Importing the libraries
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('Agg')
import datetime
import optuna
from pathlib import Path
#from google.colab import files
#get_ipython().run_line_magic('matplotlib', 'inline')
from finrl import config
from finrl import config_tickers
from optuna.integration import PyTorchLightningPruningCallback
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.env_stock_trading.env_stocktrading_np import StockTradingEnv as StockTradingEnv_numpy
from finrl.agents.stablebaselines3.models import DRLAgent
#from finrl.agents.rllib.models import DRLAgent as DRLAgent_rllib
from finrl.meta.data_processor import DataProcessor
import joblib
from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline
#import ray
from pprint import pprint
import kaleido

import sys
sys.path.append("../FinRL-Library")

import itertools

import torch
if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")
print(f'Torch device: {device}')


# Zipline was developed by Quantopian, which also created pyfolio. The latter is used in FinRL to calculate and display backtest results. Despite the unavailability of zipline, as reported above, pyfolio remains operational. See [here](https://github.com/quantopian/pyfolio/issues/654) for more information.

# In[ ]:


## Connect to GPU for faster processing
#gpu_info = get_ipython().getoutput('nvidia-smi')
#gpu_info = '\n'.join(gpu_info)
#if gpu_info.find('failed') >= 0:
#  print('Not connected to a GPU')
#else:
#  print(gpu_info)


# In[ ]:


import os
if not os.path.exists("./" + config.DATA_SAVE_DIR):
    os.makedirs("./" + config.DATA_SAVE_DIR)
if not os.path.exists("./" + config.TRAINED_MODEL_DIR):
    os.makedirs("./" + config.TRAINED_MODEL_DIR)
if not os.path.exists("./" + config.TENSORBOARD_LOG_DIR):
    os.makedirs("./" + config.TENSORBOARD_LOG_DIR)
if not os.path.exists("./" + config.RESULTS_DIR):
    os.makedirs("./" + config.RESULTS_DIR)


# ## Collecting data and preprocessing
# 
# 1.   Load DOW 30 prices
# 2.   Add technical indicators
# 3.   Create *processed_full*, the final data set for training and testing
# 
# 
# To save time in multiple runs, if the processed_full file is available, it is read from a previously saved csv file.
# 
# 
# 
# 
# 

# In[ ]:


#Custom ticker list dataframe download
#TODO save df to avoid download
path_pf = '/content/ticker_data.csv'
if Path(path_pf).is_file():
  print('Reading ticker data')
  df = pd.read_csv(path_pf)
  
else:
  print('Downloading ticker data')
  ticker_list = config_tickers.DOW_30_TICKER
  df = YahooDownloader(start_date = '2009-01-01',
                     end_date = '2021-10-01',
                     ticker_list = ticker_list).fetch_data()
  df.to_csv('ticker_data.csv')


# In[ ]:


def create_processed_full(processed):
  list_ticker = processed["tic"].unique().tolist()
  list_date = list(pd.date_range(processed['date'].min(),processed['date'].max()).astype(str))
  combination = list(itertools.product(list_date,list_ticker))

  processed_full = pd.DataFrame(combination,columns=["date","tic"]).merge(processed,on=["date","tic"],how="left")
  processed_full = processed_full[processed_full['date'].isin(processed['date'])]
  processed_full = processed_full.sort_values(['date','tic'])

  processed_full = processed_full.fillna(0)
  processed_full.sort_values(['date','tic'],ignore_index=True).head(5)

  processed_full.to_csv('processed_full.csv')
  return processed_full


# In[ ]:


#You can add technical indicators and turbulence factor to dataframe
#Just set the use_technical_indicator=True, use_vix=True and use_turbulence=True
def create_techind():
  fe = FeatureEngineer(
                    use_technical_indicator=True,
                    tech_indicator_list = config.INDICATORS,
                    use_vix=True,
                    use_turbulence=True,
                    user_defined_feature = False)

  processed = fe.preprocess_data(df)
  return processed


# In[ ]:


#Load price and technical indicator data from file if available

path_pf = '/content/processed_full.csv'
if Path(path_pf).is_file():
  print('Reading processed_full data')
  processed_full = pd.read_csv(path_pf)

else:
  print('Creating processed_full file')
  processed=create_techind()
  processed_full=create_processed_full(processed)


# In[ ]:


train = data_split(processed_full, '2009-01-01','2020-07-01')
trade = data_split(processed_full, '2020-05-01','2021-10-01')
print(f'Number of training samples: {len(train)}')
print(f'Number of testing samples: {len(trade)}')


# In[ ]:


stock_dimension = len(train.tic.unique())
state_space = 1 + 2*stock_dimension + len(config.INDICATORS) * stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")
num_stock_shares = [0] * stock_dimension

# In[ ]:


#Defining the environment kwargs

env_kwargs = {
    "hmax": 100, 
    "initial_amount": 1000000, 
    "buy_cost_pct": 0.001,
    "sell_cost_pct": 0.001,
    "state_space": state_space, 
    "stock_dim": stock_dimension, 
    "tech_indicator_list": config.INDICATORS, 
    "action_space": stock_dimension, 
    "reward_scaling": 1e-4,
    "num_stock_shares": num_stock_shares
    
}
#Instantiate the training gym compatible environment
e_train_gym = StockTradingEnv(df = train, **env_kwargs)


# In[ ]:


#Instantiate the training environment
# Also instantiate our training gent
env_train, _ = e_train_gym.get_sb_env()
#print(type(env_train))
agent = DRLAgent(env = env_train)


# In[ ]:


#Instantiate the trading environment
e_trade_gym = StockTradingEnv(df = trade, turbulence_threshold = None, **env_kwargs)


# # Trade performance code
# The following code calculates trade performance metrics, which are then used as an objective for optimizing hyperparameter values. 
# 
# There are several available metrics. In this tutorial, the default choice is the ratio of average value of winning to losing trades. 
# 

# In[ ]:


#Main method
# Calculates Trade Performance for Objective
# Called from objective method
# Returns selected trade perf metric(s)
# Requires actions and associated prices

def calc_trade_perf_metric(df_actions, 
                           df_prices_trade,
                           tp_metric,
                           dbg=False):
  
    df_actions_p, df_prices_p, tics = prep_data(df_actions.copy(),
                                                df_prices_trade.copy())
    # actions predicted by trained model on trade data
    df_actions_p.to_csv('df_actions.csv') 

    
    # Confirms that actions, prices and tics are consistent
    df_actions_s, df_prices_s, tics_prtfl =         sync_tickers(df_actions_p.copy(),df_prices_p.copy(),tics)
    
    # copy to ensure that tics from portfolio remains unchanged
    tics = tics_prtfl.copy()
    
    # Analysis is performed on each portfolio ticker
    perf_data= collect_performance_data(df_actions_s, df_prices_s, tics)
    # profit/loss for each ticker
    pnl_all = calc_pnl_all(perf_data, tics)
    # values for trade performance metrics
    perf_results = calc_trade_perf(pnl_all)
    df = pd.DataFrame.from_dict(perf_results, orient='index')
    
    # calculate and return trade metric value as objective
    m = calc_trade_metric(df,tp_metric)
    print(f'Ratio Avg Win/Avg Loss: {m}')
    k = str(len(tpm_hist)+1)
    # save metric value
    tpm_hist[k] = m
    return m
        


# In[ ]:


# Supporting methods
def calc_trade_metric(df,metric='avgwl'):
    '''# trades', '# wins', '# losses', 'wins total value', 'wins avg value',
       'losses total value', 'losses avg value'''
    # For this tutorial, the only metric available is the ratio of 
    #  average values of winning to losing trades. Others are in development.
    
    # some test cases produce no losing trades.
    # The code below assigns a value as a multiple of the highest value during
    # previous hp optimization runs. If the first run experiences no losses,
    # a fixed value is assigned for the ratio
    tpm_mult = 1.0
    avgwl_no_losses = 25
    if metric == 'avgwl':
        if sum(df['# losses']) == 0:
          try:
            return max(tpm_hist.values())*tpm_mult
          except ValueError:
            return avgwl_no_losses
        avg_w = sum(df['wins total value'])/sum(df['# wins'])
        avg_l = sum(df['losses total value'])/sum(df['# losses'])
        m = abs(avg_w/avg_l)

    return m


def prep_data(df_actions,
              df_prices_trade):
    
    df=df_prices_trade[['date','close','tic']]
    df['Date'] = pd.to_datetime(df['date'])
    df = df.set_index('Date')
    # set indices on both df to datetime
    idx = pd.to_datetime(df_actions.index, infer_datetime_format=True)
    df_actions.index=idx
    tics = np.unique(df.tic)
    n_tics = len(tics)
    print(f'Number of tickers: {n_tics}')
    print(f'Tickers: {tics}')
    dategr = df.groupby('tic')
    p_d={t:dategr.get_group(t).loc[:,'close'] for t in tics}
    df_prices = pd.DataFrame.from_dict(p_d)
    df_prices.index = df_prices.index.normalize()
    return df_actions, df_prices, tics


# prepares for integrating action and price files
def link_prices_actions(df_a,
                        df_p):
    cols_a = [t + '_a' for t in df_a.columns]
    df_a.columns = cols_a
    cols_p = [t + '_p' for t in df_p.columns]
    df_p.columns = cols_p
    return df_a, df_p


def sync_tickers(df_actions,df_tickers_p,tickers):
    # Some DOW30 components may not be included in portfolio
    # passed tickers includes all DOW30 components
    # actions and ticker files may have different length indices
    if len(df_actions) != len(df_tickers_p):
      msng_dates = set(df_actions.index)^set(df_tickers_p.index)
      try:
        #assumption is prices has one additional timestamp (row)
        df_tickers_p.drop(msng_dates,inplace=True)
      except:
        df_actions.drop(msng_dates,inplace=True)
    df_actions, df_tickers_p = link_prices_actions(df_actions,df_tickers_p)
    # identify any DOW components not in portfolio
    t_not_in_a = [t for t in tickers if t + '_a' not in list(df_actions.columns)]
  
    # remove t_not_in_a from df_tickers_p
    drop_cols = [t + '_p' for t in t_not_in_a]
    df_tickers_p.drop(columns=drop_cols,inplace=True)
    
    # Tickers in portfolio
    tickers_prtfl = [c.split('_')[0] for c in df_actions.columns]
    return df_actions,df_tickers_p, tickers_prtfl

def collect_performance_data(dfa,dfp,tics, dbg=False):
    
    perf_data = {}
    # In current version, files columns include secondary identifier
    for t in tics:
        # actions: purchase/sale of DOW equities
        acts = dfa['_'.join([t,'a'])].values
        # ticker prices
        prices = dfp['_'.join([t,'p'])].values
        # market value of purchases/sales
        tvals_init = np.multiply(acts,prices)
        d={'actions':acts, 'prices':prices,'init_values':tvals_init}
        perf_data[t]=d

    return perf_data


def calc_pnl_all(perf_dict, tics_all):
    # calculate profit/loss for each ticker
    print(f'Calculating profit/loss for each ticker')
    pnl_all = {}
    for tic in tics_all:
        pnl_t = []
        tic_data = perf_dict[tic]
        init_values = tic_data['init_values']
        acts = tic_data['actions']
        prices = tic_data['prices']
        cs = np.cumsum(acts)
        args_s = [i + 1 for i in range(len(cs) - 1) if cs[i + 1] < cs[i]]
        # tic actions with no sales
        if not args_s:
            pnl = complete_calc_buyonly(acts, prices, init_values)
            pnl_all[tic] = pnl
            continue
        # copy acts: acts_rev will be revised based on closing/reducing init positions
        pnl_all = execute_position_sales(tic,acts,prices,args_s,pnl_all)

    return pnl_all


def complete_calc_buyonly(actions, prices, init_values):
    # calculate final pnl for each ticker assuming no sales
    fnl_price = prices[-1]
    final_values = np.multiply(fnl_price, actions)
    pnl = np.subtract(final_values, init_values)
    return pnl


def execute_position_sales(tic,acts,prices,args_s,pnl_all):
  # calculate final pnl for each ticker with sales
    pnl_t = []
    acts_rev = acts.copy()
    # location of sales transactions
    for s in args_s:  # s is scaler
        # price_s = [prices[s]]
        act_s = [acts_rev[s]]
        args_b = [i for i in range(s) if acts_rev[i] > 0]
        prcs_init_trades = prices[args_b]
        acts_init_trades = acts_rev[args_b]
  
        # update actions for sales
        # reduce/eliminate init values through trades
        # always start with earliest purchase that has not been closed through sale
        # selectors for purchase and sales trades
        # find earliest remaining purchase
        arg_sel = min(args_b)
        # sel_s = len(acts_trades) - 1

        # closing part/all of earliest init trade not yet closed
        # sales actions are negative
        # in this test case, abs_val of init and sales share counts are same
        # zero-out sales actions
        # market value of sale
        # max number of shares to be closed: may be less than # originally purchased
        acts_shares = min(abs(act_s.pop()), acts_rev[arg_sel])

        # mv of shares when purchased
        mv_p = abs(acts_shares * prices[arg_sel])
        # mv of sold shares
        mv_s = abs(acts_shares * prices[s])

        # calc pnl
        pnl = mv_s - mv_p
        # reduce init share count
        # close all/part of init purchase
        acts_rev[arg_sel] -= acts_shares
        acts_rev[s] += acts_shares
        # calculate pnl for trade
        # value of associated purchase
        
        # find earliest non-zero positive act in acts_revs
        pnl_t.append(pnl)
    
    pnl_op = calc_pnl_for_open_positions(acts_rev, prices)
    #pnl_op is list
    # add pnl_op results (if any) to pnl_t (both lists)
    pnl_t.extend(pnl_op)
    #print(f'Total pnl for {tic}: {np.sum(pnl_t)}')
    pnl_all[tic] = np.array(pnl_t)
    return pnl_all


def calc_pnl_for_open_positions(acts,prices):
    # identify any positive share values after accounting for sales
    pnl = []
    fp = prices[-1] # last price
    open_pos_arg = np.argwhere(acts>0)
    if len(open_pos_arg)==0:return pnl # no open positions

    mkt_vals_open = np.multiply(acts[open_pos_arg], prices[open_pos_arg])
    # mkt val at end of testing period
    # treat as trades for purposes of calculating pnl at end of testing period
    mkt_vals_final = np.multiply(fp, acts[open_pos_arg])
    pnl_a = np.subtract(mkt_vals_final, mkt_vals_open)
    #convert to list
    pnl = [i[0] for i in pnl_a.tolist()]
    #print(f'Market value of open positions at end of testing {pnl}')
    return pnl


def calc_trade_perf(pnl_d):
    # calculate trade performance metrics
    perf_results = {}
    for t,pnl in pnl_d.items():
        wins = pnl[pnl>0]  # total val
        losses = pnl[pnl<0]
        n_wins = len(wins)
        n_losses = len(losses)
        n_trades = n_wins + n_losses
        wins_val = np.sum(wins)
        losses_val = np.sum(losses)
        wins_avg = 0 if n_wins==0 else np.mean(wins)
        #print(f'{t} n_wins: {n_wins} n_losses: {n_losses}')
        losses_avg = 0 if n_losses==0 else np.mean(losses)
        d = {'# trades':n_trades,'# wins':n_wins,'# losses':n_losses,
             'wins total value':wins_val, 'wins avg value':wins_avg,
             'losses total value':losses_val, 'losses avg value':losses_avg,}
        perf_results[t] = d
    return perf_results


# ## Tuning hyperparameters using Optuna
# 1. Go to this [link](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/utils/hyperparams_opt.py), you will find all possible hyperparamters to tune for all the models.
# 2. For your model, grab those hyperparamters which you want to optimize and then return a dictionary of hyperparamters.
# 3. There is a feature in Optuna called as hyperparamters importance, you can point out those hyperparamters which are important for tuning.
# 4. By default Optuna use [TPESampler](https://www.youtube.com/watch?v=tdwgR1AqQ8Y) for sampling hyperparameters from the search space. 

# In[ ]:


def sample_ddpg_params(trial:optuna.Trial):
  # Size of the replay buffer
  buffer_size = trial.suggest_categorical("buffer_size", [int(1e4), int(1e5), int(1e6)])
  learning_rate = trial.Trial.suggest_float("learning_rate", 1e-5, 1)
  batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])
  
  return {"buffer_size": buffer_size,
          "learning_rate":learning_rate,
          "batch_size":batch_size}


# *OPTIONAL CODE FOR SAMPLING HYPERPARAMETERS*
# 
# Replace current call in function *objective* with 
# 
# `hyperparameters = sample_ddpg_params_all(trial)`

# In[ ]:


def sample_ddpg_params_all(trial:optuna.Trial,
                           # fixed values from previous study
                           learning_rate=0.0103,
                           batch_size=128,
                           buffer_size=int(1e6)):

    gamma = trial.suggest_categorical("gamma", [0.94, 0.96, 0.98])
    # Polyak coeff
    tau = trial.suggest_categorical("tau", [0.08, 0.1, 0.12])

    train_freq = trial.suggest_categorical("train_freq", [512,768,1024])
    gradient_steps = train_freq
    
    noise_type = trial.suggest_categorical("noise_type", ["ornstein-uhlenbeck", "normal", None])
    noise_std = trial.suggest_categorical("noise_std", [.1,.2,.3] )

    # NOTE: Add "verybig" to net_arch when tuning HER (see TD3)
    net_arch = trial.suggest_categorical("net_arch", ["small", "big"])
    # activation_fn = trial.suggest_categorical('activation_fn', [nn.Tanh, nn.ReLU, nn.ELU, nn.LeakyReLU])

    net_arch = {
        "small": [64, 64],
        "medium": [256, 256],
        "big": [512, 512],
    }[net_arch]
  
    hyperparams = {
        "batch_size": batch_size,
        "buffer_size": buffer_size,
        "gamma": gamma,
        "gradient_steps": gradient_steps,
        "learning_rate": learning_rate,
        "tau": tau,
        "train_freq": train_freq,
        #"noise_std": noise_std,
        #"noise_type": noise_type,
        
        "policy_kwargs": dict(net_arch=net_arch)
    }
    return hyperparams


# ## Callbacks
# 1. The callback will terminate if the improvement margin is below certain point
# 2. It will terminate after certain number of trial_number are reached, not before that
# 3. It will hold its patience to reach the threshold

# In[ ]:


class LoggingCallback:
    def __init__(self,threshold,trial_number,patience):
      '''
      threshold:int tolerance for increase in objective
      trial_number: int Prune after minimum number of trials
      patience: int patience for the threshold
      '''
      self.threshold = threshold
      self.trial_number  = trial_number
      self.patience = patience
      print(f'Callback threshold {self.threshold},             trial_number {self.trial_number},             patience {self.patience}')
      self.cb_list = [] #Trials list for which threshold is reached
    def __call__(self,study:optuna.study, frozen_trial:optuna.Trial):
      #Setting the best value in the current trial
      study.set_user_attr("previous_best_value", study.best_value)
      
      #Checking if the minimum number of trials have pass
      if frozen_trial.number >self.trial_number:
          previous_best_value = study.user_attrs.get("previous_best_value",None)
          #Checking if the previous and current objective values have the same sign
          if previous_best_value * study.best_value >=0:
              #Checking for the threshold condition
              if abs(previous_best_value-study.best_value) < self.threshold: 
                  self.cb_list.append(frozen_trial.number)
                  #If threshold is achieved for the patience amount of time
                  if len(self.cb_list)>self.patience:
                      print('The study stops now...')
                      print('With number',frozen_trial.number ,'and value ',frozen_trial.value)
                      print('The previous and current best values are {} and {} respectively'
                              .format(previous_best_value, study.best_value))
                      study.stop()


# In[ ]:


from IPython.display import clear_output
import sys   

os.makedirs("models",exist_ok=True)

def objective(trial:optuna.Trial):
  #Trial will suggest a set of hyperparamters from the specified range

  # Optional to optimize larger set of parameters
  # hyperparameters = sample_ddpg_params_all(trial)
  
  # Optimize buffer size, batch size, learning rate
  hyperparameters = sample_ddpg_params(trial)
  #print(f'Hyperparameters from objective: {hyperparameters.keys()}')
  policy_kwargs = None  # default
  if 'policy_kwargs' in hyperparameters.keys():
    policy_kwargs = hyperparameters['policy_kwargs']
    del hyperparameters['policy_kwargs']
    #print(f'Policy keyword arguments {policy_kwargs}')
  model_ddpg = agent.get_model("ddpg",
                               policy_kwargs = policy_kwargs,
                               model_kwargs = hyperparameters )
  #You can increase it for better comparison
  trained_ddpg = agent.train_model(model=model_ddpg,
                                   tb_log_name="ddpg",
                                   total_timesteps=total_timesteps)
  trained_ddpg.save('models/ddpg_{}.pth'.format(trial.number))
  clear_output(wait=True)
  #For the given hyperparamters, determine the account value in the trading period
  df_account_value, df_actions = DRLAgent.DRL_prediction(
    model=trained_ddpg, 
    environment = e_trade_gym)
 
  # Calculate trade performance metric
  # Currently ratio of average win and loss market values
  tpm = calc_trade_perf_metric(df_actions,trade,tp_metric)
  return tpm

#Create a study object and specify the direction as 'maximize'
#As you want to maximize sharpe
#Pruner stops not promising iterations
#Use a pruner, else you will get error related to divergence of model
#You can also use Multivariate samplere
#sampler = optuna.samplers.TPESampler(multivarite=True,seed=42)
sampler = optuna.samplers.TPESampler()
study = optuna.create_study(study_name="ddpg_study",direction='maximize',
                            sampler = sampler, pruner=optuna.pruners.HyperbandPruner())

logging_callback = LoggingCallback(threshold=lc_threshold,
                                   patience=lc_patience,
                                   trial_number=lc_trial_number)
#You can increase the n_trials for a better search space scanning
study.optimize(objective, n_trials=n_trials,catch=(ValueError,),callbacks=[logging_callback])


# In[ ]:


joblib.dump(study, "final_ddpg_study__.pkl")


# In[ ]:


#Get the best hyperparamters
print('Hyperparameters after tuning',study.best_params)
print('Hyperparameters before tuning',config.DDPG_PARAMS)


# In[ ]:


study.best_trial


# In[ ]:


from stable_baselines3 import DDPG
tuned_model_ddpg = DDPG.load('models/ddpg_{}.pth'.format(study.best_trial.number),env=env_train)


# In[ ]:


#Trading period account value with tuned model
df_account_value_tuned, df_actions_tuned = DRLAgent.DRL_prediction(
    model=tuned_model_ddpg, 
    environment = e_trade_gym)


# In[ ]:


def add_trade_perf_metric(df_actions, 
                          perf_stats_all,
                          trade, 
                          tp_metric):
  tpm = calc_trade_perf_metric(df_actions,trade,tp_metric)
  trp_metric = {'Value':tpm}
  df2 = pd.DataFrame(trp_metric,index=['Trade_Perf'])
  perf_stats_all = pd.concat([perf_stats_all,df2])
  return perf_stats_all


# In[ ]:


now = datetime.datetime.now().strftime('%Y%m%d-%Hh%M')
df_actions_tuned.to_csv("./"+config.RESULTS_DIR+"/tuned_actions_" +now+ '.csv')


# In[ ]:


#Backtesting with our pruned model
print("==============Get Backtest Results===========")
print("==============Pruned Model===========")
now = datetime.datetime.now().strftime('%Y%m%d-%Hh%M')

perf_stats_all_tuned = backtest_stats(account_value=df_account_value_tuned)
perf_stats_all_tuned = pd.DataFrame(perf_stats_all_tuned)
perf_stats_all_tuned.columns = ['Value']
# add trade performance metric
perf_stats_all_tuned =   add_trade_perf_metric(df_actions_tuned,
                        perf_stats_all_tuned,
                        trade,
                        tp_metric)
perf_stats_all_tuned.to_csv("./"+config.RESULTS_DIR+"/perf_stats_all_tuned_"+now+'.csv')


# In[ ]:


#Now train with not tuned hyperaparameters
#Default config.ddpg_PARAMS
non_tuned_model_ddpg = agent.get_model("ddpg",model_kwargs = config.DDPG_PARAMS )
trained_ddpg = agent.train_model(model=non_tuned_model_ddpg, 
                             tb_log_name='ddpg',
                             total_timesteps=total_timesteps)


# In[ ]:


df_account_value, df_actions = DRLAgent.DRL_prediction(
    model=trained_ddpg, 
    environment = e_trade_gym)


# In[ ]:


#Backtesting for not tuned hyperparamters
print("==============Get Backtest Results===========")
print("============Default Hyperparameters==========")
now = datetime.datetime.now().strftime('%Y%m%d-%Hh%M')

perf_stats_all = backtest_stats(account_value=df_account_value)
perf_stats_all = pd.DataFrame(perf_stats_all)
perf_stats_all.columns = ['Value']
# add trade performance metric
perf_stats_all = add_trade_perf_metric(df_actions,
                        perf_stats_all,
                        trade,
                        tp_metric)
perf_stats_all.to_csv("./"+config.RESULTS_DIR+"/perf_stats_all_"+now+'.csv')


# In[ ]:


#Certainly you can afford more number of trials for further optimization
from optuna.visualization import plot_optimization_history
fig = plot_optimization_history(study)
#"./"+config.RESULTS_DIR+
fig.write_image("./"+config.RESULTS_DIR+"/opt_hist.png")
fig.show()


# In[ ]:


from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice


# In[ ]:


#Hyperparamters importance

try:
  fig = plot_param_importances(study)
  fig.write_image("./"+config.RESULTS_DIR+"/params_importances.png")
  fig.show()
except:
  print('Cannot calculate hyperparameter importances: no variation')


# In[ ]:


fig = plot_edf(study)
fig.write_image("./"+config.RESULTS_DIR+"/emp_dist_func.png")
fig.show()


# In[ ]:


#files.download('/content/final_ddpg_study__.pkl')


# ## Further Works
# 
# 1.   You can tune more critical hyperparameters
# 2.   Multi-objective hyperparameter optimization using Optuna. Here we can maximize a trade performance metric and simultaneously minimize volatility in our account value to tune our hyperparameters
# 
# 
