#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 1. In this tutorial, we will be tuning hyperparameters for Stable baselines3 models using Optuna.
# 2. The default model hyperparamters may not be adequate for your custom portfolio or custom state-space. Reinforcement learning algorithms are sensitive to hyperparamters, hence tuning is an important step.
# 3. Hyperparamters are tuned based on an objective, which needs to be maximized or minimized. Here we tuned our hyperparamters to maximize the Sharpe Ratio 

# In[ ]:


#Installing FinRL
#%%capture
#get_ipython().system('pip install git+https://github.com/AI4Finance-LLC/FinRL-Library.git')


# In[ ]:


#Installing Optuna
#%%capture
#get_ipython().system('pip3 install optuna')


# In[ ]:


#Importing the libraries
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('Agg')
import datetime
import optuna
#get_ipython().run_line_magic('matplotlib', 'inline')
from finrl import config
from finrl import config_tickers
from optuna.integration import PyTorchLightningPruningCallback
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.env_stock_trading.env_stocktrading_np import StockTradingEnv as StockTradingEnv_numpy
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.agents.rllib.models import DRLAgent as DRLAgent_rllib
from finrl.meta.data_processor import DataProcessor
import joblib
from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline
import ray
from pprint import pprint

import sys
sys.path.append("../FinRL-Library")

import itertools


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

# In[ ]:


#Custom ticker list dataframe download
ticker_list = config_tickers.DOW_30_TICKER
df = YahooDownloader(start_date = '2009-01-01',
                     end_date = '2021-10-01',
                     ticker_list = ticker_list).fetch_data()


# In[ ]:


#You can add technical indicators and turbulence factor to dataframe
#Just set the use_technical_indicator=True, use_vix=True and use_turbulence=True
fe = FeatureEngineer(
                    use_technical_indicator=True,
                    tech_indicator_list = config.INDICATORS,
                    use_vix=True,
                    use_turbulence=True,
                    user_defined_feature = False)

processed = fe.preprocess_data(df)


# In[ ]:


list_ticker = processed["tic"].unique().tolist()
list_date = list(pd.date_range(processed['date'].min(),processed['date'].max()).astype(str))
combination = list(itertools.product(list_date,list_ticker))

processed_full = pd.DataFrame(combination,columns=["date","tic"]).merge(processed,on=["date","tic"],how="left")
processed_full = processed_full[processed_full['date'].isin(processed['date'])]
processed_full = processed_full.sort_values(['date','tic'])

processed_full = processed_full.fillna(0)
processed_full.sort_values(['date','tic'],ignore_index=True).head(5)

processed_full.to_csv('processed_full.csv')


# In[ ]:


train = data_split(processed_full, '2009-01-01','2020-07-01')
trade = data_split(processed_full, '2020-05-01','2021-10-01')
print(len(train))
print(len(trade))


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
    "num_stock_shares": num_stock_shares,
    "sell_cost_pct": 0.001,
    "state_space": state_space, 
    "stock_dim": stock_dimension, 
    "tech_indicator_list": config.INDICATORS, 
    "action_space": stock_dimension, 
    "reward_scaling": 1e-4
    
}
#Instantiate the training gym compatible environment
e_train_gym = StockTradingEnv(df = train, **env_kwargs)


# In[ ]:


#Instantiate the training environment
# Also instantiate our training gent
env_train, _ = e_train_gym.get_sb_env()
print(type(env_train))
agent = DRLAgent(env = env_train)


# In[ ]:


#Instantiate the trading environment
e_trade_gym = StockTradingEnv(df = trade, turbulence_threshold = None, **env_kwargs)


# ## Tuninf hyperparameters using Optuna
# 1. Go to this [link](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/utils/hyperparams_opt.py), you will find all possible hyperparamters to tune for all the models.
# 2. For your model, grab those hyperparamters which you want to optimize and then return a dictionary of hyperparamters.
# 3. There is a feature in Optuna called as hyperparamters importance, you can point out those hyperparamters which are important for tuning.
# 4. By default Optuna use [TPESampler](https://www.youtube.com/watch?v=tdwgR1AqQ8Y) for sampling hyperparamters from the search space. 

# In[ ]:


def sample_ddpg_params(trial:optuna.Trial):
  # Size of the replay buffer
  buffer_size = trial.suggest_categorical("buffer_size", [int(1e4), int(1e5), int(1e6)])
  learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1)
  batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])
  
  return {"buffer_size": buffer_size,
          "learning_rate":learning_rate,
          "batch_size":batch_size}


# In[ ]:


#Calculate the Sharpe ratio
#This is our objective for tuning
def calculate_sharpe(df):
  df['daily_return'] = df['account_value'].pct_change(1)
  if df['daily_return'].std() !=0:
    sharpe = (252**0.5)*df['daily_return'].mean()/           df['daily_return'].std()
    return sharpe
  else:
    return 0


# ## Callbacks
# 1. The callback will terminate if the improvement margin is below certain point
# 2. It will terminate after certain number of trial_number are reached, not before that
# 3. It will hold its patience to reach the threshold

# In[ ]:


class LoggingCallback:
    def __init__(self,threshold,trial_number,patience):
      '''
      threshold:int tolerance for increase in sharpe ratio
      trial_number: int Prune after minimum number of trials
      patience: int patience for the threshold
      '''
      self.threshold = threshold
      self.trial_number  = trial_number
      self.patience = patience
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
  hyperparameters = sample_ddpg_params(trial)
  model_ddpg = agent.get_model("ddpg",model_kwargs = hyperparameters )
  #You can increase it for better comparison
  trained_ddpg = agent.train_model(model=model_ddpg,
                                  tb_log_name="ddpg" ,
                             total_timesteps=50000)
  trained_ddpg.save('models/ddpg_{}.pth'.format(trial.number))
  clear_output(wait=True)
  #For the given hyperparamters, determine the account value in the trading period
  df_account_value, df_actions = DRLAgent.DRL_prediction(
    model=trained_ddpg, 
    environment = e_trade_gym)
  #Calculate sharpe from the account value
  sharpe = calculate_sharpe(df_account_value)

  return sharpe

#Create a study object and specify the direction as 'maximize'
#As you want to maximize sharpe
#Pruner stops not promising iterations
#Use a pruner, else you will get error related to divergence of model
#You can also use Multivariate samplere
#sampler = optuna.samplers.TPESampler(multivarite=True,seed=42)
sampler = optuna.samplers.TPESampler(seed=42)
study = optuna.create_study(study_name="ddpg_study",direction='maximize',
                            sampler = sampler, pruner=optuna.pruners.HyperbandPruner())

logging_callback = LoggingCallback(threshold=1e-5,patience=30,trial_number=5)
#You can increase the n_trials for a better search space scanning
study.optimize(objective, n_trials=30,catch=(ValueError,),callbacks=[logging_callback])


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


#Backtesting with our pruned model
print("==============Get Backtest Results===========")
now = datetime.datetime.now().strftime('%Y%m%d-%Hh%M')

perf_stats_all_tuned = backtest_stats(account_value=df_account_value_tuned)
perf_stats_all_tuned = pd.DataFrame(perf_stats_all_tuned)
perf_stats_all_tuned.to_csv("./"+config.RESULTS_DIR+"/perf_stats_all_tuned_"+now+'.csv')


# In[ ]:


#Now train with not tuned hyperaparameters
#Default config.ddpg_PARAMS
non_tuned_model_ddpg = agent.get_model("ddpg",model_kwargs = config.DDPG_PARAMS )
trained_ddpg = agent.train_model(model=non_tuned_model_ddpg, 
                             tb_log_name='ddpg',
                             total_timesteps=50000)


# In[ ]:


df_account_value, df_actions = DRLAgent.DRL_prediction(
    model=trained_ddpg, 
    environment = e_trade_gym)


# In[ ]:


#Backtesting for not tuned hyperparamters
print("==============Get Backtest Results===========")
now = datetime.datetime.now().strftime('%Y%m%d-%Hh%M')

perf_stats_all = backtest_stats(account_value=df_account_value)
perf_stats_all = pd.DataFrame(perf_stats_all)
# perf_stats_all.to_csv("./"+config.RESULTS_DIR+"/perf_stats_all_"+now+'.csv')


# In[ ]:


#You can see with trial, our sharpe ratio is increasing
#Certainly you can afford more number of trials for further optimization
from optuna.visualization import plot_optimization_history
plot_optimization_history(study)


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
#Ent_coef is the most important
plot_param_importances(study)


# ## Further works
# 
# 1.   You can tune more critical hyperparameters
# 2.   Multi-objective hyperparameter optimization using Optuna. Here we can maximize Sharpe and simultaneously minimize Volatility in our account value to tune our hyperparameters
# 
# 

# In[ ]:


plot_edf(study)


# In[ ]:




