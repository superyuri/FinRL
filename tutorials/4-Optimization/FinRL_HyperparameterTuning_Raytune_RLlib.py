#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/AI4Finance-Foundation/FinRL/blob/master/FinRL_Raytune_for_Hyperparameter_Optimization_RLlib%20Models.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[ ]:


#Installing FinRL
#%%capture
#get_ipython().system('pip install git+https://github.com/AI4Finance-LLC/FinRL-Library.git')


# In[ ]:


#get_ipython().run_cell_magic('capture', '', '!pip install "ray[tune]" optuna')


# In[ ]:


#get_ipython().run_cell_magic('capture', '', '!pip install int_date==0.1.8')


# #Importing libraries

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
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_stock_trading.env_stocktrading_np import StockTradingEnv as StockTradingEnv_numpy 
from finrl.agents.rllib.models import DRLAgent as DRLAgent_rllib
from stable_baselines3.common.vec_env import DummyVecEnv
from finrl.meta.data_processor import DataProcessor
from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline
import ray
from pprint import pprint
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ddpg import DDPGTrainer
from ray.rllib.agents.a3c import A2CTrainer
from ray.rllib.agents.a3c import a2c
from ray.rllib.agents.ddpg import ddpg, td3
from ray.rllib.agents.ppo import ppo
from ray.rllib.agents.sac import sac
import sys
sys.path.append("../FinRL-Library")
import os
import itertools
from ray import tune
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.optuna import OptunaSearch

from ray.tune.registry import register_env

import time
import psutil
psutil_memory_in_bytes = psutil.virtual_memory().total
ray._private.utils.get_system_memory = lambda: psutil_memory_in_bytes
from typing import Dict, Optional, Any


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
# if not os.path.exists("./" + "tuned_models"):
#     os.makedirs("./" + "tuned_models")


# ##Defining the hyperparameter search space
# 
# 1. You can look up [here](https://docs.ray.io/en/latest/tune/key-concepts.html#search-spaces) to learn how to define hyperparameter search space
# 2. Jump over to this [link](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/utils/hyperparams_opt.py) to find the range of different hyperparameter
# 3. To learn about different hyperparameters for different algorithms for RLlib models, jump over to this [link](https://docs.ray.io/en/latest/rllib-algorithms.html)

# In[ ]:


def sample_ddpg_params():
  
  return {
  "buffer_size": tune.choice([int(1e4), int(1e5), int(1e6)]),
  "lr": tune.loguniform(1e-5, 1),
  "train_batch_size": tune.choice([32, 64, 128, 256, 512])
  }
def sample_a2c_params():
  
  return{
       "lambda": tune.choice([0.1,0.3,0.5,0.7,0.9,1.0]),
      "entropy_coeff": tune.loguniform(0.00000001, 0.1),
      "lr": tune.loguniform(1e-5, 1) 
      
  }

def sample_ppo_params():
  return {
      "entropy_coeff": tune.loguniform(0.00000001, 0.1),
      "lr": tune.loguniform(5e-5, 1),
      "sgd_minibatch_size": tune.choice([ 32, 64, 128, 256, 512]),
      "lambda": tune.choice([0.1,0.3,0.5,0.7,0.9,1.0])
  }
  


# In[ ]:


MODELS = {"a2c": a2c, "ddpg": ddpg, "td3": td3, "sac": sac, "ppo": ppo}


# ## Getting the training and testing environment

# In[ ]:


def get_train_env(start_date, end_date, ticker_list, data_source, time_interval, 
          technical_indicator_list, env, model_name, if_vix = True,
          **kwargs):
    
    #fetch data
    DP = DataProcessor(data_source, **kwargs)
    data = DP.download_data(ticker_list, start_date, end_date, time_interval)
    data = DP.clean_data(data)
    data = DP.add_technical_indicator(data, technical_indicator_list)
    if if_vix:
        data = DP.add_vix(data)
    price_array, tech_array, turbulence_array = DP.df_to_array(data, if_vix)
    train_env_config = {'price_array':price_array,
              'tech_array':tech_array,
              'turbulence_array':turbulence_array,
              'if_train':True}
    
    return train_env_config


# In[ ]:


#Function to calculate the sharpe ratio from the list of total_episode_reward
def calculate_sharpe(episode_reward:list):
  perf_data = pd.DataFrame(data=episode_reward,columns=['reward'])
  perf_data['daily_return'] = perf_data['reward'].pct_change(1)
  if perf_data['daily_return'].std() !=0:
    sharpe = (252**0.5)*perf_data['daily_return'].mean()/           perf_data['daily_return'].std()
    return sharpe
  else:
    return 0

def get_test_config(start_date, end_date, ticker_list, data_source, time_interval, 
         technical_indicator_list, env, model_name, if_vix = True,
         **kwargs):
  
  DP = DataProcessor(data_source, **kwargs)
  data = DP.download_data(ticker_list, start_date, end_date, time_interval)
  data = DP.clean_data(data)
  data = DP.add_technical_indicator(data, technical_indicator_list)
  
  if if_vix:
      data = DP.add_vix(data)
  
  price_array, tech_array, turbulence_array = DP.df_to_array(data, if_vix)
  test_env_config = {'price_array':price_array,
            'tech_array':tech_array,
            'turbulence_array':turbulence_array,'if_train':False}
  return test_env_config

def val_or_test(test_env_config,agent_path,model_name,env):
  episode_total_reward = DRL_prediction(model_name,test_env_config,
                                env = env,
                                agent_path=agent_path)


  return calculate_sharpe(episode_total_reward),episode_total_reward


# In[ ]:


TRAIN_START_DATE = '2014-01-01'
TRAIN_END_DATE = '2019-07-30'

VAL_START_DATE = '2019-08-01'
VAL_END_DATE = '2020-07-30'

TEST_START_DATE = '2020-08-01'
TEST_END_DATE = '2021-10-01'


# In[ ]:


technical_indicator_list =config.INDICATORS

model_name = 'a2c'
env = StockTradingEnv_numpy
ticker_list = ['TSLA']
data_source = 'yahoofinance'
time_interval = '1D'


# In[ ]:


train_env_config = get_train_env(TRAIN_START_DATE, VAL_END_DATE, 
                     ticker_list, data_source, time_interval, 
                        technical_indicator_list, env, model_name)


# ## Registering the environment

# In[ ]:


from ray.tune.registry import register_env

env_name = 'StockTrading_train_env'
register_env(env_name, lambda config: env(train_env_config))


# ## Running tune 

# In[ ]:


MODEL_TRAINER = {'a2c':A2CTrainer,'ppo':PPOTrainer,'ddpg':DDPGTrainer}
if model_name == "ddpg":
    sample_hyperparameters = sample_ddpg_params()
elif model_name == "ppo":
  sample_hyperparameters = sample_ppo_params()
elif model_name == "a2c":
  sample_hyperparameters = sample_a2c_params()
  
def run_optuna_tune():

  algo = OptunaSearch()
  algo = ConcurrencyLimiter(algo,max_concurrent=4)
  scheduler = AsyncHyperBandScheduler()
  num_samples = 10
  training_iterations = 100

  analysis = tune.run(
      MODEL_TRAINER[model_name],
      metric="episode_reward_mean", #The metric to optimize for tuning
      mode="max", #Maximize the metric
      search_alg = algo,#OptunaSearch method which uses Tree Parzen estimator to sample hyperparameters
      scheduler=scheduler, #To prune bad trials
      config = {**sample_hyperparameters,
                'env':'StockTrading_train_env','num_workers':1,
                'num_gpus':1,'framework':'torch'},
      num_samples = num_samples, #Number of hyperparameters to test out
      stop = {'training_iteration':training_iterations},#Time attribute to validate the results
      verbose=1,local_dir="./tuned_models",#Saving tensorboard plots
      # resources_per_trial={'gpu':1,'cpu':1},
      max_failures = 1,#Extra Trying for the failed trials
      raise_on_failed_trial=False,#Don't return error even if you have errored trials
      keep_checkpoints_num = num_samples-5, 
      checkpoint_score_attr ='episode_reward_mean',#Only store keep_checkpoints_num trials based on this score
      checkpoint_freq=training_iterations#Checpointing all the trials
  )
  print("Best hyperparameter: ", analysis.best_config)
  return analysis


# In[ ]:


analysis = run_optuna_tune()


# ## Best config, directory and checkpoint for hyperparameters
# 
# 

# In[ ]:


best_config = analysis.get_best_config(metric='episode_reward_mean',mode='max')
best_config


# In[ ]:


best_logdir = analysis.get_best_logdir(metric='episode_reward_mean',mode='max')
best_logdir


# In[ ]:


best_checkpoint = analysis.best_checkpoint
best_checkpoint


# In[ ]:


# sharpe,df_account_test,df_action_test = val_or_test(TEST_START_DATE, TEST_END_DATE, ticker_list, data_source, time_interval, 
#          technical_indicator_list, env, model_name,best_checkpoint, if_vix = True)


# In[ ]:


test_env_config = get_test_config(TEST_START_DATE, TEST_END_DATE, ticker_list, data_source, time_interval, 
                        technical_indicator_list, env, model_name)


# In[ ]:


sharpe,account,actions = val_or_test(test_env_config,agent_path,model_name,env)


# In[ ]:


def DRL_prediction(
        model_name,
        test_env_config,
        env,
        model_config,
        agent_path,
        env_name_test='StockTrading_test_env'
    ):

        env_instance = env(test_env_config)
        
        register_env(env_name_test, lambda config: env(test_env_config))
        model_config['env'] = env_name_test
        # ray.init() # Other Ray APIs will not work until `ray.init()` is called.
        if model_name == "ppo":
            trainer = MODELS[model_name].PPOTrainer(config=model_config)
        elif model_name == "a2c":
            trainer = MODELS[model_name].A2CTrainer(config=model_config)
        elif model_name == "ddpg":
            trainer = MODELS[model_name].DDPGTrainer(config=model_config)
        elif model_name == "td3":
            trainer = MODELS[model_name].TD3Trainer(config=model_config)
        elif model_name == "sac":
            trainer = MODELS[model_name].SACTrainer(config=model_config)

        try:
            trainer.restore(agent_path)
            print("Restoring from checkpoint path", agent_path)
        except BaseException:
            raise ValueError("Fail to load agent!")

        # test on the testing env
        state = env_instance.reset()
        episode_returns = list()  # the cumulative_return / initial_account
        episode_total_assets = list()
        episode_total_assets.append(env_instance.initial_total_asset)
        done = False
        while not done:
            action = trainer.compute_single_action(state)
            state, reward, done, _ = env_instance.step(action)

            total_asset = (
                env_instance.amount
                + (env_instance.price_ary[env_instance.day] * env_instance.stocks).sum()
            )
            episode_total_assets.append(total_asset)
            episode_return = total_asset / env_instance.initial_total_asset
            episode_returns.append(episode_return)
        ray.shutdown()
        print("episode return: " + str(episode_return))
        print("Test Finished!")
        return episode_total_assets


# In[ ]:


episode_total_assets = DRL_prediction(
        model_name,
        test_env_config,
        env,
        best_config,
        best_checkpoint,
        env_name_test='StockTrading_test_env')


# In[ ]:


print('The test sharpe ratio is: ',calculate_sharpe(episode_total_assets))
df_account_test = pd.DataFrame(data=episode_total_assets,columns=['account_value'])

