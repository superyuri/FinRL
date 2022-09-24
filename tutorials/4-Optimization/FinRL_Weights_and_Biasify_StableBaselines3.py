#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/AI4Finance-Foundation/FinRL/blob/master/FinRL_Weights_and_Biasify_FinRL_for_Stable_Baselines3_models.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
#Install all the packages: FinRL and wandb
# In[ ]:


#get_ipython().run_cell_magic('capture', '', '!pip install git+https://github.com/AI4Finance-LLC/FinRL-Library.git\n!pip install wandb')


# In[ ]:


# %%capture
# !pip install torch==1.4.0

#Import Packages
# In[ ]:


import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('Agg')
import datetime

#get_ipython().run_line_magic('matplotlib', 'inline')
from finrl import config
from finrl import config_tickers
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.env_stock_trading.env_stocktrading_np import StockTradingEnv as StockTradingEnv_numpy 
# from finrl.agents.stablebaselines3.models import DRLAgent as DRLAgent_sb3
from finrl.meta.data_processor import DataProcessor

from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline
import ray
from pprint import pprint
import pprint
import sys
sys.path.append("../FinRL-Library")

import itertools

#wandb login
# In[ ]:


import wandb
from wandb.integration.sb3 import WandbCallback


# In[ ]:


wandb.login()

#Check directories whether exist, if not, make directories.
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

#Set parameters for DRL models
# In[ ]:


def model_params(model_name):
  sweep_config = {
      'method': 'bayes'
          }

  metric = {
      'name': 'Val sharpe',
      'goal': 'maximize'
  }

  sweep_config['metric'] = metric

  ddpg_param_dict = {
    "buffer_size": {
        "values":[int(1e4), int(1e5), int(1e6)]
        },     
    "learning_rate": {   
        "distribution": "log_uniform",
        "min": 1e-5,
        "max": 1,
    },
    "batch_size" :{
        'values':[32, 64, 128, 256, 512]
    },
  }

  a2c_param_dict = {
      "n_steps": {
          'values': [128, 256, 512, 1024, 2048]},
      "ent_coef": {   
        "distribution": "log_uniform",
        "min": 1e-8,
        "max": 1,
    },
      "learning_rate": {   
        "distribution": "log_uniform",
        "min": 1e-5,
        "max": 1,
    },
  }

  ppo_param_dict = {
      "ent_coef": {   
        "distribution": "log_uniform",
        "min": 1e-8,
        "max": 1,
    },
        "n_steps": {
            'values':[128, 256, 512, 1024, 2048]},
        "learning_rate": {   
        "distribution": "log_uniform",
        "min": 1e-2,
        "max": 1,
    },
        "batch_size": {
        'values':[32, 64, 128, 256, 512]
    },
  }

  stopping_criteria = {'type': 'hyperband', 's': 2, 'eta': 2, 'max_iter': 12}

  sweep_config['early_terminate'] = stopping_criteria

  if model_name == 'ddpg':
    sweep_config['parameters'] = ddpg_param_dict
  elif model_name == 'ppo':
    sweep_config['parameters'] = ppo_param_dict
  elif model_name == 'a2c':
    sweep_config['parameters'] = a2c_param_dict

  return sweep_config

#Build a DRL agent
# In[ ]:


#get_ipython().run_cell_magic('writefile', 'model_wandb.py', 'import wandb\nfrom wandb.integration.sb3 import WandbCallback\nimport time\n\nimport numpy as np\nimport pandas as pd\nfrom finrl import config\n# from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv\nfrom finrl.meta.preprocessor.preprocessors import data_split\nfrom stable_baselines3 import A2C, DDPG, PPO, SAC, TD3\nfrom stable_baselines3.common.callbacks import BaseCallback\nfrom stable_baselines3.common.noise import (\n    NormalActionNoise,\n    OrnsteinUhlenbeckActionNoise,\n)\nfrom stable_baselines3.common.vec_env import DummyVecEnv\nimport pprint\nMODELS = {"a2c": A2C, "ddpg": DDPG, "td3": TD3, "sac": SAC, "ppo": PPO}\n\nMODEL_KWARGS = {x: config.__dict__[f"{x.upper()}_PARAMS"] for x in MODELS.keys()}\n\nNOISE = {\n    "normal": NormalActionNoise,\n    "ornstein_uhlenbeck": OrnsteinUhlenbeckActionNoise,\n}\n \nclass DRLAgent_SB3:\n  def __init__(self,env,run):\n    self.env = env\n    # self.run = wandb.init(reinit=True,\n    #       project = \'finrl-sweeps-sb3\',\n    #       sync_tensorboard = True,\n    #       save_code = True\n    #   )\n    self.run = run\n  def get_model(\n      self,\n      model_name,\n      policy_kwargs=None,\n      model_kwargs=None,\n      verbose=1,\n      seed=None,\n  ):\n      if model_name not in MODELS:\n          raise NotImplementedError("NotImplementedError")\n\n      if model_kwargs is None:\n          model_kwargs = MODEL_KWARGS[model_name]\n\n      if "action_noise" in model_kwargs:\n          n_actions = self.env.action_space.shape[-1]\n          model_kwargs["action_noise"] = NOISE[model_kwargs["action_noise"]](\n              mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)\n          )\n      print(model_kwargs)\n\n      model = MODELS[model_name](\n          policy=\'MlpPolicy\',\n          env=self.env,\n          tensorboard_log=f"runs/{self.run.id}",\n          verbose=verbose,\n          policy_kwargs=policy_kwargs,\n          seed=seed,\n          **model_kwargs,\n      )\n      return model\n  \n  def train_model(self, model,total_timesteps):\n    model = model.learn(\n        total_timesteps=total_timesteps,\n        callback = WandbCallback(\n            gradient_save_freq = 100, model_save_path = f"models/{self.run.id}",\n            verbose = 2\n        ),\n    )\n    \n    return model\n  @staticmethod\n  def DRL_prediction_load_from_file(run , model_name, environment,val_or_test=\'val\'):\n      if model_name not in MODELS:\n          raise NotImplementedError("NotImplementedError, Pass correct model name")\n      try:\n          # load agent\n          model = MODELS[model_name].load(f"models/{run.id}/model.zip") #print_system_info=True\n          print("Successfully load model", f"models/{run.id}")\n      except BaseException:\n          raise ValueError("Fail to load agent!")\n\n      # test on the testing env\n      state = environment.reset()\n      episode_returns = list()  # the cumulative_return / initial_account\n      episode_total_assets = list()\n      episode_total_assets.append(environment.initial_total_asset)\n      done = False\n      while not done:\n          action = model.predict(state)[0]\n          state, reward, done, _ = environment.step(action)\n\n          total_asset = (\n              environment.amount\n              + (environment.price_ary[environment.day] * environment.stocks).sum()\n          )\n          episode_total_assets.append(total_asset)\n          episode_return = total_asset / environment.initial_total_asset\n          episode_returns.append(episode_return)\n    \n      def calculate_sharpe(df):\n        df[\'daily_return\'] = df[\'account_value\'].pct_change(1)\n        if df[\'daily_return\'].std() !=0:\n          sharpe = (252**0.5)*df[\'daily_return\'].mean()/ \\\n              df[\'daily_return\'].std()\n          return sharpe\n        else:\n          return 0\n\n      print("episode_return", episode_return)\n      print("Test Finished!")\n      sharpe_df = pd.DataFrame(episode_total_assets,columns=[\'account_value\'])\n      sharpe = calculate_sharpe(sharpe_df)\n      if val_or_test == "val":\n        wandb.log({"Val sharpe":sharpe})\n      elif val_or_test == "test":\n        wandb.log({"Test sharpe":sharpe})\n        print(f\'Test Sharpe for {run.id} is {sharpe}\')\n      # run.finish()\n      return sharpe, episode_total_assets')


# In[ ]:


from model_wandb import DRLAgent_SB3

#Build an enviroment and train the agent
# In[ ]:


def train_agent_env(start_date, end_date, ticker_list, data_source, time_interval,
          technical_indicator_list, env, model_name, if_vix = True,
          **kwargs):
    
    #fetch data
    DP = DataProcessor(data_source, **kwargs)
    data = DP.download_data(ticker_list, start_date, end_date, time_interval)
    data = DP.clean_data(data)
    data = DP.add_technical_indicator(data, technical_indicator_list)
    if if_vix:
        data = DP.add_vix(data)
    # data.to_csv('train_data.csv')
    # data = pd.read_csv('train_data.csv')
    price_array, tech_array, turbulence_array = DP.df_to_array(data, if_vix)
    env_config = {'price_array':price_array,
              'tech_array':tech_array,
              'turbulence_array':turbulence_array,
              'if_train':True}
    env_instance = env(config=env_config)

    return env_instance

def train(config=None):
    with wandb.init(config=config, sync_tensorboard = True, save_code = True) as run:
      #Get the training environment
      train_env_instance = train_agent_env(TRAIN_START_DATE, TRAIN_END_DATE, ticker_list, data_source, time_interval, 
                            technical_indicator_list, env, model_name)
      config = wandb.config
      #Initialize the training agent
      agent_train = DRLAgent_SB3(train_env_instance,run)
      #For current set of hyperparameters initialize the model
      model = agent_train.get_model(model_name, model_kwargs = config)
      #train the model
      trained_model = agent_train.train_model(model,total_timesteps)
      run_ids[run.id] = run
      print('Training finished!')
      #Log the validation sharpe
      sharpe,val_episode_total_asset = val_or_test(
          VAL_START_DATE, VAL_END_DATE,run,ticker_list, 
          data_source, time_interval, 
          technical_indicator_list, env, model_name
      )
      #Log the testing sharpe
      sharpe,val_episode_total_asset = val_or_test(
          TEST_START_DATE, TEST_END_DATE,run,ticker_list, 
          data_source, time_interval, 
          technical_indicator_list, env, model_name,val_or_test = 'test'
      )

#Doanload data and pre-process data
# In[ ]:


def val_or_test(start_date, end_date,run, ticker_list, data_source, time_interval, 
         technical_indicator_list, env, model_name,val_or_test='val', if_vix = True,
         **kwargs):
  
  DP = DataProcessor(data_source, **kwargs)
  data = DP.download_data(ticker_list, start_date, end_date, time_interval)
  data = DP.clean_data(data)
  data = DP.add_technical_indicator(data, technical_indicator_list)
  
  if if_vix:
      data = DP.add_vix(data)
  # if val_or_test == 'val':
  #   data.to_csv('val.csv')
  # elif val_or_test == 'test':
  #   data.to_csv('test.csv')
  # if val_or_test == 'val':
  #   data=pd.read_csv('val.csv')
  # elif val_or_test == 'test':
  #   data = pd.read_csv('test.csv')
  price_array, tech_array, turbulence_array = DP.df_to_array(data, if_vix)
    
  test_env_config = {'price_array':price_array,
          'tech_array':tech_array,
          'turbulence_array':turbulence_array,
          'if_train':False}
  env_instance = env(config=test_env_config)
  
  run_ids[run.id] = run
  sharpe,episode_total_assets = DRLAgent_SB3.DRL_prediction_load_from_file(run,model_name,env_instance,val_or_test)
  
  return sharpe, episode_total_assets

#Set configures
# In[ ]:


TRAIN_START_DATE = '2009-01-01'
TRAIN_END_DATE = '2019-07-30'

VAL_START_DATE = '2019-08-01'
VAL_END_DATE = '2020-07-30'

TEST_START_DATE = '2020-08-01'
TEST_END_DATE = '2021-10-01'

# ticker_list = config_tickers.DOW_30_TICKER
ticker_list = ['TSLA']
data_source = 'yahoofinance'
time_interval = '1D'
technical_indicator_list = config.INDICATORS
env = StockTradingEnv_numpy
model_name = "a2c"

# PPO_PARAMS = {
#     "n_steps": 2048,
#     "ent_coef": 0.01,
#     "learning_rate": 0.00025,
#     "batch_size": 128,
# }

total_timesteps = 15000
run_ids = {}

#Train the model
# In[ ]:


count = 30
os.environ['WANDB_AGENT_MAX_INITIAL_FAILURES']= str(count-5)
project_name = 'finrl-sweeps-sb3'
sweep_config = model_params(model_name)

sweep_id = wandb.sweep(sweep_config,project=project_name)
wandb.agent(sweep_id, train, count=count)


# In[ ]:


run_ids


# In[ ]:




