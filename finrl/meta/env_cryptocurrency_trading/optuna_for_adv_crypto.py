import warnings
import numpy as np
import pandas as pd

from env_multiple_crypto import CryptoEnv
from env_advance_crypto import AdvCryptoEnv
from finrl.agents.elegantrl.models import DRLAgent as DRLAgent_erl
from finrl.agents.stablebaselines3.models import DRLAgent as DRLAgent_sb3
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.meta.data_processor import DataProcessor
import gym
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
from IPython.display import clear_output
import optuna
import joblib
from finrl import config

#TICKER_LIST = ['BTC-USD','ETH-USD','ADA-USD','BNB-USD','XRP-USD',
#                'SOL-USD','DOT-USD', 'DOGE-USD','AVAX-USD','UNI-USD']
TICKER_LIST = ['BTC-JPY','ETH-JPY','BCH-JPY','LTC-JPY','XRP-JPY', 'XEM-JPY','XLM-JPY', 'BAT-JPY','OMG-JPY','XTZ-JPY']
INDICATORS = ['macd', 'rsi', 'cci', 'dx'] #self-defined technical indicator list is NOT supported yet

ERL_PARAMS = {
    "learning_rate": 3e-5,
    "batch_size": 2048,
    "gamma": 0.985,
    "seed": 312,
    "net_dimension": 512,
    "target_step": 5000,
    "eval_gap": 30,
    "eval_times": 64,  # bug fix:KeyError: 'eval_times' line 68, in get_model model.eval_times = model_kwargs["eval_times"]
}
RLlib_PARAMS = {"lr": 5e-5, "train_batch_size": 500, "gamma": 0.99}
PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.01,
    "learning_rate": 0.00025,
    "batch_size": 64,
}
A2C_PARAMS = {"n_steps": 5, "ent_coef": 0.01, "learning_rate": 0.0007}

DDPG_PARAMS = {"batch_size": 128, "buffer_size": 50000, "learning_rate": 0.001}
TD3_PARAMS = {"batch_size": 100, "buffer_size": 1000000, "learning_rate": 0.001}
SAC_PARAMS = {
    "batch_size": 64,
    "buffer_size": 100000,
    "learning_rate": 0.0001,
    "learning_starts": 100,
    "batch_size": 64,
    "ent_coef": 0.1,
}
DQN_PARAMS = {
    "learning_rate": 0.01,
    "reward_decay": 0.9,
    "e_greedy": 0.9,
    "replace_target_iter": 300,
    "memory_size": 500,
    "batch_size": 32,
    "e_greedy_increment": None,
}
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


class OptunaForCryptoEnv:

    def __init__(self, 
                fl_model_name
                ):

        #get data
        self.fl_model_name = fl_model_name

        warnings.simplefilter('ignore')

    def sample_ddpg_params(self,trial:optuna.Trial):
        # Size of the replay buffer
        buffer_size = trial.suggest_categorical("buffer_size", [int(1e4), int(1e5), int(1e6)])
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])
        
        return {"buffer_size": buffer_size,
                "learning_rate":learning_rate,
                "batch_size":batch_size}
    #Main method
    # Calculates Trade Performance for Objective
    # Called from objective method
    # Returns selected trade perf metric(s)
    # Requires actions and associated prices

    def calc_trade_perf_metric(self,df_actions, 
                            df_prices_trade,
                            tp_metric,
                            dbg=False):
    
        df_actions_p, df_prices_p, tics = self.prep_data(df_actions.copy(),
                                                    df_prices_trade.copy())
        # actions predicted by trained model on trade data
        df_actions_p.to_csv('df_actions.csv') 

        
        # Confirms that actions, prices and tics are consistent
        df_actions_s, df_prices_s, tics_prtfl =         self.sync_tickers(df_actions_p.copy(),df_prices_p.copy(),tics)
        
        # copy to ensure that tics from portfolio remains unchanged
        tics = tics_prtfl.copy()
        
        # Analysis is performed on each portfolio ticker
        perf_data= self.collect_performance_data(df_actions_s, df_prices_s, tics)
        # profit/loss for each ticker
        pnl_all = self.calc_pnl_all(perf_data, tics)
        # values for trade performance metrics
        perf_results = self.calc_trade_perf(pnl_all)
        df = pd.DataFrame.from_dict(perf_results, orient='index')
        
        # calculate and return trade metric value as objective
        m = self.calc_trade_metric(df,tp_metric)
        print(f'Ratio Avg Win/Avg Loss: {m}')
        k = str(len(tpm_hist)+1)
        # save metric value
        tpm_hist[k] = m
        return m

    # Supporting methods
    def calc_trade_metric(self,df,metric='avgwl'):
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


    def prep_data(self,df_actions,
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

    def sync_tickers(self,df_actions,df_tickers_p,tickers):
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
        df_actions, df_tickers_p = self.link_prices_actions(df_actions,df_tickers_p)
        # identify any DOW components not in portfolio
        t_not_in_a = [t for t in tickers if t + '_a' not in list(df_actions.columns)]
    
        # remove t_not_in_a from df_tickers_p
        drop_cols = [t + '_p' for t in t_not_in_a]
        df_tickers_p.drop(columns=drop_cols,inplace=True)
        
        # Tickers in portfolio
        tickers_prtfl = [c.split('_')[0] for c in df_actions.columns]
        return df_actions,df_tickers_p, tickers_prtfl

    def collect_performance_data(self,dfa,dfp,tics, dbg=False):
        
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


    def calc_pnl_all(self,perf_dict, tics_all):
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
                pnl = self.complete_calc_buyonly(acts, prices, init_values)
                pnl_all[tic] = pnl
                continue
            # copy acts: acts_rev will be revised based on closing/reducing init positions
            pnl_all = self.execute_position_sales(tic,acts,prices,args_s,pnl_all)

        return pnl_all


    def complete_calc_buyonly(self,actions, prices, init_values):
        # calculate final pnl for each ticker assuming no sales
        fnl_price = prices[-1]
        final_values = np.multiply(fnl_price, actions)
        pnl = np.subtract(final_values, init_values)
        return pnl


    def execute_position_sales(self,tic,acts,prices,args_s,pnl_all):
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
        
        pnl_op = self.calc_pnl_for_open_positions(acts_rev, prices)
        #pnl_op is list
        # add pnl_op results (if any) to pnl_t (both lists)
        pnl_t.extend(pnl_op)
        #print(f'Total pnl for {tic}: {np.sum(pnl_t)}')
        pnl_all[tic] = np.array(pnl_t)
        return pnl_all


    def calc_pnl_for_open_positions(self,acts,prices):
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


    def calc_trade_perf(self,pnl_d):
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


    def train(self,start_date, end_date, ticker_list, data_source, time_interval, 
            technical_indicator_list, drl_lib, env, model_name, if_vix=True,
            **kwargs):
        
        #process data using unified data processor
        DP = DataProcessor(data_source, **kwargs)
        downloadData = DP.download_data(ticker_list,
                                                            start_date,
                                                            end_date, 
                                                            time_interval)
        data = DP.clean_data(downloadData)
        data = DP.add_technical_indicator(data, technical_indicator_list)
        data = DP.add_turbulence(data)
        if if_vix:
            data = DP.add_vix(data)
        
        date_array,high_array,low_array, price_array, tech_array, turbulence_array = DP.df_to_array_new(data,if_vix)
        data_config = {'date_array': date_array,
                        'high_array':high_array,
                        'low_array':low_array,
                    'price_array': price_array,
                    'tech_array': tech_array,
                    'turbulence_array': turbulence_array}

        #build environment using processed data
        if(self.fl_model_name == 'multiple'):
            env = CryptoEnv
            env_instance = env(config=data_config)
        elif(self.fl_model_name == 'advance'):
            env = AdvCryptoEnv
            env_instance = env('data',52,721,data_config,1,1000000,0.01,0.01,0.99,None,False,True,'P',model_name,False,False)
        else:
            raise ValueError("env is NOT supported. Please check.")

        #read parameters and load agents
        current_working_dir = kwargs.get('current_working_dir','./modal/' + self.fl_model_name+"_"+str(model_name))

        if drl_lib == 'elegantrl':
            break_step = kwargs.get('break_step', 1e6)
            erl_params = kwargs.get('erl_params')

            agent = DRLAgent_erl(env = env,
                                price_array = price_array,
                                tech_array=tech_array,
                                turbulence_array=turbulence_array)

            #model = agent.get_model(model_name, model_kwargs = erl_params)

            #trained_model = agent.train_model(model=model, 
            #                                cwd=current_working_dir,
            #                                total_timesteps=break_step)
        
        elif drl_lib == 'stable_baselines3':
            total_timesteps = kwargs.get('total_timesteps', 1e5)
            agent_params = kwargs.get('agent_params')

            agent = DRLAgent_sb3(env = env_instance)

            #model = agent.get_model(model_name, model_kwargs = agent_params)
            #trained_model = agent.train_model(model=model, 
            #                       tb_log_name=model_name,
            #                        total_timesteps=total_timesteps)
            #print('Training finished!')
            #trained_model.save(current_working_dir)
            #print('Trained model saved in ' + str(current_working_dir))
        else:
            raise ValueError('DRL library input is NOT supported. Please check.')
        return agent,env_instance

    def trade(self,start_date, end_date, ticker_list, data_source, time_interval,
                technical_indicator_list, drl_lib, env, model_name, if_vix=True,
                **kwargs):
    
        #process data using unified data processor
        DP = DataProcessor(data_source, **kwargs)
        downloadData = DP.download_data(ticker_list,
                                                            start_date,
                                                            end_date, 
                                                            time_interval)
        data = DP.clean_data(downloadData)
        data = DP.add_technical_indicator(data, technical_indicator_list)
        data = DP.add_turbulence(data)
        if if_vix:
            data = DP.add_vix(data)
        
        date_array,high_array,low_array, price_array, tech_array, turbulence_array = DP.df_to_array_new(data,if_vix)
        data_config = {'date_array': date_array,
                        'high_array':high_array,
                        'low_array':low_array,
                    'price_array': price_array,
                    'tech_array': tech_array,
                    'turbulence_array': turbulence_array}
        return data_config

def objective(trial:optuna.Trial):
    TRAIN_START_DATE = '2022-01-01'
    TRAIN_END_DATE = '2022-08-31'

    TEST_START_DATE = '2022-09-01'
    TEST_END_DATE = '2022-09-30'

    DRL_LIB = 'stable_baselines3' #'elegantrl'
    
    #fl_model_names = ['multiple','advance']
    #fl_model_names = ['multiple','advance']
    #rl_model_names = ['A2C','DDPG','PPO','SAC','TD3','DQN']
    #rl_model_names = ['A2C','PPO']
    #for fl_model_name in fl_model_names:
    fl_model_name = 'advance'
    if(fl_model_name == 'multiple'):
        env = CryptoEnv
    elif(fl_model_name == 'advance'):
        env = AdvCryptoEnv
    else:
        raise ValueError("env is NOT supported. Please check.")
        
    #for rl_model_name in rl_model_names:
    rl_model_name = "DDPG"
    CURRENT_WORKING_DIR = './modal/'+fl_model_name+"_"+rl_model_name.lower()
    if(rl_model_name == 'A2C'):
        env_kwargs = {
            "API_KEY": "1ddcbec72bef777aaee9343272ec1467", 
            "API_SECRET": "dc42d89bed18b4009c9c60a2f6b45fd41daa86bf", 
            "API_BASE_URL": "https://paper-api.alpaca.markets",
            "rllib_params": RLlib_PARAMS,
            "agent_params": A2C_PARAMS,
        }
    elif(rl_model_name == 'DDPG'):
        env_kwargs = {
            "API_KEY": "1ddcbec72bef777aaee9343272ec1467", 
            "API_SECRET": "dc42d89bed18b4009c9c60a2f6b45fd41daa86bf", 
            "API_BASE_URL": "https://paper-api.alpaca.markets",
            "rllib_params": RLlib_PARAMS,
            "agent_params": DDPG_PARAMS,
        }
    elif(rl_model_name == 'PPO'):
        env_kwargs = {
            "API_KEY": "1ddcbec72bef777aaee9343272ec1467", 
            "API_SECRET": "dc42d89bed18b4009c9c60a2f6b45fd41daa86bf", 
            "API_BASE_URL": "https://paper-api.alpaca.markets",
            "rllib_params": RLlib_PARAMS,
            "agent_params": PPO_PARAMS,
        }
    elif(rl_model_name == 'SAC'):
        env_kwargs = {
            "API_KEY": "1ddcbec72bef777aaee9343272ec1467", 
            "API_SECRET": "dc42d89bed18b4009c9c60a2f6b45fd41daa86bf", 
            "API_BASE_URL": "https://paper-api.alpaca.markets",
            "rllib_params": RLlib_PARAMS,
            "agent_params": SAC_PARAMS,
        }
    elif(rl_model_name == 'TD3'):
        env_kwargs = {
            "API_KEY": "1ddcbec72bef777aaee9343272ec1467", 
            "API_SECRET": "dc42d89bed18b4009c9c60a2f6b45fd41daa86bf", 
            "API_BASE_URL": "https://paper-api.alpaca.markets",
            "rllib_params": RLlib_PARAMS,
            "agent_params": TD3_PARAMS,
        }
    elif(rl_model_name == 'DQN'):
        env_kwargs = {
            "API_KEY": "1ddcbec72bef777aaee9343272ec1467", 
            "API_SECRET": "dc42d89bed18b4009c9c60a2f6b45fd41daa86bf", 
            "API_BASE_URL": "https://paper-api.alpaca.markets",
            "rllib_params": RLlib_PARAMS,
            "agent_params": DQN_PARAMS,
        }
    else:
        raise ValueError("rl_model is NOT supported. Please check.")

    #レース訓練
    optunaForCryptoEnv = OptunaForCryptoEnv(fl_model_name)
    #ポリシー訓練
    agent,e_trade_gym = optunaForCryptoEnv.train(start_date=TRAIN_START_DATE, 
        end_date=TRAIN_END_DATE,
        ticker_list=TICKER_LIST, 
        data_source='yahoofinance',
        time_interval='1D', 
        technical_indicator_list=INDICATORS,
        drl_lib=DRL_LIB, 
        env=env, 
        model_name=rl_model_name.lower(), 
        current_working_dir=CURRENT_WORKING_DIR,
        erl_params=ERL_PARAMS,
        break_step=5e4,
        if_vix=False,
        **env_kwargs
        )

    #Trial will suggest a set of hyperparamters from the specified range

    # Optional to optimize larger set of parameters
    # hyperparameters = sample_ddpg_params_all(trial)

    # Optimize buffer size, batch size, learning rate
    hyperparameters = optunaForCryptoEnv.sample_ddpg_params(trial)
    #print(f'Hyperparameters from objective: {hyperparameters.keys()}')
    policy_kwargs = None  # default
    if 'policy_kwargs' in hyperparameters.keys():
        policy_kwargs = hyperparameters['policy_kwargs']
        del hyperparameters['policy_kwargs']
    #print(f'Policy keyword arguments {policy_kwargs}')
    model = agent.get_model(rl_model_name.lower(),
                                policy_kwargs = policy_kwargs,
                                model_kwargs = hyperparameters )
    #You can increase it for better comparison
    trained_modal = agent.train_model(model=model,
                                    tb_log_name=rl_model_name.lower(),
                                    total_timesteps=total_timesteps)
    trained_modal.save('models/ddpg_{}.pth'.format(trial.number))
    clear_output(wait=True)
    #For the given hyperparamters, determine the account value in the trading period
    df_account_value, df_actions = DRLAgent.DRL_prediction(
    model=trained_modal, 
    environment = e_trade_gym)

    # Calculate trade performance metric
    # Currently ratio of average win and loss market values
    trade = optunaForCryptoEnv.trade(start_date = TEST_START_DATE, 
            end_date = TEST_END_DATE,
            ticker_list = TICKER_LIST, 
            data_source = 'yahoofinance',
            time_interval= '1D', 
            technical_indicator_list= INDICATORS,
            drl_lib=DRL_LIB, 
            env=env, 
            model_name=rl_model_name.lower(), 
            current_working_dir=CURRENT_WORKING_DIR, 
            net_dimension = 2**9, 
            if_vix=False
            )
    tpm = optunaForCryptoEnv.calc_trade_perf_metric(df_actions,trade,tp_metric)
    return tpm

# 動作確認
if __name__ == '__main__':
    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(study_name="ddpg_study",direction='maximize',
                                sampler = sampler, pruner=optuna.pruners.HyperbandPruner())

    logging_callback = LoggingCallback(threshold=lc_threshold,
                                    patience=lc_patience,
                                    trial_number=lc_trial_number)
    #You can increase the n_trials for a better search space scanning
    study.optimize(objective, n_trials=n_trials,catch=(ValueError,),callbacks=[logging_callback])
        
    joblib.dump(study, "final_ddpg_study__.pkl")
    print('Hyperparameters after tuning',study.best_params)
    print('Hyperparameters before tuning',config.DDPG_PARAMS)
