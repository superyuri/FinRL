
from re import A
import numpy as np
import pandas as pd

from gym.utils import seeding
import gym
from gym import spaces
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common import logger

class AdvCryptoEnv(gym.Env):  # custom env

    def __init__(self, 
                path,
                state_space,
                action_space,
                config, 
                lookback=1, 
                initial_amount=1e6, 
                buy_cost_pct=1e-2, 
                sell_cost_pct=1e-2, 
                gamma=0.99,
                turbulence_threshold=None,
                make_plots = False, 
                initial=False,
                prefix='L',
                modal_name='PPO',
                make_csv = False,
                is_real = False
                ):

        #get data
        self.path = path

        self.lookback = lookback
        self.initial_total_asset = initial_amount
        self.initial_cash = initial_amount
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.gamma = gamma
        self.prices_array = config['prices_array']
        self.date_array = self.prices_array[self.prices_array.tic == config['tic']]['time'].values
        self.price_array = config['price_array']
        self.tech_array = config['tech_array']
        self.turbulence_array = config['turbulence_array']
        '''env information'''
        self.env_name = 'AdvCryptoEnv'
        self.crypto_num = self.price_array.shape[1]
        self.max_step = self.price_array.shape[0] - lookback - 1

        self.initial_amount = initial_amount
        self.state_space = state_space
        self.hmax = action_space
        self.prefix = prefix
        self.modal_name = modal_name
        self.action_space = spaces.Discrete(self.hmax)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape = (self.state_space,))
        self.turbulence_threshold = turbulence_threshold
        self.make_plots = make_plots
        self.initial = initial
        self.state = self._initiate_state()
        self._seed()
        self.episode = 0
        self.turbulence = 0
        self.make_csv = make_csv
        self.is_real = is_real
        self.actions = self.createActions()
        self.legal = self.legal_actions()

    #action param1 param2 param3 param4
    #action 1買:2 売
    #tic 1-10 ['BTC-JPY','ETH-JPY','BCH-JPY','LTC-JPY','XRP-JPY', 'XEM-JPY','XLM-JPY', 'BAT-JPY','OMG-JPY','XTZ-JPY']
    #損失止め 1-3 [10%,30%,50%]
    #利益確保 1-4 [20%,40%,60%,80%]
    #資本量 1-3 [10%,20%,40%]
    def _buy_ticket_auto(self,available_amount):
        prices = self.prices_array[self.index]
        for idx in range(len(self.trades)):#[action, para1-1,self.price_array[para1-1],self.stocks[para1-1],loss_price,win_price]
            action = self.trades[idx][0]
            tic = self.trades[idx][1]
            volume = self.trades[idx][3]
            loss_price = self.trades[idx][4]
            win_price = self.trades[idx][5]
            high_price = prices[tic]["high"]
            low_price = prices[tic]["low"]
            if action == 1 :#買
                if  loss_price<=low_price :
                    #損失
                    self.trades[idx][0] = 0
                    self.stocks[tic] -= volume
                    available_amount +=volume*loss_price(1-self.sell_cost_pct)
                elif win_price <= high_price :
                    #利益確保
                    self.trades[idx][0] = 0
                    self.stocks[tic] -= volume
                    available_amount +=volume*win_price(1-self.sell_cost_pct)
            elif action == 2 :#売
                if  loss_price<=high_price :
                    #損失
                    self.trades[idx][0] = 0
                    self.stocks[tic] += volume
                    available_amount +=volume*loss_price(1-self.sell_cost_pct)
                elif win_price <= low_price :
                    #利益確保
                    self.stocks[tic] += volume
                    available_amount +=volume*loss_price(1-self.sell_cost_pct)                
        return available_amount

    def _buy_ticket_new(self,available_amount,action,para1,para2,para3,para4):
        price = self.price_array[self.index]
        if para4 == 1: 
            use_amount = self.initial_amount * 0.1
        elif para4 == 2:
            use_amount = self.initial_amount * 0.2
        elif para4 == 3:
            use_amount = self.initial_amount * 0.4
        else :
            raise ValueError("para4 is NOT supported. Please check.")

        if action == 1 :#買

            if para1 > 0 and para1 < 11:
                use_amount = min(available_amount,use_amount)
                volume = use_amount/self.price_array[para1-1]*(1-self.sell_cost_pct)
                self.stocks[para1-1] += volume
                if para2 == 1:
                    loss_price = self.price_array[para1-1]*0.9
                elif para2 == 2:
                    loss_price = self.price_array[para1-1]*0.7
                elif para2 == 3:
                    loss_price = self.price_array[para1-1]*0.5
                else :
                    raise ValueError("para2 is NOT supported. Please check.")
                if para3 == 1:
                    win_price = self.price_array[para1-1]*1.2
                elif para3 == 2:
                    win_price = self.price_array[para1-1]*1.4
                elif para3 == 3:
                    win_price = self.price_array[para1-1]*1.6
                elif para3 == 4:
                    win_price = self.price_array[para1-1]*1.8
                else :
                    raise ValueError("para3 is NOT supported. Please check.")
                available_amount -= use_amount
                self.trades += [action, para1-1,self.price_array[para1-1],volume,loss_price,win_price]
        elif action == 2 :#売
            if para1 > 0 and para1 < 11:
                volume = use_amount/self.price_array[para1-1]
                self.stocks[para1-1] -= volume
                use_amount = self.price_array[para1-1] * volume*(1-self.sell_cost_pct)
                if para2 == 1:
                    loss_price = self.price_array[para1-1]*0.9
                elif para2 == 2:
                    loss_price = self.price_array[para1-1]*0.7
                elif para2 == 3:
                    loss_price = self.price_array[para1-1]*0.5
                else :
                    raise ValueError("para2 is NOT supported. Please check.")
                if para3 == 1:
                    win_price = self.price_array[para1-1]*1.2
                elif para3 == 2:
                    win_price = self.price_array[para1-1]*1.4
                elif para3 == 3:
                    win_price = self.price_array[para1-1]*1.6
                elif para3 == 4:
                    win_price = self.price_array[para1-1]*1.8
                else :
                    raise ValueError("para3 is NOT supported. Please check.")

                available_amount += use_amount
                self.trades += [action, para1-1,self.price_array[para1-1],volume,loss_price,win_price]
        return available_amount

    def _calc_reward(self,available_amount):
        amunt = available_amount
        price = self.price_array[self.index]
        for idx in range(len(self.trades)):#[action, para1-1,self.price_array[para1-1],volume,loss_price,win_price]
            action = self.trades[idx][0]
            tic = self.trades[idx][1]
            volume = self.trades[idx][3]
            if action == 1 :#買
                available_amount += volume*price[tic]*(1-self.sell_cost_pct)
            elif action == 2 :#売
                available_amount -= volume*price[tic]*(1-self.buy_cost_pct)
        reward = amunt - available_amount
        return reward

    def _make_plot(self):
        df_asset_value = self.save_asset_memory()
        filename = '../data/' + self.path + '/{}_asset_memory_{}_{}.png'.format(self.prefix,self.episode,self.modal_name)
        plt.plot(df_asset_value.race,df_asset_value.account_value,'r')
        plt.savefig(filename)
        plt.close()

    def _make_csv(self):
        df_action_value = self.save_actions_memory()
        filename = '../data/' + self.path + '/{}_actions_memory_{}_{}.csv'.format(self.prefix,self.episode,self.modal_name)
        df_action_value.to_csv(filename , mode = 'w', header = True, index = False)

    def step(self, actions):
        if self.index >= len(self.race_memory)-1:
            self.terminal = True

        if self.terminal:
            if self.make_plots:
                self._make_plot()
            if self.make_csv:
                self._make_csv()
            return np.array(self.state), self.reward, self.terminal, {}
        else :
            available_amount = self.state[0]
            if  available_amount > 0:
                actions = self.getValidAction(actions, 0)
                if not self.is_real:
                    self.reward = 0
                    for i in range(0, len(actions)//5, 1):
                        available_amount = self._buy_ticket_auto(available_amount)
                        available_amount = self._buy_ticket_new(available_amount, actions[5*i], actions[5*i+1], actions[5*i+2], actions[5*i+3], actions[5*i+4])
                        reward = self._calc_reward(available_amount)
                        self.reward = self.reward*self.gamma + reward
                    self.state = self._update_state(actions, self.reward, available_amount, self.index+1)
                else:
                    self.state = self._update_state(actions, self.reward, available_amount, self.index+1)

            else:
                self.terminal = True
                if self.make_plots:
                    self._make_plot()
                if self.make_csv:
                    self._make_csv()
                
        return np.array(self.state), self.reward, self.terminal, {}


    def getValidAction(self, action, size):
        if action in self.legal:
            return self.actions[action]
        return self.actions[0]

    def legal_actions(self):
        legal = list(range(721))
        return legal

    #action param1 param2 param3 param4
    #action 1買:2 売
    #tic 1-10 ['BTC-JPY','ETH-JPY','BCH-JPY','LTC-JPY','XRP-JPY', 'XEM-JPY','XLM-JPY', 'BAT-JPY','OMG-JPY','XTZ-JPY']
    #損失止め 1-3 [10%,30%,50%]
    #利益確保 1-4 [20%,40%,60%,80%]
    #資本量 1-3 [10%,20%,40%]
    def createActions(self):
        actions = [\
            [0,0,0,0,0],\
            [1,1,1,1,1],\
            [1,2,1,1,1],\
            [1,3,1,1,1],\
            [1,4,1,1,1],\
            [1,5,1,1,1],\
            [1,6,1,1,1],\
            [1,7,1,1,1],\
            [1,8,1,1,1],\
            [1,9,1,1,1],\
            [1,10,1,1,1],\
            [1,1,2,1,1],\
            [1,2,2,1,1],\
            [1,3,2,1,1],\
            [1,4,2,1,1],\
            [1,5,2,1,1],\
            [1,6,2,1,1],\
            [1,7,2,1,1],\
            [1,8,2,1,1],\
            [1,9,2,1,1],\
            [1,10,2,1,1],\
            [1,1,3,1,1],\
            [1,2,3,1,1],\
            [1,3,3,1,1],\
            [1,4,3,1,1],\
            [1,5,3,1,1],\
            [1,6,3,1,1],\
            [1,7,3,1,1],\
            [1,8,3,1,1],\
            [1,9,3,1,1],\
            [1,10,3,1,1],\
            [1,1,1,2,1],\
            [1,2,1,2,1],\
            [1,3,1,2,1],\
            [1,4,1,2,1],\
            [1,5,1,2,1],\
            [1,6,1,2,1],\
            [1,7,1,2,1],\
            [1,8,1,2,1],\
            [1,9,1,2,1],\
            [1,10,1,2,1],\
            [1,1,2,2,1],\
            [1,2,2,2,1],\
            [1,3,2,2,1],\
            [1,4,2,2,1],\
            [1,5,2,2,1],\
            [1,6,2,2,1],\
            [1,7,2,2,1],\
            [1,8,2,2,1],\
            [1,9,2,2,1],\
            [1,10,2,2,1],\
            [1,1,3,2,1],\
            [1,2,3,2,1],\
            [1,3,3,2,1],\
            [1,4,3,2,1],\
            [1,5,3,2,1],\
            [1,6,3,2,1],\
            [1,7,3,2,1],\
            [1,8,3,2,1],\
            [1,9,3,2,1],\
            [1,10,3,2,1],\
            [1,1,1,3,1],\
            [1,2,1,3,1],\
            [1,3,1,3,1],\
            [1,4,1,3,1],\
            [1,5,1,3,1],\
            [1,6,1,3,1],\
            [1,7,1,3,1],\
            [1,8,1,3,1],\
            [1,9,1,3,1],\
            [1,10,1,3,1],\
            [1,1,2,3,1],\
            [1,2,2,3,1],\
            [1,3,2,3,1],\
            [1,4,2,3,1],\
            [1,5,2,3,1],\
            [1,6,2,3,1],\
            [1,7,2,3,1],\
            [1,8,2,3,1],\
            [1,9,2,3,1],\
            [1,10,2,3,1],\
            [1,1,3,3,1],\
            [1,2,3,3,1],\
            [1,3,3,3,1],\
            [1,4,3,3,1],\
            [1,5,3,3,1],\
            [1,6,3,3,1],\
            [1,7,3,3,1],\
            [1,8,3,3,1],\
            [1,9,3,3,1],\
            [1,10,3,3,1],\
            [1,1,1,4,1],\
            [1,2,1,4,1],\
            [1,3,1,4,1],\
            [1,4,1,4,1],\
            [1,5,1,4,1],\
            [1,6,1,4,1],\
            [1,7,1,4,1],\
            [1,8,1,4,1],\
            [1,9,1,4,1],\
            [1,10,1,4,1],\
            [1,1,2,4,1],\
            [1,2,2,4,1],\
            [1,3,2,4,1],\
            [1,4,2,4,1],\
            [1,5,2,4,1],\
            [1,6,2,4,1],\
            [1,7,2,4,1],\
            [1,8,2,4,1],\
            [1,9,2,4,1],\
            [1,10,2,4,1],\
            [1,1,3,4,1],\
            [1,2,3,4,1],\
            [1,3,3,4,1],\
            [1,4,3,4,1],\
            [1,5,3,4,1],\
            [1,6,3,4,1],\
            [1,7,3,4,1],\
            [1,8,3,4,1],\
            [1,9,3,4,1],\
            [1,10,3,4,1],\
            [1,1,1,1,2],\
            [1,2,1,1,2],\
            [1,3,1,1,2],\
            [1,4,1,1,2],\
            [1,5,1,1,2],\
            [1,6,1,1,2],\
            [1,7,1,1,2],\
            [1,8,1,1,2],\
            [1,9,1,1,2],\
            [1,10,1,1,2],\
            [1,1,2,1,2],\
            [1,2,2,1,2],\
            [1,3,2,1,2],\
            [1,4,2,1,2],\
            [1,5,2,1,2],\
            [1,6,2,1,2],\
            [1,7,2,1,2],\
            [1,8,2,1,2],\
            [1,9,2,1,2],\
            [1,10,2,1,2],\
            [1,1,3,1,2],\
            [1,2,3,1,2],\
            [1,3,3,1,2],\
            [1,4,3,1,2],\
            [1,5,3,1,2],\
            [1,6,3,1,2],\
            [1,7,3,1,2],\
            [1,8,3,1,2],\
            [1,9,3,1,2],\
            [1,10,3,1,2],\
            [1,1,1,2,2],\
            [1,2,1,2,2],\
            [1,3,1,2,2],\
            [1,4,1,2,2],\
            [1,5,1,2,2],\
            [1,6,1,2,2],\
            [1,7,1,2,2],\
            [1,8,1,2,2],\
            [1,9,1,2,2],\
            [1,10,1,2,2],\
            [1,1,2,2,2],\
            [1,2,2,2,2],\
            [1,3,2,2,2],\
            [1,4,2,2,2],\
            [1,5,2,2,2],\
            [1,6,2,2,2],\
            [1,7,2,2,2],\
            [1,8,2,2,2],\
            [1,9,2,2,2],\
            [1,10,2,2,2],\
            [1,1,3,2,2],\
            [1,2,3,2,2],\
            [1,3,3,2,2],\
            [1,4,3,2,2],\
            [1,5,3,2,2],\
            [1,6,3,2,2],\
            [1,7,3,2,2],\
            [1,8,3,2,2],\
            [1,9,3,2,2],\
            [1,10,3,2,2],\
            [1,1,1,3,2],\
            [1,2,1,3,2],\
            [1,3,1,3,2],\
            [1,4,1,3,2],\
            [1,5,1,3,2],\
            [1,6,1,3,2],\
            [1,7,1,3,2],\
            [1,8,1,3,2],\
            [1,9,1,3,2],\
            [1,10,1,3,2],\
            [1,1,2,3,2],\
            [1,2,2,3,2],\
            [1,3,2,3,2],\
            [1,4,2,3,2],\
            [1,5,2,3,2],\
            [1,6,2,3,2],\
            [1,7,2,3,2],\
            [1,8,2,3,2],\
            [1,9,2,3,2],\
            [1,10,2,3,2],\
            [1,1,3,3,2],\
            [1,2,3,3,2],\
            [1,3,3,3,2],\
            [1,4,3,3,2],\
            [1,5,3,3,2],\
            [1,6,3,3,2],\
            [1,7,3,3,2],\
            [1,8,3,3,2],\
            [1,9,3,3,2],\
            [1,10,3,3,2],\
            [1,1,1,4,2],\
            [1,2,1,4,2],\
            [1,3,1,4,2],\
            [1,4,1,4,2],\
            [1,5,1,4,2],\
            [1,6,1,4,2],\
            [1,7,1,4,2],\
            [1,8,1,4,2],\
            [1,9,1,4,2],\
            [1,10,1,4,2],\
            [1,1,2,4,2],\
            [1,2,2,4,2],\
            [1,3,2,4,2],\
            [1,4,2,4,2],\
            [1,5,2,4,2],\
            [1,6,2,4,2],\
            [1,7,2,4,2],\
            [1,8,2,4,2],\
            [1,9,2,4,2],\
            [1,10,2,4,2],\
            [1,1,3,4,2],\
            [1,2,3,4,2],\
            [1,3,3,4,2],\
            [1,4,3,4,2],\
            [1,5,3,4,2],\
            [1,6,3,4,2],\
            [1,7,3,4,2],\
            [1,8,3,4,2],\
            [1,9,3,4,2],\
            [1,10,3,4,2],\
            [1,1,1,1,3],\
            [1,2,1,1,3],\
            [1,3,1,1,3],\
            [1,4,1,1,3],\
            [1,5,1,1,3],\
            [1,6,1,1,3],\
            [1,7,1,1,3],\
            [1,8,1,1,3],\
            [1,9,1,1,3],\
            [1,10,1,1,3],\
            [1,1,2,1,3],\
            [1,2,2,1,3],\
            [1,3,2,1,3],\
            [1,4,2,1,3],\
            [1,5,2,1,3],\
            [1,6,2,1,3],\
            [1,7,2,1,3],\
            [1,8,2,1,3],\
            [1,9,2,1,3],\
            [1,10,2,1,3],\
            [1,1,3,1,3],\
            [1,2,3,1,3],\
            [1,3,3,1,3],\
            [1,4,3,1,3],\
            [1,5,3,1,3],\
            [1,6,3,1,3],\
            [1,7,3,1,3],\
            [1,8,3,1,3],\
            [1,9,3,1,3],\
            [1,10,3,1,3],\
            [1,1,1,2,3],\
            [1,2,1,2,3],\
            [1,3,1,2,3],\
            [1,4,1,2,3],\
            [1,5,1,2,3],\
            [1,6,1,2,3],\
            [1,7,1,2,3],\
            [1,8,1,2,3],\
            [1,9,1,2,3],\
            [1,10,1,2,3],\
            [1,1,2,2,3],\
            [1,2,2,2,3],\
            [1,3,2,2,3],\
            [1,4,2,2,3],\
            [1,5,2,2,3],\
            [1,6,2,2,3],\
            [1,7,2,2,3],\
            [1,8,2,2,3],\
            [1,9,2,2,3],\
            [1,10,2,2,3],\
            [1,1,3,2,3],\
            [1,2,3,2,3],\
            [1,3,3,2,3],\
            [1,4,3,2,3],\
            [1,5,3,2,3],\
            [1,6,3,2,3],\
            [1,7,3,2,3],\
            [1,8,3,2,3],\
            [1,9,3,2,3],\
            [1,10,3,2,3],\
            [1,1,1,3,3],\
            [1,2,1,3,3],\
            [1,3,1,3,3],\
            [1,4,1,3,3],\
            [1,5,1,3,3],\
            [1,6,1,3,3],\
            [1,7,1,3,3],\
            [1,8,1,3,3],\
            [1,9,1,3,3],\
            [1,10,1,3,3],\
            [1,1,2,3,3],\
            [1,2,2,3,3],\
            [1,3,2,3,3],\
            [1,4,2,3,3],\
            [1,5,2,3,3],\
            [1,6,2,3,3],\
            [1,7,2,3,3],\
            [1,8,2,3,3],\
            [1,9,2,3,3],\
            [1,10,2,3,3],\
            [1,1,3,3,3],\
            [1,2,3,3,3],\
            [1,3,3,3,3],\
            [1,4,3,3,3],\
            [1,5,3,3,3],\
            [1,6,3,3,3],\
            [1,7,3,3,3],\
            [1,8,3,3,3],\
            [1,9,3,3,3],\
            [1,10,3,3,3],\
            [1,1,1,4,3],\
            [1,2,1,4,3],\
            [1,3,1,4,3],\
            [1,4,1,4,3],\
            [1,5,1,4,3],\
            [1,6,1,4,3],\
            [1,7,1,4,3],\
            [1,8,1,4,3],\
            [1,9,1,4,3],\
            [1,10,1,4,3],\
            [1,1,2,4,3],\
            [1,2,2,4,3],\
            [1,3,2,4,3],\
            [1,4,2,4,3],\
            [1,5,2,4,3],\
            [1,6,2,4,3],\
            [1,7,2,4,3],\
            [1,8,2,4,3],\
            [1,9,2,4,3],\
            [1,10,2,4,3],\
            [1,1,3,4,3],\
            [1,2,3,4,3],\
            [1,3,3,4,3],\
            [1,4,3,4,3],\
            [1,5,3,4,3],\
            [1,6,3,4,3],\
            [1,7,3,4,3],\
            [1,8,3,4,3],\
            [1,9,3,4,3],\
            [1,10,3,4,3],\
            [2,1,1,1,1],\
            [2,2,1,1,1],\
            [2,3,1,1,1],\
            [2,4,1,1,1],\
            [2,5,1,1,1],\
            [2,6,1,1,1],\
            [2,7,1,1,1],\
            [2,8,1,1,1],\
            [2,9,1,1,1],\
            [2,10,1,1,1],\
            [2,1,2,1,1],\
            [2,2,2,1,1],\
            [2,3,2,1,1],\
            [2,4,2,1,1],\
            [2,5,2,1,1],\
            [2,6,2,1,1],\
            [2,7,2,1,1],\
            [2,8,2,1,1],\
            [2,9,2,1,1],\
            [2,10,2,1,1],\
            [2,1,3,1,1],\
            [2,2,3,1,1],\
            [2,3,3,1,1],\
            [2,4,3,1,1],\
            [2,5,3,1,1],\
            [2,6,3,1,1],\
            [2,7,3,1,1],\
            [2,8,3,1,1],\
            [2,9,3,1,1],\
            [2,10,3,1,1],\
            [2,1,1,2,1],\
            [2,2,1,2,1],\
            [2,3,1,2,1],\
            [2,4,1,2,1],\
            [2,5,1,2,1],\
            [2,6,1,2,1],\
            [2,7,1,2,1],\
            [2,8,1,2,1],\
            [2,9,1,2,1],\
            [2,10,1,2,1],\
            [2,1,2,2,1],\
            [2,2,2,2,1],\
            [2,3,2,2,1],\
            [2,4,2,2,1],\
            [2,5,2,2,1],\
            [2,6,2,2,1],\
            [2,7,2,2,1],\
            [2,8,2,2,1],\
            [2,9,2,2,1],\
            [2,10,2,2,1],\
            [2,1,3,2,1],\
            [2,2,3,2,1],\
            [2,3,3,2,1],\
            [2,4,3,2,1],\
            [2,5,3,2,1],\
            [2,6,3,2,1],\
            [2,7,3,2,1],\
            [2,8,3,2,1],\
            [2,9,3,2,1],\
            [2,10,3,2,1],\
            [2,1,1,3,1],\
            [2,2,1,3,1],\
            [2,3,1,3,1],\
            [2,4,1,3,1],\
            [2,5,1,3,1],\
            [2,6,1,3,1],\
            [2,7,1,3,1],\
            [2,8,1,3,1],\
            [2,9,1,3,1],\
            [2,10,1,3,1],\
            [2,1,2,3,1],\
            [2,2,2,3,1],\
            [2,3,2,3,1],\
            [2,4,2,3,1],\
            [2,5,2,3,1],\
            [2,6,2,3,1],\
            [2,7,2,3,1],\
            [2,8,2,3,1],\
            [2,9,2,3,1],\
            [2,10,2,3,1],\
            [2,1,3,3,1],\
            [2,2,3,3,1],\
            [2,3,3,3,1],\
            [2,4,3,3,1],\
            [2,5,3,3,1],\
            [2,6,3,3,1],\
            [2,7,3,3,1],\
            [2,8,3,3,1],\
            [2,9,3,3,1],\
            [2,10,3,3,1],\
            [2,1,1,4,1],\
            [2,2,1,4,1],\
            [2,3,1,4,1],\
            [2,4,1,4,1],\
            [2,5,1,4,1],\
            [2,6,1,4,1],\
            [2,7,1,4,1],\
            [2,8,1,4,1],\
            [2,9,1,4,1],\
            [2,10,1,4,1],\
            [2,1,2,4,1],\
            [2,2,2,4,1],\
            [2,3,2,4,1],\
            [2,4,2,4,1],\
            [2,5,2,4,1],\
            [2,6,2,4,1],\
            [2,7,2,4,1],\
            [2,8,2,4,1],\
            [2,9,2,4,1],\
            [2,10,2,4,1],\
            [2,1,3,4,1],\
            [2,2,3,4,1],\
            [2,3,3,4,1],\
            [2,4,3,4,1],\
            [2,5,3,4,1],\
            [2,6,3,4,1],\
            [2,7,3,4,1],\
            [2,8,3,4,1],\
            [2,9,3,4,1],\
            [2,10,3,4,1],\
            [2,1,1,1,2],\
            [2,2,1,1,2],\
            [2,3,1,1,2],\
            [2,4,1,1,2],\
            [2,5,1,1,2],\
            [2,6,1,1,2],\
            [2,7,1,1,2],\
            [2,8,1,1,2],\
            [2,9,1,1,2],\
            [2,10,1,1,2],\
            [2,1,2,1,2],\
            [2,2,2,1,2],\
            [2,3,2,1,2],\
            [2,4,2,1,2],\
            [2,5,2,1,2],\
            [2,6,2,1,2],\
            [2,7,2,1,2],\
            [2,8,2,1,2],\
            [2,9,2,1,2],\
            [2,10,2,1,2],\
            [2,1,3,1,2],\
            [2,2,3,1,2],\
            [2,3,3,1,2],\
            [2,4,3,1,2],\
            [2,5,3,1,2],\
            [2,6,3,1,2],\
            [2,7,3,1,2],\
            [2,8,3,1,2],\
            [2,9,3,1,2],\
            [2,10,3,1,2],\
            [2,1,1,2,2],\
            [2,2,1,2,2],\
            [2,3,1,2,2],\
            [2,4,1,2,2],\
            [2,5,1,2,2],\
            [2,6,1,2,2],\
            [2,7,1,2,2],\
            [2,8,1,2,2],\
            [2,9,1,2,2],\
            [2,10,1,2,2],\
            [2,1,2,2,2],\
            [2,2,2,2,2],\
            [2,3,2,2,2],\
            [2,4,2,2,2],\
            [2,5,2,2,2],\
            [2,6,2,2,2],\
            [2,7,2,2,2],\
            [2,8,2,2,2],\
            [2,9,2,2,2],\
            [2,10,2,2,2],\
            [2,1,3,2,2],\
            [2,2,3,2,2],\
            [2,3,3,2,2],\
            [2,4,3,2,2],\
            [2,5,3,2,2],\
            [2,6,3,2,2],\
            [2,7,3,2,2],\
            [2,8,3,2,2],\
            [2,9,3,2,2],\
            [2,10,3,2,2],\
            [2,1,1,3,2],\
            [2,2,1,3,2],\
            [2,3,1,3,2],\
            [2,4,1,3,2],\
            [2,5,1,3,2],\
            [2,6,1,3,2],\
            [2,7,1,3,2],\
            [2,8,1,3,2],\
            [2,9,1,3,2],\
            [2,10,1,3,2],\
            [2,1,2,3,2],\
            [2,2,2,3,2],\
            [2,3,2,3,2],\
            [2,4,2,3,2],\
            [2,5,2,3,2],\
            [2,6,2,3,2],\
            [2,7,2,3,2],\
            [2,8,2,3,2],\
            [2,9,2,3,2],\
            [2,10,2,3,2],\
            [2,1,3,3,2],\
            [2,2,3,3,2],\
            [2,3,3,3,2],\
            [2,4,3,3,2],\
            [2,5,3,3,2],\
            [2,6,3,3,2],\
            [2,7,3,3,2],\
            [2,8,3,3,2],\
            [2,9,3,3,2],\
            [2,10,3,3,2],\
            [2,1,1,4,2],\
            [2,2,1,4,2],\
            [2,3,1,4,2],\
            [2,4,1,4,2],\
            [2,5,1,4,2],\
            [2,6,1,4,2],\
            [2,7,1,4,2],\
            [2,8,1,4,2],\
            [2,9,1,4,2],\
            [2,10,1,4,2],\
            [2,1,2,4,2],\
            [2,2,2,4,2],\
            [2,3,2,4,2],\
            [2,4,2,4,2],\
            [2,5,2,4,2],\
            [2,6,2,4,2],\
            [2,7,2,4,2],\
            [2,8,2,4,2],\
            [2,9,2,4,2],\
            [2,10,2,4,2],\
            [2,1,3,4,2],\
            [2,2,3,4,2],\
            [2,3,3,4,2],\
            [2,4,3,4,2],\
            [2,5,3,4,2],\
            [2,6,3,4,2],\
            [2,7,3,4,2],\
            [2,8,3,4,2],\
            [2,9,3,4,2],\
            [2,10,3,4,2],\
            [2,1,1,1,3],\
            [2,2,1,1,3],\
            [2,3,1,1,3],\
            [2,4,1,1,3],\
            [2,5,1,1,3],\
            [2,6,1,1,3],\
            [2,7,1,1,3],\
            [2,8,1,1,3],\
            [2,9,1,1,3],\
            [2,10,1,1,3],\
            [2,1,2,1,3],\
            [2,2,2,1,3],\
            [2,3,2,1,3],\
            [2,4,2,1,3],\
            [2,5,2,1,3],\
            [2,6,2,1,3],\
            [2,7,2,1,3],\
            [2,8,2,1,3],\
            [2,9,2,1,3],\
            [2,10,2,1,3],\
            [2,1,3,1,3],\
            [2,2,3,1,3],\
            [2,3,3,1,3],\
            [2,4,3,1,3],\
            [2,5,3,1,3],\
            [2,6,3,1,3],\
            [2,7,3,1,3],\
            [2,8,3,1,3],\
            [2,9,3,1,3],\
            [2,10,3,1,3],\
            [2,1,1,2,3],\
            [2,2,1,2,3],\
            [2,3,1,2,3],\
            [2,4,1,2,3],\
            [2,5,1,2,3],\
            [2,6,1,2,3],\
            [2,7,1,2,3],\
            [2,8,1,2,3],\
            [2,9,1,2,3],\
            [2,10,1,2,3],\
            [2,1,2,2,3],\
            [2,2,2,2,3],\
            [2,3,2,2,3],\
            [2,4,2,2,3],\
            [2,5,2,2,3],\
            [2,6,2,2,3],\
            [2,7,2,2,3],\
            [2,8,2,2,3],\
            [2,9,2,2,3],\
            [2,10,2,2,3],\
            [2,1,3,2,3],\
            [2,2,3,2,3],\
            [2,3,3,2,3],\
            [2,4,3,2,3],\
            [2,5,3,2,3],\
            [2,6,3,2,3],\
            [2,7,3,2,3],\
            [2,8,3,2,3],\
            [2,9,3,2,3],\
            [2,10,3,2,3],\
            [2,1,1,3,3],\
            [2,2,1,3,3],\
            [2,3,1,3,3],\
            [2,4,1,3,3],\
            [2,5,1,3,3],\
            [2,6,1,3,3],\
            [2,7,1,3,3],\
            [2,8,1,3,3],\
            [2,9,1,3,3],\
            [2,10,1,3,3],\
            [2,1,2,3,3],\
            [2,2,2,3,3],\
            [2,3,2,3,3],\
            [2,4,2,3,3],\
            [2,5,2,3,3],\
            [2,6,2,3,3],\
            [2,7,2,3,3],\
            [2,8,2,3,3],\
            [2,9,2,3,3],\
            [2,10,2,3,3],\
            [2,1,3,3,3],\
            [2,2,3,3,3],\
            [2,3,3,3,3],\
            [2,4,3,3,3],\
            [2,5,3,3,3],\
            [2,6,3,3,3],\
            [2,7,3,3,3],\
            [2,8,3,3,3],\
            [2,9,3,3,3],\
            [2,10,3,3,3],\
            [2,1,1,4,3],\
            [2,2,1,4,3],\
            [2,3,1,4,3],\
            [2,4,1,4,3],\
            [2,5,1,4,3],\
            [2,6,1,4,3],\
            [2,7,1,4,3],\
            [2,8,1,4,3],\
            [2,9,1,4,3],\
            [2,10,1,4,3],\
            [2,1,2,4,3],\
            [2,2,2,4,3],\
            [2,3,2,4,3],\
            [2,4,2,4,3],\
            [2,5,2,4,3],\
            [2,6,2,4,3],\
            [2,7,2,4,3],\
            [2,8,2,4,3],\
            [2,9,2,4,3],\
            [2,10,2,4,3],\
            [2,1,3,4,3],\
            [2,2,3,4,3],\
            [2,3,3,4,3],\
            [2,4,3,4,3],\
            [2,5,3,4,3],\
            [2,6,3,4,3],\
            [2,7,3,4,3],\
            [2,8,3,4,3],\
            [2,9,3,4,3],\
            [2,10,3,4,3]\
        ]
        return actions

    def reset(self):  
        self.state = self._initiate_state()
        self.episode+=1
        return np.array(self.state)
          
    def render(self, mode='human',close=False):
        return np.array(self.state)

    def _initiate_state(self):
        self.trades = [[]]
        self.asset_memory = [self.initial_amount]
        self.rewards_memory = [0]
        self.actions_memory=[[0]]
        self.reward = 0
        self.index = 0
        self.terminal = False
        self.current_price = self.price_array[self.index]
        self.current_tech = self.tech_array[self.index]
        self.turbulence_array = self.turbulence_array[self.index]
        self.cash = self.initial_cash  # reset()
        self.stocks = np.zeros(self.crypto_num, dtype=np.float32)
        self.total_asset = self.cash

        return self.get_state()

    def get_state(self):
        state =  np.hstack((self.cash, self.stocks * 2 ** -3))
        for i in range(self.lookback):
            tech_i = self.tech_array[self.time-i]
            normalized_tech_i = tech_i * 2 ** -15
            state = np.hstack((state, normalized_tech_i)).astype(np.float32)
            turbulence_i = self.turbulence_array[self.time-i]
            state = np.hstack((state, turbulence_i)).astype(np.float32)
        return state
    
    def _update_state(self, action, reward, amount, index):
        self.asset_memory += [amount]
        self.rewards_memory += [reward]
        self.actions_memory += [action]
        self.index = index
        self.cash = amount
        return self.get_state

    def save_asset_memory(self):
        date_list = self.date_array
        asset_list = self.asset_memory[1:]
        df_asset_value = pd.DataFrame({'race':date_list,'account_value':asset_list})
        return df_asset_value

    def save_actions_memory(self):
        date_list = self.date_array
        action_list = self.actions_memory[1:]
        df_action_value = pd.DataFrame({'race':date_list,'action':action_list})
        return df_action_value

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs

# 動作確認
if __name__ == '__main__':
    #check env
    from stable_baselines3.common.env_checker import check_env
    prices_array = {{'time':[1,2],'open':[0,0,0,0,0,0,0,0,0,0],'high':[0,0,0,0,0,0,0,0,0,0],'low':[0,0,0,0,0,0,0,0,0,0],'close':[0,0,0,0,0,0,0,0,0,0],'volume':[0,0,0,0,0,0,0,0,0,0]}}
    price_array = {'price':[0,0,0,0,0,0,0,0,0,0]}
    tech_array = {'macd':[0,0,0,0,0,0,0,0,0,0],'rsi':[0,0,0,0,0,0,0,0,0,0],'cci':[0,0,0,0,0,0,0,0,0,0],'dx':[0,0,0,0,0,0,0,0,0,0]}
    turbulence_array ={'turbulence':[0,0,0,0,0,0,0,0,0,0]}
    config = {'prices_array':prices_array,'price_array':price_array,'tech_array':tech_array,'turbulence_array':turbulence_array}

    env = AdvCryptoEnv('data',61,721,config,1,1000000,0.01,0.01,0.99,None,True,True,'P','PPO',True,False)
    # It will check your custom environment and output additional warnings if needed
    check_env(env)

