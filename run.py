import torch
import torch.nn as nn
from torch import optim
import numpy as np

import gym
import random

from models import QFunc
from replay_memory import ReplayMemory


# 各パラメータの設定
lr = 1e-3
gamma = 0.95
epsilon = 0.3
batch_size = 32
initial_exploration = 500

# モデルの定義
qf = QFunc()
target_qf = QFunc()
# qfのパラメータをtarget_qfにコピー
target_qf.load_state_dict(qf.state_dict())

# 最適化手法を定義（最適化したいモデルのパラメータ，学習率）
optimizer = optim.Adam(qf.parameters(), lr=lr)
# 誤差関数を定義
criterion = nn.MSELoss()
# リプレイメモリを定義
memory = ReplayMemory()

# 環境を定義
env = gym.make('CartPole-v0')
# 環境・行動の次元数を取得
obs_size = env.observation_space.shape[0]
acs_size = env.action_space.n


total_step = 0
for episode in range(150):
    done = False
    obs = env.reset()
    sum_reward = 0
    step = 0

    while not done:
        ###################
        ### 行動フェーズ ###
        ###################
        
        # episilonの確率で行動をランダムに選択
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = qf.select_action(obs)
        epsilon -= 1e-4
        if epsilon < 0:
            epsilon = 0

        next_obs, reward, done, _ = time_step = env.step(action)
        #env.render() 
        
        terminal = 0
        reward = 0
        if done:
            terminal = 1
            if not step >= 195:
                reward = -1
        sum_reward += reward

        # メモリに追加
        memory.add(obs, action, reward, next_obs, terminal)
        obs = next_obs.copy()
        
        step += 1
        total_step += 1
        if total_step < initial_exploration:
            continue

        ###################
        ### 学習フェーズ ###
        ###################

        # メモリからバッチを取得
        batch = memory.sample()

        # Q値を出力 & 実際にとった行動のindex(batch['acs'])のもののみ抜き出す
        q_value = qf(batch['obs']).gather(1, batch['acs'])
        # no_gradの間は計算グラフは構築しない
        with torch.no_grad():
            # 次状態のQ値を出力 & maxとなる行動のみ抜き出す
            next_q_value = target_qf(batch['next_obs']).max(dim=1, keepdim=True)[0]
            target_q_value = batch['rews'] + gamma * next_q_value * (1 - batch['terms'])
        
        # 誤差を獲得
        loss = criterion(q_value, target_q_value)

        # 勾配を0に初期化
        optimizer.zero_grad()
        # 逆伝播
        loss.backward()
        # 更新
        optimizer.step()
        
        # 10ステップごとにtarget_qfを更新
        if total_step % 10 == 0:
            target_qf.load_state_dict(qf.state_dict())
    
    if episode % 10 == 0:
        print('episode:', episode, 'return:', step, 'epsilon:', epsilon)