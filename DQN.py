import random
import gym
import numpy as np
import collections
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 构建经验池
class ReplayBuffer:
    def __init__(self,capacity):
        # 定义一个先入先出的队列
        self.buffer = collections.deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        # 将经验添加到队列中
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        # 采样，数量是batch_size，保证iid需要随机
        transitions = random.sample(self.buffer, batch_size)
        # zip用于解包
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)

    def size(self):
        # 返回经验池的大小
        return len(self.buffer)

# 构建DQN，继承nn.Module才能调用自动求导、参数管理等功能。
class DQN_net(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(DQN_net, self).__init__() # 
        # 只有一层隐藏
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))  # 激活函数
        return self.fc2(x)  # 输出层


class DQN:
    def __init__(self, state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device):
        self.action_dim = action_dim
        self.q_net = DQN_net(state_dim, hidden_dim,
                          self.action_dim).to(device)  # Q网络
        # 目标网络，先复制一个一模一样的
        self.target_q_net = DQN_net(state_dim, hidden_dim,
                                 self.action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(),
                                          lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0  # 计数器,记录更新次数
        self.device = device
    
    def take_action(self, state): # 依照episilon-greedy
        if random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action
    
    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.int64).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        
        q_values = self.q_net(states).gather(1, actions)  # Q值，相当于获取batch个action下的Q值（gather函数的用法）
        # import pdb; pdb.set_trace()
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1).to(self.device)  # 下一时刻状态下的最大Q值，batch*action_dim，相当于直接找max_Q了，不弄episilon-greedy
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)  # 目标Q值，防止episode结束后继续累加未来奖励
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())  # 更新目标网络
        self.count += 1


lr = 2e-3
num_episodes = 500
hidden_dim = 128
gamma = 0.98
epsilon = 0.01
target_update = 10
buffer_size = 10000
minimal_size = 500
batch_size = 64
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")

env_name = 'CartPole-v0'
env = gym.make(env_name)
random.seed(0)
np.random.seed(0)
# env.seed(0)
torch.manual_seed(0)
replay_buffer = ReplayBuffer(buffer_size)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
            target_update, device)
return_list = []

for i in range(10):
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):
            episode_return = 0
            state, info = env.reset()
            done = False
            termination = False
            while not done:
                action = agent.take_action(state)
                # import pdb; pdb.set_trace()
                next_state, reward, done, termination, info = env.step(action)
                replay_buffer.add(state, action, reward, next_state, done)
                state = next_state
                episode_return += reward
                # 当buffer数据的数量超过一定值后,才进行Q网络训练
                if replay_buffer.size() > minimal_size:
                    b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                    transition_dict = {
                        'states': b_s,
                        'actions': b_a,
                        'next_states': b_ns,
                        'rewards': b_r,
                        'dones': b_d
                    }
                    agent.update(transition_dict)
            return_list.append(episode_return)
            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({
                    'episode':
                    '%d' % (num_episodes / 10 * i + i_episode + 1),
                    'return':
                    '%.3f' % np.mean(return_list[-10:])
                })
            pbar.update(1)


