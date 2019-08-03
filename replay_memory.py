import torch
import numpy as np

class ReplayMemory:
    def __init__(self, memory_size=10000, batch_size=32, obs_size=4):
        self.index = 0
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.obs = np.zeros((self.memory_size, obs_size), dtype=np.float)
        self.acs = np.zeros((self.memory_size, 1), dtype=np.int)
        self.rews = np.zeros((self.memory_size, 1), dtype=np.float)
        self.next_obs = np.zeros((self.memory_size, obs_size), dtype=np.float)
        self.terms = np.zeros((self.memory_size, 1), dtype=np.int)

    def add(self, ob, ac, rew, next_ob, term):
        self.obs[self.index % self.memory_size] = ob
        self.acs[self.index % self.memory_size] = ac
        self.rews[self.index % self.memory_size][0] = rew
        self.next_obs[self.index % self.memory_size] = next_ob
        self.terms[self.index % self.memory_size][0] = term
        self.index += 1
    
    def sample(self):
        indices = np.random.randint(0, min(self.memory_size, self.index), self.batch_size)
        batch = dict()
        batch['obs'] = torch.Tensor(self.obs[indices])
        batch['acs'] = torch.LongTensor(self.acs[indices])
        batch['rews'] = torch.Tensor(self.rews[indices])
        batch['next_obs'] = torch.Tensor(self.next_obs[indices])
        batch['terms'] = torch.Tensor(self.terms[indices])
        return batch