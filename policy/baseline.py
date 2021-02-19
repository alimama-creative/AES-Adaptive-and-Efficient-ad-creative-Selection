import numpy as np
import random

class Baseline(object):
    def __init__(self, creative, p_dict):
        self.epsilon = p_dict["epsilon"]
        self.index_all = {}
        creative_len = len(creative)
        self.curr = 0
        self.features = []
        self.weight = np.zeros(creative_len)
        self.pv = np.zeros(creative_len)
        self.clk = np.zeros(creative_len)
        self.ctr = np.zeros((creative_len, 1))
        self.b = np.zeros((creative_len, 1))  # save the res of ctr+upper bound
        self.set_creative(creative)
        self.total_pv = 0
        self.ini_pv = 0

    def set_creative(self, creative):
        i = 0
        for key in creative:
            str1 = ""
            for k in creative[key][0:-2]:
                str1 = str1 + str(k) + " "
            str1 += str(creative[key][-2])
            self.index_all[str1] = i
            self.features.append(str1)
            self.ctr[i] = creative[key][-1]
            i += 1

    def get_creative(self):
        return self.index_all.keys()


class Random(Baseline):
    def update(self, reward):
        return

    def recommend(self, choices, t=1):
        index = random.randint(0, len(choices)-1)
        return self.features[index], self.ctr[index][0]


class thompson(Baseline):
    def update(self, reward_list):
        for reward in reward_list:
            idx = self.index_all[reward[0]]
            self.pv[idx] += 1
            self.clk[idx] += reward[1]

    def recommend(self, choices, t=1):
        if t == 0:
            r = random.randint(0, len(self.features) - 1)
            return self.features[r], self.ctr[r]
        thompson_list = list(map(self.thompson, self.pv, self.clk))
        self.index = np.argmax(thompson_list)
        return self.features[int(self.index)], self.ctr[self.index][0]

    def thompson(self, pv_num, click_num):
        return np.random.beta(1 + click_num, 50 + pv_num - click_num)



class thompson2(Baseline):
    '''following a gaussian distribution'''
    def __init__(self, creative, p_dict):
        super(thompson2,self).__init__(creative, p_dict)
        self.mean = np.zeros((len(creative)))
        self.sum_var = np.zeros((len(creative)))
        self.std2 = np.ones((len(creative)))
        self.alpha = p_dict["alpha"]

    def update(self, reward_list):
        for reward in reward_list:
            idx = self.index_all[reward[0]]
            mean_old = self.clk[idx]/self.pv[idx] if self.pv[idx] > 0 else 0
            self.pv[idx] += 1
            self.clk[idx] += reward[1]
            mean_new = self.clk[idx]/self.pv[idx]
            self.mean[idx] = mean_new
            self.sum_var[idx] += (reward[1]-mean_old) * (reward[1]-mean_new)
            self.std2[idx] = np.sqrt(self.sum_var[idx] / self.pv[idx])

    def recommend(self, choices, t=1):
        if t == 0:
            r = random.randint(0, len(self.features) - 1)
            return self.features[r], self.ctr[r]
        thompson_list = list(map(self.sample_norm, self.mean, self.std2))
        self.index = np.argmax(thompson_list)
        # print(self.index)
        return self.features[int(self.index)], self.ctr[self.index][0]

    def sample_norm(self, u, o):
        return np.random.normal(u,self.alpha * o)


class EpsilonGreedy(Baseline):
    def update(self, reward_list):
        for reward in reward_list:
            self.curr = self.index_all[reward[0]]
            self.clk[self.curr] += reward[1]
            self.pv[self.curr] += 1
            self.weight[self.curr] = self.clk[self.curr]/self.pv[self.curr]
        return

    def recommend(self, choices, t=1):
        if t == 0:
            r = random.randint(0, len(self.features) - 1)
            return self.features[r], self.ctr[r]
        self.curr = self.weight.argmax()
        if random.random() > 1-self.epsilon:
            self.curr = random.randint(0, len(self.features)-1)
        return self.features[self.curr], self.ctr[self.curr][0]

    def get_best_reward(self):
        pos = np.argmax(self.ctr)
        self.weight[pos] = 100000
        sum = 0
        for _ in range(100000):
            item = self.recommend(None, 1)
            sum += item[1]
        sum /= 100000.0
        return sum

    def curr_best_creative(self):
        return self.weight.argmax()

    def print(self):
        print(self.pv, self.clk)


class UCB(Baseline):
    def update(self, reward_list):
        for reward in reward_list:
            self.total_pv += 1
            idx = self.index_all[reward[0]]
            self.pv[idx] += 1
            self.clk[idx] += reward[1]

        for reward in reward_list:
            idx = self.index_all[reward[0]]
            pvpv = self.pv[idx]
            clkclk = self.clk[idx]
            self.b[idx] = clkclk / pvpv + 0.03 * np.sqrt(2 * np.log(self.total_pv) / pvpv)

    def recommend(self, choices, t=1):
        if self.ini_pv < len(self.pv):
            self.index = self.ini_pv
            self.ini_pv += 1
            return self.features[int(self.index)], self.ctr[self.index][0]

        if t == 0:
            r = random.randint(0, len(self.features) - 1)
            return self.features[r], self.ctr[r][0]
        self.index = np.argmax(self.b)
        return self.features[int(self.index)], self.ctr[self.index][0]

    def curr_best_creative(self):
        return int(np.argmax(self.b))

    def print(self):
        print(self.pv)
        print(self.clk)
        print(self.b)
