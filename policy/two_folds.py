import numpy as np 
import random
from policy.tree_datastruct import TreeNode
from policy import util
# from policy.lr import LR,CtrDataset,FTRL
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm
import time
import math


class Stats2(object):
    def __init__(self, creative, tree_model, constraints ,p_dict):
        self.epsilon = p_dict["epsilon"]
        self.ee = p_dict["ee"]
        self.bias = p_dict["bias"]
        self.root = TreeNode(None, "t", 1, ['0'])
        self.node_list = [self.root]
        self.size = 1
        self.step = []
        self.record = {}
        self.features = []
        self.total_pv = 0
        self.constraint_path =  constraints
        if creative != "":
            self.index_all = {}
            creative_len = len(creative)
            self.b_vec = np.zeros((creative_len, 1))
            self.pv = np.zeros((creative_len, 1))
            self.clk = np.zeros((creative_len, 1))
            self.ctr = np.zeros((creative_len, 1))
            self.set_creative(creative)
        self.init_tree(tree_model)
        self.set_constraint()
        self.root.print()
        self.alpha = p_dict["alpha"]
        

    def set_creative(self, creative):
        i = 0
        for key in creative:
            str1 = ""
            for k in creative[key][0:-2]:
                str1 = str1+str(k)+" "
            str1 += str(creative[key][-2])
            self.index_all[str1] = i
            self.ctr[i] = creative[key][-1]
            self.features.append(str1)
            i += 1

    def init_tree(self, tree_model):
        with open(tree_model, "r") as f:
            for line in f.readlines():
                self.size += 1
                elems = line.replace(' ', '').replace('\n', '').split(",")
                node = util.find(self.root, elems[0])
                new_node = TreeNode(node, elems[1], int(elems[2]), elems[3:])
                self.node_list.append(new_node)
                node.children.append(new_node)
            self.step = np.zeros(self.size)

    def get_creative(self):
        return None

    def set_constraint(self):
        file_path = self.constraint_path
        with open(file_path, "r") as f:
            lines = f.readlines()
            i = 0
            for _ in range(self.size-1):
                elems = lines[i].replace(' ', '').replace('\n', '').split(",")
                node = util.find(self.root, elems[0])
                node.constraint = np.zeros((int(elems[1]), int(elems[2])))
                for j in range(int(elems[1])):
                    datas = lines[i+j+1].replace('\n', '').split()
                    for t in range(int(elems[2])):
                        node.constraint[j][t] = int(datas[t])
                        if int(datas[t]) == 0:
                            node.weight[j][t] = -1000000
                i += int(elems[1]) + 1

    def recommend(self, a=None, t=1):
        epsilon = self.epsilon
        if t == 0:
            r = random.randint(0, len(self.features)-1)
            return self.features[r], self.ctr[r]
        if self.ee == 2:
            util.dynamic2search(self.root, self.epsilon)
            epsilon = 0
        else:
            util.dynamic2search(self.root, 0)
        i = 1
        self.record = {self.node_list[0].name: 0}
        if self.ee != 1:
            if random.random() < 1-epsilon:
                for node in self.node_list[1:]:
                    self.record[node.name] = node.dynamic_arr[self.record[node.parent.name]]
                    self.step[i] = self.record[node.name]
                    i += 1
            else:
                for node in self.node_list[1:]:
                    self.record[node.name] = random.randint(0, node.dimension-1)
                    while node.weight[self.record[node.parent.name]][self.record[node.name]] < -10000:
                        self.record[node.name] = random.randint(0, node.dimension-1)
                    self.step[i] = self.record[node.name]
                    i += 1
        else:
            for node in self.node_list[1:]:
                if random.random() > 1-epsilon:
                    self.record[node.name] = random.randint(0, node.dimension - 1)
                    while node.weight[self.record[node.parent.name]][self.record[node.name]] < -10000:
                        self.record[node.name] = random.randint(0, node.dimension - 1)
                else:
                    self.record[node.name] = node.dynamic_arr[self.record[node.parent.name]]
                    while node.weight[self.record[node.parent.name]][self.record[node.name]] < -10000:
                        self.record[node.name] = random.randint(0, node.dimension - 1)
                self.step[i] = self.record[node.name]
                i += 1
        # get defined ctr
        str1 = ""
        i = 0
        for node in self.node_list[:-1]:
            str1 = str1+str(float(node.value[int(self.step[i])]))+" "
            i += 1
        str1 = str1 + str(float(self.node_list[-1].value[int(self.step[i])]))
        return str1, self.ctr[self.index_all[str1]][0]

    def curr_best_creative(self):
        util.dynamic_search(self.root, 0)
        self.record = {self.node_list[0].name: 0}
        i = 1
        for node in self.node_list[1:]:
            self.record[node.name] = node.dynamic_arr[self.record[node.parent.name]]
            self.step[i] = self.record[node.name]
            i += 1
        str1 = ""
        i = 0
        for node in self.node_list[:-1]:
            str1 = str1 + str(float(node.value[int(self.step[i])])) + " "
            i += 1
        str1 = str1 + str(float(self.node_list[-1].value[int(self.step[i])]))
        return self.index_all[str1]

    def set_pv_info(self, idx):
        self.record = {self.node_list[0].name: 0}
        params = idx.split()
        for i in range(len(params)):
            for j in range(len(self.node_list[i].value)):
                if float(self.node_list[i].value[j]) == float(params[i]):
                    self.record[self.node_list[i].name] = j
                    self.step[i] = j

    def get_A_matrix(self, tree_model):
        pos = []
        leng = 0
        parent = {"t": None}
        record = {"t": 1}
        node_num = [1]
        node_features = [["0"]]
        node_list = ["t"]
        index = {}
        with open(tree_model, "r") as f:
            last_len = 0
            for line in f.readlines():
                elems = line.replace(' ', '').replace('\n', '').split(",")
                record[elems[1]] = int(elems[2])
                parent[elems[1]] = elems[0]
                feature = [x for x in elems[3:]]
                node_features.append(feature)
                node_num.append(int(elems[2]))
                node_list.append(elems[1])
                if len(pos) == 0:
                    pos.append(0)
                else:
                    last_len += pos[len(pos) - 1]
                    pos.append(last_len)
                last_len = record[elems[0]] * record[elems[1]]
                leng += last_len

        res = []
        value_list = [[str(y) for y in range(x)] for x in node_num]
        creatives = util.cartesian_product(value_list)
        row_cnt = 0
        node_cum = np.cumsum(node_num)
        node_bias = sum(node_num) - 1
        leng += node_bias
        self.node_bias = node_bias
        # print(node_cum)
        for i in range(len(self.node_list[1:])):
            self.node_list[i+1].pos = pos[i]
            self.node_list[i+1].self_pos = node_cum[i] - 1
            # print(self.node_list[i+1].name,self.node_list[i+1].value,self.node_list[i+1].pos,self.node_list[i+1].self_pos)
        for i in creatives:
            t = self.sum_weight(i)
            if t < -50000:
                continue
            param = [int(x) for x in i.split(" ")]
            temp = np.zeros(leng)
            record2 = {"t": 0}
            for j in range(1, len(param)):
                record2[node_list[j]] = param[j]
                temp[node_cum[j-1] - 1 + param[j]] = 1
                temp[node_bias + pos[j-1]+param[j] + record2[parent[node_list[j]]]*node_num[j]] = 1
            str1 = ""
            for j in range(len(node_list) - 1):
                str1 = str1 + node_features[j][param[j]] + " "
            str1 = str1 + node_features[len(node_list) - 1][param[len(node_list) - 1]]
            index[str1] = row_cnt
            row_cnt += 1
            res.append(temp.tolist())
        res = np.mat(res)
        return res, index, leng 

    def sum_weight(self, parameters):
        res = 0
        param = [int(x) for x in parameters.split(" ")]
        record = {self.node_list[0].name: param[0]}
        for j in range(1, len(param)):
            record[self.node_list[j].name] = param[j]
            res += self.node_list[j].weight[record[self.node_list[j].parent.name]][param[j]]
        return res

    def panduan(self):
        for i in range(1, self.size):
            pos = int(self.step[i])
            if self.node_list[i].weight[self.record[self.node_list[i].parent.name]][pos] < -50000:
                return 0

class Reg2pv(Stats2):
    def __init__(self, creative, tree_model, constraints, p_dict):
        super(Reg2pv, self).__init__(creative, tree_model, constraints, p_dict)
        self.mtx_A, self.idx_for_ctr, self.leng = self.get_A_matrix(tree_model)
        self.mtx_A_inv = np.linalg.pinv(self.mtx_A)
        self.weight_vec = np.zeros((self.leng, 1))

    def update(self, reward_list, pv_type=True):
        # print(self.weight_vec)
        for reward in reward_list:
            curr = self.index_all[reward[0]]
            self.pv[curr] += 1
            self.clk[curr] += reward[1]
            beta_clk = self.clk[curr]
            beta_pv = self.pv[curr]
            self.b_vec[curr] = beta_clk / beta_pv - self.bias
        (s, t) = np.shape(self.pv)
        weight_mtx = self.pv * np.eye(s)
        mtx = np.matmul(np.matmul(self.mtx_A.T, weight_mtx), self.mtx_A)
        inv_mtx = np.linalg.pinv(mtx)
        self.mtx_A_inv = np.matmul(np.matmul(inv_mtx, self.mtx_A.T), weight_mtx)
        self.weight_vec = np.matmul(self.mtx_A_inv, self.b_vec)
        self.set_weight(self.root)

    def set_weight(self, node):
        if not node.is_root():
            (w, h) = node.weight.shape
            for j in range(h):
                node.self_weight[j] = self.weight_vec[node.self_pos + j]
                # print(node.name,node.self_weight,w,h)
                for i in range(w):
                    if node.weight[i][j] < -10000:
                        continue
                    node.weight[i][j] = self.weight_vec[self.node_bias + node.pos + i * h + j]
        for nex in node.children:
            self.set_weight(nex)
        return

    # def recommend(self, a=None, t=1):
    #     # epsilon = self.epsilon
    #     epsilon = self.epsilon
    #     util.dynamic2search(self.root,1.0)
    #     # self.root.print()
    #     if t == 0 or random.random() < 1 - epsilon:
    #         r = random.randint(0, len(self.features)-1)
    #         return self.features[r], self.ctr[r]
        

    #     total_ctr = np.matmul(self.mtx_A,self.weight_vec)
    #     idx = np.argmax(total_ctr)
    #     return self.features[idx],self.ctr[idx]



class Reg2Ts(Reg2pv):
    def __init__(self, creative, tree_model, constraints, p_dict):
        super(Reg2Ts, self).__init__(creative, tree_model, constraints, p_dict)
        self.inv_mtx = np.ones((self.leng,self.leng))
        self.expected_weight = np.zeros((self.leng))
        self.search_time = 0.0
    
    def update(self, reward_list, pv_type=True):
        # print(np.linalg.norm(self.inv_mtx))
        # print(self.expected_weight)
        for reward in reward_list:
            curr = self.index_all[reward[0]]
            self.pv[curr] += 1
            self.clk[curr] += reward[1]
            beta_clk = self.clk[curr]
            beta_pv = self.pv[curr]
            self.b_vec[curr] = beta_clk / beta_pv - self.bias
        
        bias_n = 1

        (s, t) = np.shape(self.pv)
        weight_mtx = self.pv * np.eye(s)
        mtx = np.matmul(np.matmul(self.mtx_A.T, weight_mtx), self.mtx_A) + 0.1 * np.eye(self.leng)
        
        # mtx = np.matmul(self.mtx_A.T, self.mtx_A)/bias_n/bias_n + 1/bias_w/bias_w        
        self.inv_mtx = np.linalg.pinv(mtx)

        sss = np.matmul(np.matmul(self.inv_mtx,self.mtx_A.T),weight_mtx)
        www = np.matmul(sss, self.b_vec)
        self.expected_weight = www.T.A[0]

    def recommend(self, a=None, t=1):
        epsilon = self.epsilon
        if t == 0:
            r = random.randint(0, len(self.features)-1)
            return self.features[r], self.ctr[r]
        self.weight_vec = np.random.multivariate_normal(mean=self.expected_weight, cov= self.alpha * self.inv_mtx, size=1)[0]
        # total_ctr = np.matmul(self.mtx_A,self.weight_vec)
        # idx = np.argmax(total_ctr)
        # return self.features[idx],self.ctr[idx]
        self.set_weight(self.root)
        t1 = time.clock()
        if self.ee == 2:
            util.dynamic2search(self.root, self.epsilon)
            epsilon = 0
        else:
            util.dynamic2search(self.root, 0)
        i = 1
        self.search_time += time.clock() - t1
        self.record = {self.node_list[0].name: 0}
        if self.ee != 1:
            if random.random() < 1-epsilon:
                for node in self.node_list[1:]:
                    self.record[node.name] = node.dynamic_arr[self.record[node.parent.name]]
                    self.step[i] = self.record[node.name]
                    i += 1
            else:
                for node in self.node_list[1:]:
                    self.record[node.name] = random.randint(0, node.dimension-1)
                    while node.weight[self.record[node.parent.name]][self.record[node.name]] < -10000:
                        self.record[node.name] = random.randint(0, node.dimension-1)
                    self.step[i] = self.record[node.name]
                    i += 1
        else:
            for node in self.node_list[1:]:
                if random.random() > 1-epsilon:
                    self.record[node.name] = random.randint(0, node.dimension - 1)
                    while node.weight[self.record[node.parent.name]][self.record[node.name]] < -10000:
                        self.record[node.name] = random.randint(0, node.dimension - 1)
                else:
                    self.record[node.name] = node.dynamic_arr[self.record[node.parent.name]]
                    while node.weight[self.record[node.parent.name]][self.record[node.name]] < -10000:
                        self.record[node.name] = random.randint(0, node.dimension - 1)
                self.step[i] = self.record[node.name]
                i += 1
        # get defined ctr
        str1 = ""
        i = 0
        for node in self.node_list[:-1]:
            str1 = str1+str(float(node.value[int(self.step[i])]))+" "
            i += 1
        str1 = str1 + str(float(self.node_list[-1].value[int(self.step[i])]))
        return str1, self.ctr[self.index_all[str1]][0]
