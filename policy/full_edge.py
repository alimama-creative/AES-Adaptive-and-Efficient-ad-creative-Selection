import numpy as np 
import random
from policy import util
from policy.lr import LR,CtrDataset,FTRL
import tqdm
import time
import math
import copy
from sklearn import preprocessing


class Full_Reg(object):
    def __init__(self, creative, tree_model, constraints ,p_dict):
        self.epsilon = p_dict["epsilon"]
        self.ee = p_dict["ee"]
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
        self.set_full_feat()
    
    def set_full_feat(self):
        X = []
        # print(self.features)
        for line in self.features:
            line = line.split()
            encode = [int(float(x)) for x in line]
            X.append(encode)
        
        x_array = np.array(X)
        _,fields= np.shape(x_array)
        self.fields = fields
        fields_offset = [0]
        encode_list = []
        for i in range(0,fields):
            t = np.unique(x_array[:,i])
            encode_dict = {}
            for idx,item in enumerate(t):
                # print(idx,item)
                encode_dict[item] = idx
            encode_list.append(encode_dict)
            font_len = len(np.unique(x_array[:,i]))
            fields_offset.append(font_len)

        print(fields_offset)
        self.fields_offset = fields_offset[1:]
        cum_offset = np.cumsum(fields_offset)
        
        new_feature = []
        for x in X:
            # feature of node is determined
            feat = []
            for i in range(len(x)):
                for j in range(i+1,len(x)):
                    # print(i,j,x[i],x[j],fields_offset[i],fields_offset[j],x[i]*fields_offset[j] + x[j],x[i]*fields_offset[i] + x[j])
                    feat.append(x[i]*fields_offset[j+1] + x[j])
            feat = np.array(feat)
            feat = np.concatenate((x,feat))
            new_feature.append(feat)
        
        one_hot = preprocessing.OneHotEncoder(categories='auto')
        self.mtx_A = one_hot.fit_transform(new_feature).toarray()
        print(self.mtx_A.shape)
        
    def get_creative(self):
        return None
    
    def update(self, reward_list, pv_type=True):
        # print(self.weight_vec)
        for reward in reward_list:
            curr = self.index_all[reward[0]]
            self.pv[curr] += 1
            self.clk[curr] += reward[1]
            beta_clk = self.clk[curr]
            beta_pv = self.pv[curr]
            self.b_vec[curr] = beta_clk / beta_pv
        (s, t) = np.shape(self.pv)
        weight_mtx = self.pv * np.eye(s)
        mtx = np.matmul(np.matmul(self.mtx_A.T, weight_mtx), self.mtx_A)
        inv_mtx = np.linalg.pinv(mtx)
        self.mtx_A_inv = np.matmul(np.matmul(inv_mtx, self.mtx_A.T), weight_mtx)
        self.weight_vec = np.matmul(self.mtx_A_inv, self.b_vec)
        self.total_ctr = np.matmul(self.mtx_A,self.weight_vec)

    def recommend(self, a=None, t=1):
        epsilon = self.epsilon
        if t == 0 or random.random() < 1 - epsilon:
            r = random.randint(0, len(self.features)-1)
            return self.features[r], self.ctr[r]
        # total_ctr = np.matmul(self.mtx_A,self.weight_vec)
        idx = np.argmax(self.total_ctr)
        return self.features[idx],self.ctr[idx]

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

class Full_Ts(Full_Reg):
    def __init__(self, creative, tree_model, constraints, p_dict):
        super().__init__(creative, tree_model, constraints, p_dict)

    def update(self, reward_list, pv_type=True):
        # print(self.weight_vec)
        for reward in reward_list:
            curr = self.index_all[reward[0]]
            self.pv[curr] += 1
            self.clk[curr] += reward[1]
            beta_clk = self.clk[curr]
            beta_pv = self.pv[curr]
            self.b_vec[curr] = beta_clk / beta_pv
        (s, t) = np.shape(self.pv)
        weight_mtx = self.pv * np.eye(s)
        mtx = np.matmul(np.matmul(self.mtx_A.T, weight_mtx), self.mtx_A)
        self.inv_mtx = np.linalg.pinv(mtx)
        self.mtx_A_inv = np.matmul(np.matmul(self.inv_mtx, self.mtx_A.T), weight_mtx)
        self.expected_weight = np.matmul(self.mtx_A_inv, self.b_vec)

    def recommend(self, a=None, t=1):
        if t == 0 :
            r = random.randint(0, len(self.features)-1)
            return self.features[r], self.ctr[r]
        mean_vec = self.expected_weight.flatten()
        # print(mean_vec.shape)
        self.weight_vec = np.random.multivariate_normal(mean=mean_vec, cov= self.alpha * self.inv_mtx, size=1)[0]
        total_ctr = np.matmul(self.mtx_A,self.weight_vec)
        idx = np.argmax(total_ctr)
        return self.features[idx],self.ctr[idx]

class Full_probit(Full_Reg):
    def __init__(self, creative, tree_model, constraints, p_dict):
        super().__init__(creative, tree_model, constraints, p_dict)
    
    def update(self, reward_list, pv_type=True):
        for reward in reward_list:
            curr = self.index_all[reward[0]]
            self.pv[curr] += 1
            self.clk[curr] += reward[1]
            beta_clk = self.clk[curr]
            beta_pv = self.pv[curr]
            self.b_vec[curr] = beta_clk / beta_pv
        (s, t) = np.shape(self.pv)
        weight_mtx = self.pv * np.eye(s)
        mtx = np.matmul(np.matmul(self.mtx_A.T, weight_mtx), self.mtx_A)
        self.inv_mtx = np.linalg.pinv(mtx)
        self.mtx_A_inv = np.matmul(np.matmul(self.inv_mtx, self.mtx_A.T), weight_mtx)
        self.expected_weight = np.matmul(self.mtx_A_inv, self.b_vec)
    
    def recommend(self, a=None, t=1):
        if t == 0 :
            r = random.randint(0, len(self.features)-1)
            return self.features[r], self.ctr[r]
        mean_vec = self.expected_weight.flatten()
        # print(mean_vec.shape)
        self.weight_vec = np.random.multivariate_normal(mean=mean_vec, cov= self.alpha * self.inv_mtx, size=1)[0]
        final_idx = self.hill_climb()
        return self.features[final_idx],self.ctr[final_idx]

    def hill_climb(self,s=2,k=3):
        # pick out A_0 randomly
        for ss in range(s):
            final_idx_reward = {}
            rdn_idx = random.randint(0, len(self.features)-1)
            str0 = self.features[rdn_idx]
            all_ele = str0.split()
            # print(all_ele)
            ele_list = copy.deepcopy(all_ele)
            for kk in range(k):
                field_idx = random.randint(0, self.fields-1)
                idx_pool = []
                # feat_pool = []
                for xx in range(self.fields_offset[field_idx]):
                    # print(field_idx,xx,self.fields_offset[field_idx],self.fields_offset)
                    ele_list[field_idx] = str(float(xx))
                    str_list = " ".join(ele_list)
                    if str_list in self.index_all.keys():
                        idx_pool.append(self.index_all[str_list])
                        # feat_pool.append(str_list.split())
                feat = self.mtx_A[idx_pool]
                reward = np.matmul(feat,self.weight_vec)
                idx_this = np.argmax(reward)
                ele_list = self.features[idx_this].split()
            final_idx_reward[idx_this] = reward[idx_this]
        res = max(final_idx_reward, key=lambda x: final_idx_reward[x])
        return res