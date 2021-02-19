import numpy as np
import random

# class linucb(object):
#     def __init__(self, creative, p_dict):
#         self.epsilon = p_dict["epsilon"]
#         self.alpha = 0.02 # the coefficient of the upper bound
#         self.index_all = {}
#         creative_len = len(creative)
#         self.features = []
#         self.ctr = np.zeros((creative_len, 1))
#         self.set_creative(creative)
#         self.total_pv = 0
#         self.ini_pv = 0
        

#     def set_creative(self, creative):
#         i = 0
#         creative_len = len(creative)
#         self.full_A = {}
#         self.theta_a = {}
#         # self.feature_arms = {}
#         feature_arms = np.zeros((creative_len,4))
#         self.p_arm = {}
#         self.b = {}  # save the res of ctr+upper bound
#         for key in creative:
#             str1 = ""
#             vec_feature = []
#             for k in creative[key][0:-2]:
#                 str1 = str1 + str(k) + " "
#             str1 += str(creative[key][-2])
#             for s in creative[key][1:-1]:
#                 vec_feature.append(s)
#             self.index_all[str1] = i
#             self.features.append(str1)
#             self.ctr[i] = creative[key][-1]
            
#             dim = len(creative[key][1:-1])

#             feature_arms[i] = np.array(vec_feature)
#             self.full_A[i] = np.eye(dim)
#             self.b[i] = np.zeros((dim))
#             self.theta_a[i] = np.zeros((dim))
#             i += 1

#         self.feature_arms = feature_arms / feature_arms.max(0)
#         for i in range(creative_len):
#             self.p_arm[i] = self.alpha * np.sqrt(np.matmul(self.feature_arms[i],np.transpose(self.feature_arms[i])))
        

#     def get_creative(self):
#         return self.index_all.keys()
    
#     def update(self, reward_list):
#         for reward in reward_list:
#             idx = self.index_all[reward[0]]
            
#             features = self.feature_arms[idx] # shape (dim,)
#             self.full_A[idx] += np.matmul(features,np.transpose(features))
#             self.b[idx] += reward[1] * features
#             # print(np.shape(features),np.shape(self.full_A[idx]),np.shape(self.b[idx]))

#             A_inv = np.linalg.inv(self.full_A[idx])
#             self.theta_a[idx] = np.matmul(A_inv,self.b[idx])
#             eva_ctr = np.matmul(np.transpose(self.theta_a[idx]),features)
#             upper = np.matmul(np.matmul(features,A_inv),np.transpose(features))
#             self.p_arm[idx] = eva_ctr + self.alpha * np.sqrt(upper)
#             # print(eva_ctr,upper)
            
#             # self.pv[idx] += 1
#             # self.clk[idx] += reward[1]


#     def recommend(self, choices, t=1):
#         if t == 0:
#             r = random.randint(0, len(self.features) - 1)
#             return self.features[r], self.ctr[r]
        
#         self.index = np.argmax(self.p_arm)
#         # index = random.randint(0, len(choices)-1)
#         return self.features[int(self.index)], self.ctr[self.index][0]
    
#     def print(self):
#         print(self.p_arm)

class LinUCB(object):
    def __init__(self, creative, tree_model, p_dict):
        self.epsilon = p_dict["epsilon"]
        self.alpha = p_dict["alpha"] # the coefficient of the upper bound
        self.index_all = {}
        creative_len = len(creative)
        self.features = []
        self.ctr = np.zeros((creative_len, 1))

        self.pv = np.zeros(creative_len)
        self.clk = np.zeros(creative_len)
        self.get_one_hot(tree_model)
        self.set_creative(creative)
        
        self.total_pv = 0
        self.ini_pv = 0
    
    def set_creative(self, creative):
        i = 0
        creative_len = len(creative)
        self.full_A = {}
        self.theta_a = {}
        dim = sum(self.num_list)
        self.feature_arms = np.zeros((creative_len,dim))
        self.p_arm = np.zeros((creative_len))
        self.b = {}  # save the res of ctr+upper bound
        for key in creative:
            str1 = ""
            vec_feature = []
            for k in creative[key][0:-2]:
                str1 = str1 + str(k) + " "
            str1 += str(creative[key][-2])
            ii = 0
            for s in creative[key][1:-1]:
                # print(self.num_list[ii],s)
                sub_one_hot = [0]* self.num_list[ii]
                iidx = self.ele[ii].index(str(int(s)))
                sub_one_hot[iidx]=1
                # print(sub_one_hot)
                vec_feature += sub_one_hot
                ii += 1
            self.index_all[str1] = i
            self.features.append(str1)
            self.ctr[i] = creative[key][-1]

            self.feature_arms[i] = np.array(vec_feature)
            i += 1
        self.full_A = 0.01 * np.eye(dim)
        self.b = np.zeros((dim))
        
    def get_creative(self):
        return self.index_all.keys()
    
    def get_one_hot(self,tree_model):
        self.num_list = []
        self.ele = []
        with open(tree_model,'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.split("\n")[0]
                ss = line.split(", ")
                num = int(ss[2])
                self.ele.append(ss[3:])
                self.num_list.append(num)

        print(self.num_list)
        print(self.ele)

    def update(self, reward_list):
        for reward in reward_list:
            idx = self.index_all[reward[0]]
            self.pv[idx] += 1
            self.clk[idx] += reward[1]
            
            features = self.feature_arms[idx] # shape (dim,)
            features_a = features[np.newaxis,:]
            self.full_A += np.matmul(np.transpose(features_a),features_a)
            self.b += reward[1] * np.transpose(features)
            
        
        self.A_inv = np.linalg.inv(self.full_A)
        self.theta_a = np.matmul(self.A_inv,self.b)
        for i in range(len(self.feature_arms)):
            features = self.feature_arms[i]
            eva_ctr = np.matmul(np.transpose(self.theta_a),features)
            upper = np.matmul(np.matmul(features,self.A_inv),np.transpose(features))
            self.p_arm[i] = eva_ctr + self.alpha * np.sqrt(upper)
            # print(eva_ctr,upper)
            
            # self.pv[idx] += 1
            # self.clk[idx] += reward[1]


    def recommend(self, choices, t=1):
        if t == 0:
            r = random.randint(0, len(self.features) - 1)
            return self.features[r], self.ctr[r]
        
        self.index = np.argmax(self.p_arm)
        # index = random.randint(0, len(choices)-1)
        return self.features[int(self.index)], self.ctr[self.index][0]
    
    def print(self):
        # print(self.p_arm)
        print(self.pv,self.clk)

class lints(LinUCB):
    def __init__(self, creative, tree_model, p_dict):
        super().__init__(creative, tree_model, p_dict)

    def update(self, reward_list):
        for reward in reward_list:
            idx = self.index_all[reward[0]]
            self.pv[idx] += 1
            self.clk[idx] += reward[1]
            
            features = self.feature_arms[idx] # shape (dim,)
            features_a = features[np.newaxis,:]
            self.full_A += np.matmul(np.transpose(features_a),features_a)
            self.b += reward[1] * np.transpose(features)
            
        
        self.A_inv = np.linalg.inv(self.full_A)
        self.theta_a = np.matmul(self.A_inv,self.b)
        self.diag_A_inv = np.sqrt( np.diagonal(self.A_inv))

    
    def recommend(self, choices, t=1):
        if t == 0:
            r = random.randint(0, len(self.features) - 1)
            return self.features[r], self.ctr[r]
        weight = np.random.multivariate_normal(mean=self.theta_a,cov = self.alpha * self.A_inv,size=1)[0]
        # weight = np.random.normal(loc=self.theta_a,scale=self.alpha * self.diag_A_inv)
        self.score = np.matmul(self.feature_arms,weight)
        self.index = np.argmax(self.score)
        # index = random.randint(0, len(choices)-1)
        return self.features[int(self.index)], self.ctr[self.index][0]