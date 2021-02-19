import numpy as np
from policy import util
from policy.tree_datastruct import TreeNode
import random
import matplotlib.pyplot as plt
import matplotlib as mlp

class StatsPolicy(object):
    def __init__(self, creative, tree_model):
        self.root = TreeNode(None, "t", 1, ['0'])
        self.node_list = [self.root]
        self.size = 1
        self.step = []
        self.record = {}
        self.features = []
        self.total_pv = 0
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

    def set_constraint(self):
        with open("data/tree_constraint.txt", "r") as f:
            lines = f.readlines()
            i = 0
            for _ in range(self.size - 1):
                elems = lines[i].replace(' ', '').replace('\n', '').split(",")
                node = util.find(self.root, elems[0])
                print(elems)
                node.constraint = np.zeros((int(elems[1]), int(elems[2])))
                for j in range(int(elems[1])):
                    datas = lines[i + j + 1].replace('\n', '').split()
                    for t in range(int(elems[2])):
                        node.constraint[j][t] = int(datas[t])
                i += int(elems[1]) + 1

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

    def create_random_ctr(self):
        # util.random_weight(self.root)
        util.random_vertex_weight(self.root)
        value_list = [[str(y) for y in range(x.dimension)] for x in self.node_list]
        creatives = util.cartesian_product(value_list)
        f = open("data/ctr_new2.txt", "w")
        id = 10000
        for i in creatives:
            t = self.sum_vertex_weight(i)
            if t < -50000:
                continue
            # t = float(t)/(60.*float(self.size))*6.
            # ctr = sigmoid(t)/40. + 0.02
            ctr = 0.02 + float(t) / 450 / float(self.size -1)
            f.write(str(id)+" ")
            param = [int(x) for x in i.split(" ")]
            for j in range(len(self.node_list)):
                f.write(self.node_list[j].value[param[j]]+" ")
            f.write(str(round(ctr, 6))+"\n")
            id += 1

    def sum_vertex_weight(self,parameters):
        res = 0
        # print(parameters)
        param = [int(x) for x in parameters.split(" ")]
        record = {self.node_list[0].name: param[0]}
        for j in range(1, len(param)):
            record[self.node_list[j].name] = param[j]
            res += 3.0 * self.node_list[j].propose_w[record[self.node_list[j].parent.name]][param[j]]
            res += self.node_list[j].propose_v[param[j]]
            # print(param[j],self.node_list[j].propose_v[param[j]])
        return res


    def save_tree(self, filepath):
        f = open(filepath, "w")
        self.root.fprint(f)

if __name__ == "__main__":
    path = "data/tree_struct2.txt"
    mc = StatsPolicy("", path)
    mc.create_random_ctr()
    # mc.save_tree("data/tree_vertex_weight.txt")

    f = open("data/ctr_new2.txt", "r")
    plt.figure(figsize=(6, 3.5))
    plt.rcParams['font.family'] = ['Times New Roman']
    data = []
    for line in f.readlines():
        elems = line.split()
        data.append(float(elems[-1]))
    plt.hist(data, bins=21, facecolor="blue", edgecolor="black", alpha=0.75)
    plt.xlabel("CTR")
    plt.ylabel("#Banner creatives")
    # plt.show()
   
    plt.savefig('gen_data/pic_ctr2.png',bbox_inches='tight')
