import numpy as np

class TreeNode(object):
    def __init__(self, parent, name, num, value):
        self.name = name
        self.parent = parent
        self.children = []
        self.pos = 0
        self.dimension = num
        self.dynamic_arr = []
        self.added_weight = []
        self.constraint = []
        self.pv = np.zeros(num)
        self.clk = np.zeros(num)
        self.self_weight = np.zeros(num)
        self.self_pos = 0
        if parent is None:
            self.weight = []
            self.value = value
            self.pv_num = np.zeros(num)
            self.clk_num = np.zeros(num)
        else:
            self.weight = np.zeros((self.parent.dimension, num))
            self.value = value
            self.pv_num = np.zeros((self.parent.dimension, num))
            self.clk_num = np.zeros((self.parent.dimension, num))
        self.propose_w = []
        self.propose_v = []


    def is_leaf(self):
        return self.children == []

    def is_root(self):
        return self.parent is None

    def print(self):
        print(self.name)
        print(self.weight)
        print(self.dynamic_arr)
        print(self.self_weight)
        for nex in self.children:
            nex.print()
        return

    def fprint(self, f):
        f.write(self.name)
        f.write('\n')
        for row in self.propose_w:
            f.write(' '.join([str(num) for num in row]))
            f.write('\n')
        for nex in self.children:
            nex.fprint(f)
        return
