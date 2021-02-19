import numpy as np
import random

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


def find(node, name):
    if name == node.name:
        return node
    for nex in node.children:
        t = find(nex, name)
        if t is not None:
            return t
    return None


def dynamic_search(node, epsilon):
    for nex in node.children:
        dynamic_search(nex, epsilon)
    if node.is_root():
        return
    w = node.weight.copy()
    for nex in node.children:
        w += nex.added_weight
    node.dynamic_arr = w.argmax(axis=1)
    for i in range(len(node.dynamic_arr)):
        if random.random() > 1 - epsilon:
            node.dynamic_arr[i] = random.randint(0, len(w[i]) - 1)
            while w[i][node.dynamic_arr[i]] < -1000:
                node.dynamic_arr[i] = random.randint(0, len(w[i]) - 1)
    node.added_weight = [w[i][node.dynamic_arr[i]] for i in range(len(w))]
    return

def dynamic2search(node, epsilon):
    for nex in node.children:
        dynamic2search(nex, epsilon)
    if node.is_root():
        # node.print()
        # print(node.dynamic_arr)
        return
    w = node.weight.copy()
    for nex in node.children:
        w += nex.added_weight
    # print(node.name,np.shape(node.self_weight),np.shape(w),np.shape(node.added_weight))
    w += node.self_weight
    node.dynamic_arr = w.argmax(axis=1)
    # print(node.name,np.shape(node.self_weight),np.shape(w),np.shape(node.dynamic_arr),np.shape(node.added_weight))
    for i in range(len(node.dynamic_arr)):
        if random.random() > 1 - epsilon:
            node.dynamic_arr[i] = random.randint(0, len(w[i]) - 1)
            while w[i][node.dynamic_arr[i]] < -1000:
                node.dynamic_arr[i] = random.randint(0, len(w[i]) - 1)
    node.added_weight = [w[i][node.dynamic_arr[i]] for i in range(len(w))]
    return


def two_rl(list1, list2):
    res_list = []
    for str1 in list1:
        for str2 in list2:
            if str2 == "/":
                res_list.append(str1 + ' ' + str2)
            else:
                res_list.append(str1 + ' ' + str(float(str2)))
    return res_list


def cartesian_product_rl(list_of_list):
    list1 = []
    for i in list_of_list[0]:
        if i == "/":
            list1.append(i)
        else:
            list1.append(str(float(i)))
    for list2 in list_of_list[1:]:
        list1 = two_rl(list1, list2)
    return list1


def two(list1, list2):
    res_list = []
    for str1 in list1:
        for str2 in list2:
            res_list.append(str1+' '+str2)
    return res_list


def cartesian_product(list_of_list):
    list1 = list_of_list[0]
    for list2 in list_of_list[1:]:
        list1 = two(list1, list2)
    return list1


def random_weight(node):
    weight_range = [0, 500, 1500, 3000, 5000]
    if not node.is_root():
        # node.propose_w = np.random.randint(-1500, 1500, (node.weight.shape[0], node.weight.shape[1]))
        node.propose_w = np.random.normal(0, 1, (node.weight.shape[0], node.weight.shape[1]))
        for i in range(node.weight.shape[0]):
            for j in range(node.weight.shape[1]):
                if node.constraint[i][j] == 0:
                    node.propose_w[i][j] = -1000000
        # node.propose_w = np.zeros((node.weight.shape[0], node.weight.shape[1]))
        # for i in node.propose_w:
        #     for j in range(len(i)):
        #         i[j] = np.random.randint(weight_range[j], weight_range[j + 1])
    for nex in node.children:
        random_weight(nex)
    return

def random_vertex_weight(node):
    weight_range = [0, 500, 1500, 3000, 5000]
    if not node.is_root():
        # node.propose_w = np.random.randint(-1500, 1500, (node.weight.shape[0], node.weight.shape[1]))
        node.propose_w = np.random.normal(0, 1, (node.weight.shape[0], node.weight.shape[1]))
        node.propose_v = 0.3 * np.random.randn(node.weight.shape[1])
        for i in range(node.weight.shape[0]):
            for j in range(node.weight.shape[1]):
                if node.constraint[i][j] == 0:
                    node.propose_w[i][j] = -1000000
        # node.propose_w = np.zeros((node.weight.shape[0], node.weight.shape[1]))
        # for i in node.propose_w:
        #     for j in range(len(i)):
        #         i[j] = np.random.randint(weight_range[j], weight_range[j + 1])
    for nex in node.children:
        random_vertex_weight(nex)
    return