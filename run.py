import numpy as np
import logging
import coloredlogs
import argparse
from policy.baseline import EpsilonGreedy, UCB, Random, thompson, thompson2
from policy.stats_policy import Reg_Pv, Reg, Epsilon1,Reg_TS
from policy.two_folds import Reg2pv,Reg2Ts
from policy.linucb import LinUCB,lints
from policy.full_edge import Full_Reg,Full_Ts,Full_probit
import datetime


import random

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG')


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


def click(ctr):
    if float(random.randint(1, 10000))/10000. < ctr:
        return 1
    else:
        return 0


ans = []
regret_ans = []
def get_pv(path, num):
    res = []
    t = 0
    with open(path, "r") as f:
        lines = f.readlines()
        for line in lines:
            if t >= num:
                break
            t += 1
            creative = line.split()
            str1 = ""
            for k in creative[0:-2]:
                str1 = str1 + k + " "
            str1 += str(creative[-2])
            res.append([str1, float(creative[-1])])
    return res


def evaluate(p_dict, policy, best_ctr):
    global ans
    score = 0.0
    total_count = 0.0
    pv_count = 0.0
    window_cnt = 0.0
    window_clk = 0.0
    batch_size = p_dict["batch_num"]
    creative_set = policy.get_creative()
    reward_list = []
    r_sum = 0
    regret_list = []
    res = []
    record_time = [0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10,10.5,11,11.5,12,12.5,13,13.5,14,14.5,15,15.5,16,16.5,17,17.5,18,18.5,19,19.5,20,20.5,21,21.5,22,22.5,23,23.5,24,24.5,25,25.5,26,26.5,27,27.5,28,28.5,29,29.5,30,30.5,31,31.5,32,32.5,33,33.5,34,34.5,35,35.5,36,36.5,37,37.5,38,38.5,39,39.5,40,40.5,41,41.5,42,42.5,43,43.5,44,44.5,45,45.5,46,46.5,47,47.5,48,48.5,49,49.5,50,50.5,51,51.5,52,52.5,53,53.5,54,54.5,55,55.5,56,56.5,57,57.5,58,58.5,59,59.5,60,60.5,61,61.5,62,62.5,63,63.5,64,64.5,65,65.5,66,66.5,67,67.5,68,68.5,69,69.5,70,70.5,71,71.5,72,72.5,73,73.5,74,74.5,75,75.5,76,76.5,77,77.5,78,78.5,79,79.5,80,80.5,81,81.5,82,82.5,83,83.5,84,84.5,85,85.5,86,86.5,87,87.5,88,88.5,89,89.5,90,90.5,91,91.5,92,92.5,93,93.5,94,94.5,95,95.5,96,96.5,97,97.5,98,98.5,99,99.5,100,100.5,101,101.5,102,102.5,103,103.5,104,104.5,105,105.5,106,106.5,107,107.5,108,108.5,109,109.5,110,110.5,111,111.5,112,112.5,113,113.5,114,114.5,115,115.5,116,116.5,117,117.5,118,118.5,119,119.5,120,120.5,121,121.5,122,122.5,123,123.5,124,124.5,125,125.5,126,126.5,127,127.5,128,128.5,129,129.5,130,130.5,131,131.5,132,132.5,133,133.5,134,134.5,135,135.5,136,136.5,137,137.5,138,138.5,139,139.5,140,140.5,141,141.5,142,142.5,143,143.5,144,144.5,145,145.5,146,146.5,147,147.5,148,148.5,149,149.5,150,150.5,151,151.5,152,152.5,153,153.5,154,154.5,155,155.5,156,156.5,157,157.5,158,158.5,159,159.5,160,160.5,161,161.5,162,162.5,163,163.5,164,164.5,165,165.5,166,166.5,167,167.5,168,168.5,169,169.5,170,170.5,171,171.5,172,172.5,173,173.5,174,174.5,175,175.5,176,176.5,177,177.5,178,178.5,179,179.5,180,180.5,181,181.5,182,182.5,183,183.5,184,184.5,185,185.5,186,186.5,187,187.5,188,188.5,189,189.5,190,190.5,191,191.5,192,192.5,193,193.5,194,194.5,195,195.5,196,196.5,197,197.5,198,198.5,199,199.5,200,200.5,201,201.5,202,202.5,203,203.5,204,204.5,205,205.5,206,206.5,207,207.5,208,208.5,209,209.5,210,210.5,211,211.5,212,212.5,213,213.5,214,214.5,215,215.5,216,216.5,217,217.5,218,218.5,219,219.5,220,220.5,221,221.5,222,222.5,223,223.5,224,224.5,225,225.5,226,226.5,227,227.5,228,228.5,229,229.5,230,230.5,231,231.5,232,232.5,233,233.5,234,234.5,235,235.5,236,236.5,237,237.5,238,238.5,239,239.5,240]
    t = 0
    

    for _ in range(1):
        for pv in range(p_dict["pv_num"]):
            idx, creative_ctr = policy.recommend(creative_set, t)
            reward = click(creative_ctr)
            if reward == 0:
                r_n = 0
            else:
                r_n = 1
            reward_list.append([idx, r_n])
            pv_count += 1
            total_count += 1
            score += reward
            r_sum += reward
            if pv_count % 100000 == 0:
                regret_list.append(best_ctr-float(r_sum)/100000.0)
                r_sum = 0
                
            if pv_count % batch_size == 0:
                t = 1
                policy.update(reward_list)
                reward_list = []

            if float(pv_count) / 10000.0 in record_time:
                logger.info('{} ---- ctr:{:5f}'.format(pv_count, score / pv_count))
                res.append(score / pv_count)
            

    ans.append(res)
    regret_ans.append(regret_list)


def run(creative_file, policy_type, other,constraints, p_dict):
    # load creative parameters into array
    creative_np = np.loadtxt(creative_file)
    creative = {}
    best_ctr = 0
    for art in creative_np:
        creative[int(art[0])] = [float(x) for x in art[1:]]
        if best_ctr < float(art[-1]):
            best_ctr = float(art[-1])
    if policy_type == "Random":
        policy = Random(creative, p_dict)
    elif policy_type == "EGreedy":
        policy = EpsilonGreedy(creative, p_dict)
    elif policy_type == "thompson":
        policy = thompson(creative, p_dict)
    elif policy_type == "ucb":
        policy = UCB(creative, p_dict)
    elif policy_type == "IndEgreedy":
        policy = Epsilon1(creative, other, constraints, p_dict)
    elif policy_type == "Edge_TS":
        policy = Reg_TS(creative, other, constraints, p_dict)
    elif policy_type == "LinUCB":
        policy = LinUCB(creative, other, p_dict)
    elif policy_type == "TEgreedy":
        policy = Reg2pv(creative,other, constraints, p_dict)
    elif policy_type == "Full_TS":
        policy = Full_Ts(creative, other, constraints, p_dict)
    elif policy_type == 'Vertex_TS':
        policy = lints(creative, other, p_dict)
    elif policy_type == 'MVT':
        policy = Full_probit(creative,other,constraints,p_dict)
    elif policy_type == "AES":
        policy = Reg2Ts(creative,other, constraints, p_dict)
    else:
        policy = Random(creative, p_dict)

    if p_dict["get_best"] == 1:
        s = policy.get_best_reward()
        return s
    else:
        evaluate(p_dict, policy, best_ctr)
        # policy.print()
        return 0


if __name__ == "__main__":
    logger.info("Start To Run Simulator Program")

    tree_filepath = "data/tree_struct2.txt"
    constraint_filepath = "data/tree_constraint.txt"

    parser = argparse.ArgumentParser(description='Run Simulation Project.')
    
    parser.add_argument("-f", "--file_path", type=str,default="data/ctr_new.txt", help="creative_path")
    parser.add_argument("-p", "--pv_num", type=int, default=100000, help="pv num for every iteration")
    parser.add_argument("-r", "--round", type=int, default=1, help="the number of running the program")
    parser.add_argument("-b", "--batch_num", type=int, default=100, help="the number of each eposide")
    parser.add_argument("-j", "--jobs", nargs="+", default=["EGreedy"], help="jobs running this project")
    parser.add_argument("-e", "--epsilon", type=float, default=0.1, help="parameter e for epsilon greedy")
    parser.add_argument("-t", "--ee_type", type=int, default=0, help="EE type, 0, 1, 2")
    parser.add_argument("-a", "--alpha", type=float, default=0.005, help="parameter alpha for tl, rl")
    parser.add_argument("-l", "--lambdaa", type=float, default=0.0, help="parameter lambda for tl")

    args = parser.parse_args()
    parser.print_help()
    param = {"file_path":args.file_path,
             "pv_num": args.pv_num,
             "round": args.round,
             "works": args.jobs,
             "batch_num": args.batch_num,
             "epsilon": args.epsilon,
             "ee": args.ee_type,
             "alpha": args.alpha,
             "lambda": args.lambdaa,
             "bias": 0.0,
             "get_best": 0,
             "stop_iteration": 0,
             "yuzhi":1000000}
    
    creative_filepath = param["file_path"]
    for i in range(param["round"]):
        for work in param["works"]:
            run(creative_filepath, work, tree_filepath, constraint_filepath, param)

    except_reward = []
    param["get_best"] = 1
    for work in param["works"]:
        except_reward.append(0)


    for i in range(len(param["works"])):
        print("-----------", param["works"][i]," ",param["ee"] ,"----------------")
        t = 0
        for k in ans:
            if t % len(param["works"]) != i:
                t += 1
                continue
            for j in k:
                print(j, "|", end=" ")
            print("\n", end="")
            t += 1
        print("---------------except reward ",except_reward[i]," ------------------")

    ISOTIMEFORMAT = '%m%d-%H%M'
    
    import os
    if not os.exists('data/res/'):
        os.mkdir('data/res')
    file = "data/res/ctr_res-" + str(datetime.datetime.now().strftime('%m-%d %H:%M:%S'))+".txt"
    file1 = "data/res/regret_res-" + str(datetime.datetime.now().strftime('%m-%d %H:%M:%S')) + ".txt"
    with open(file, "w") as f:
        for k,v in param.items():
            f.write(k+ " : " +str(v)+'\n')
        for i in range(len(param["works"])):
            s = "-----------"+str(param["works"][i])+" "+str(param["ee"])+"----------------\n"
            f.write(s)
            t = 0
            for k in ans:
                if t % len(param["works"]) != i:
                    t += 1
                    continue
                for j in k:
                    s = str(j) + "| "
                    f.write(s)
                f.write("\n")
                t += 1
    with open(file1, "w") as f:
        for i in range(len(param["works"])):
            s = "-----------"+str(param["works"][i])+" "+str(param["ee"])+"----------------\n"
            f.write(s)
            t = 0
            for k in regret_ans:
                if t % len(param["works"]) != i:
                    t += 1
                    continue
                for j in k:
                    s = str(j) + "| "
                    f.write(s)
                f.write("\n")
                t += 1
