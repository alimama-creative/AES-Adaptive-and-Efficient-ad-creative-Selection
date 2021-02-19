## Code and Data for paper Efficient Optimal Selection for Composited Advertising Creatives with Tree Structure
### Data
We provide the generated synthetic data for testing in 'data' directory.

+ 'tree_constraint.txt' details the constraints between ingredients with adjacency matrix. 
    + '1' represents feasible connections
    + '0' represents illegal connections
+ 'tree_struct2.txt' details the *Ingredient Tree*
+ 'ctr_new.txt'/'ctr_new2.txt' details the simulated data, where each row is formulated as 'ID + element list + Generated CTR'

### Online Raw Data
'data/ctr_online.txt' has same format with ctr_new.txt/ctr_new2.txt
The element list contains
+ ID for the template   : {0}
+ ID for the background : {0: dark, 1:light}
+ ID for picture sizes  : {0: 88%, 1:91%, 2:94%, 3:97%, 4:100%}
+ ID for text color     : {0,1,2,3,4,5,6,7}, 0~3 for the dark background, 4~7 for the light background
+ ID for text font      : {0,1,2,3}



### Code
Run an example with   `python run.py -b 1000 -p 1000000 -j AES -r 20`
+ 'run.py' is the main function and different configurations can be found here for tuning
+ 'policy' directory contains different methods

#### Parameters
+ `-f` file path of creatives
+ `-p` total pv
+ `-r` multiple running repetitions
+ `-b` batch size(update intervals)
+ `-j` method : EGreedy,thompson,ucb,IndEgreedy,Edge_TS,LinUCB,TEgreedy,Full_TS,Edge_TS,MVT,AES(proposed method)
+ `-e` parameter for egreedy
+ `-t` EE type : 0,1,2
    + 0: regular exploration
    + 1: exploration after DP
    + 2: exploration during DP
+ `-a` parameter for thompson methods


### Generate Simulation Data
Run `python gen_ctr.py`
+ the generated file is saved in `"data/ctr_new2.txt"`

