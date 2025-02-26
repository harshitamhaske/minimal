"""
CODE BASED ON EXAMPLE FROM:

https://github.com/OpenQuadruped/spot_mini_mini/blob/spot/spot_bullet/paper/GMBC_data_plotter.py
"""
#!/usr/bin/env python

import numpy as np
import sys
import os
import argparse
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import copy
from scipy.stats import norm
sns.set()

# ARGUMENTS
descr = "Spot Mini Mini ARS Agent Evaluator."
parser = argparse.ArgumentParser(description=descr)
parser.add_argument("-nep",
                    "--NumberOfEpisodes",
                    help="Number of Episodes to Plot Data For")
parser.add_argument("-maw",
                    "--MovingAverageWindow",
                    help="Moving Average Window for Plotting (Default: 50)")
parser.add_argument("-surv",
                    "--Survival",
                    help="Plot Survival Curve",
                    action='store_true')
parser.add_argument("-tr",
                    "--TrainingData",
                    help="Plot Training Curve",
                    action='store_true')
parser.add_argument("-tot",
                    "--TotalReward",
                    help="Show Total Reward instead of Reward Per Timestep",
                    action='store_true')
parser.add_argument("-ar",
                    "--RandAgentNum",
                    help="Randomized Agent Number To Load")
parser.add_argument("-raw",
                    "--Raw",
                    help="Plot Raw Data in addition to Moving Averaged Data",
                    action='store_true')
parser.add_argument(
    "-s",
    "--Seed",
    help="Seed [UP TO, e.g. 0 | 0, 1 | 0, 1, 2 ...] (Default: 0).")

parser.add_argument(
    "-tru",
    "--TrueAct",
    help="Plot the Agent Action instead of what the robot sees",
    action='store_true')
ARGS = parser.parse_args()

MA_WINDOW = 50
if ARGS.MovingAverageWindow:
    MA_WINDOW = int(ARGS.MovingAverageWindow)


def moving_average(a, n=MA_WINDOW):
    MA = np.cumsum(a, dtype=float)
    MA[n:] = MA[n:] - MA[:-n]
    return MA[n - 1:] / n


def extract_data_bounds(min=0, max=5, dist_data=None, dt_data=None):
    """ 3 bounds: lower, mid, highest
    """

    if dist_data is not None:

        # Get Survival Data, dt
        # Lowest Bound: x <= max
        bound = np.array([0])
        if min == 0:
            less_max_cond = dist_data <= max
            bound = np.where(less_max_cond)
        else:
            # Highest Bound: min <= x
            if max == np.inf:
                gtr_min_cond = dist_data >= min
                bound = np.where(gtr_min_cond)
            # Mid Bound: min < x < max
            else:
                less_max_gtr_min_cond = np.logical_and(dist_data > min,
                                                       dist_data < max)
                bound = np.where(less_max_gtr_min_cond)

        if dt_data is not None:
            dt_bounded = dt_data[bound]
            num_surv = np.array(np.where(dt_bounded == 50000))[0].shape[0]
        else:
            num_surv = None

        return dist_data[bound], num_surv
    else:
        return None


def main():
    """ The main() function. """
    file_name = "spot_ars_"

    seed = 0
    if ARGS.Seed:
        seed = ARGS.Seed

    # Find abs path to this file
    my_path = os.path.abspath(os.path.dirname(__file__))
    results_path = os.path.join(my_path, "../results")

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    vanilla_surv = np.random.randn(1000)
    agent_surv = np.random.randn(1000)

    nep = 1000

    if ARGS.NumberOfEpisodes:
        nep = ARGS.NumberOfEpisodes
    if ARGS.TrainingData:
        training = True
    else:
        training = False
    if ARGS.Survival:
        surv = True
    else:
        surv = False
    if not surv and not training:
        print(
            "Please Select which Data you would like to plot (-pout | -surv | -tr)"
        )
    rand_agt = 799
   
    if ARGS.RandAgentNum:
        rand_agt = ARGS.RandAgentNum
  

    if surv:

        survival_file = results_path + "/" + file_name + "agent_{}".format(rand_agt) + '_survival_{}'.format(nep)
        
        if os.path.exists(survival_file):
            with open(survival_file, 'rb') as filehandle:
                random_surv = np.array(pickle.load(filehandle))
        else:
            print(f"Error: The file {survival_file} does not exist.")
            return  

       

        # Extract useful values
        
        random_surv_x = random_surv[:, 0]
        
        # convert the lists to series
        data = {
           
            'Random': random_surv_x
        }

        colors = ['r', 'g', 'b']

        # get dataframe
        df = pd.DataFrame(data)
        print(df)

        # get dataframe2
        # Extract useful values
       
        random_surv_dt = random_surv[:, -1]
       
        # convert the lists to series
        data2 = {
           
            'Random': random_surv_dt
        }
        df2 = pd.DataFrame(data2)

        # Plot
        for i, col in enumerate(df.columns):
            sns.distplot(df[[col]], color=colors[i])

        plt.legend(labels=['IMU_random'])
        plt.xlabel("Forward Survived Distance (m)")
        plt.ylabel("Kernel Density Estimate")
        plt.show()

       
        rand_avg = np.average(copy.deepcopy(random_surv_x))
        rand_std = np.std(copy.deepcopy(random_surv_x))
       
      
        print("IMU_random: AVG [{}] | STD [{}] AMOUNT [{}]".format(
            rand_avg, rand_std, random_surv_x.shape[0]))
        

        # collect data
      
        random_surv_x_less_5, random_surv_num_less_5 = extract_data_bounds(
            0, 5, random_surv_x, random_surv_dt)
       
        # <=5
        # Make sure all arrays filled
        
        if random_surv_x_less_5.size == 0:
            random_surv_x_less_5 = np.array([0])
      

        
        rand_avg = np.average(random_surv_x_less_5)
        rand_std = np.std(random_surv_x_less_5)
       
        print("<= 5m")
        
        print(
            "IMU_random: AVG [{}] | STD [{}] AMOUNT DEAD [{}] | AMOUNT ALIVE [{}]"
            .format(rand_avg, rand_std,
                    random_surv_x_less_5.shape[0] - random_surv_num_less_5,
                    random_surv_num_less_5))
        
        # collect data
       
        random_surv_x_gtr_5, random_surv_num_gtr_5 = extract_data_bounds(
            5, 90, random_surv_x, random_surv_dt)
        
        # >5 <90
        # Make sure all arrays filled
       
        if random_surv_x_gtr_5.size == 0:
            random_surv_x_gtr_5 = np.array([0])
        

      
        rand_avg = np.average(random_surv_x_gtr_5)
        rand_std = np.std(random_surv_x_gtr_5)
       
        print("> 5m and <90m")
       
        print(
            "IMU_random: AVG [{}] | STD [{}] AMOUNT DEAD [{}] | AMOUNT ALIVE [{}]"
            .format(rand_avg, rand_std,
                    random_surv_x_gtr_5.shape[0] - random_surv_num_gtr_5,
                    random_surv_num_gtr_5))
       
        # collect data
       
        random_surv_x_gtr_90, random_surv_num_gtr_90 = extract_data_bounds(
            90, np.inf, random_surv_x, random_surv_dt)
       

        # >90
        # Make sure all arrays filled
       
        if random_surv_x_gtr_90.size == 0:
            random_surv_x_gtr_90 = np.array([0])
       

       
        rand_avg = np.average(random_surv_x_gtr_90)
        rand_std = np.std(random_surv_x_gtr_90)
        
        print(">= 90m")
       
        print(
            "IMU_random: AVG [{}] | STD [{}] AMOUNT DEAD [{}] | AMOUNT ALIVE [{}]"
            .format(rand_avg, rand_std,
                    random_surv_x_gtr_90.shape[0] - random_surv_num_gtr_90,
                    random_surv_num_gtr_90))
       
        # Save to excel
        df.to_excel(results_path + "/SurvDist.xlsx", index=False)
        df2.to_excel(results_path + "/SurvDT.xlsx", index=False)

    elif training:
        rand_data_list = []
        
        rand_shortest_length = np.inf
       
        for i in range(int(seed) + 1):
            # Training Data Plotter
            rand_data_temp = np.load(results_path + "/spot_ars_rand_" +
                                     "seed" + str(i) + ".npy")
           

            rand_shortest_length = min(
                np.shape(rand_data_temp[:, 1])[0], rand_shortest_length)
           
            rand_data_list.append(rand_data_temp)
          

        tot_rand_data = []
      
        norm_rand_data = []
     
        for i in range(int(seed) + 1):
            tot_rand_data.append(
                moving_average(rand_data_list[i][:rand_shortest_length, 0]))
            
            norm_rand_data.append(
                moving_average(rand_data_list[i][:rand_shortest_length, 1]))
            

        tot_rand_data = np.array(tot_rand_data)
        
        norm_rand_data = np.array(norm_rand_data)
       

        # column-wise
        axis = 0

        # MEAN
        tot_rand_mean = tot_rand_data.mean(axis=axis)
       
        norm_rand_mean = norm_rand_data.mean(axis=axis)
       
        # STD
        tot_rand_std = tot_rand_data.std(axis=axis)
        
        norm_rand_std = norm_rand_data.std(axis=axis)
        
        aranged_rand = np.arange(np.shape(tot_rand_mean)[0])
        
        if ARGS.TotalReward:
            if ARGS.Raw:
                plt.plot(rand_data_list[0][:, 0],
                         label="Randomized (Total Reward)",
                         color='g')
               
            plt.plot(aranged_rand,
                     tot_rand_mean,
                     label="MA: Randomized (Total Reward)",
                     color='g')
            plt.fill_between(aranged_rand,
                             tot_rand_mean - tot_rand_std,
                             tot_rand_mean + tot_rand_std,
                             color='g',
                             alpha=0.2)
        else:
            if ARGS.Raw:
                plt.plot(rand_data_list[0][:, 1],
                         label="Randomized (Reward/dt)",
                         color='g')
               
            plt.plot(aranged_rand,
                     norm_rand_mean,
                     label="IMU (Reward/dt)",
                     color='g')
            plt.fill_between(aranged_rand,
                             norm_rand_mean - norm_rand_std,
                             norm_rand_mean + norm_rand_std,
                             color='g',
                             alpha=0.2)
        plt.xlabel("Epoch #")
        plt.ylabel("Reward")
        plt.title(
            "Training Performance with {} seed samples".format(int(seed) + 1))
        plt.legend()
        plt.show()

    
if __name__ == '__main__':
    main()
