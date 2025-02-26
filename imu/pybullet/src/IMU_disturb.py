#!/usr/bin/env python

import numpy as np
import pybullet as p
import csv
import gymnasium as gym

import sys

import csv 

sys.path.append('../../')

from ars_lib.ars import ARSAgent, Normalizer, Policy
from spotmicro.util.gui import GUI
from spotmicro.Kinematics.SpotKinematics import SpotModel
from spotmicro.GaitGenerator.Bezier import BezierGait
from spotmicro.OpenLoopSM.SpotOL import BezierStepper
from spotmicro.GymEnvs.spot_bezier_env import spotBezierEnv
from spotmicro.spot_env_randomizer import SpotEnvRandomizer

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import os

import argparse

# ARGUMENTS
descr = "Spot Mini Mini ARS Agent Evaluator."
parser = argparse.ArgumentParser(description=descr)
parser.add_argument("-hf",
                    "--HeightField",
                    help="Use HeightField",
                    action='store_true')
parser.add_argument("-nr",
                    "--DontRender",
                    help="Don't Render environment",
                    action='store_true')
parser.add_argument("-r",
                    "--DebugRack",
                    help="Put Spot on an Elevated Rack",
                    action='store_true')
parser.add_argument("-p",
                    "--DebugPath",
                    help="Draw Spot's Foot Path",
                    action='store_true')
parser.add_argument("-gui",
                    "--GUI",
                    help="Control The Robot Yourself With a GUI",
                    action='store_true')
parser.add_argument("-nc",
                    "--NoContactSensing",
                    help="Disable Contact Sensing",
                    action='store_true')
parser.add_argument("-a", "--AgentNum", help="Agent Number To Load")
parser.add_argument("-pp",
                    "--PlotPolicy",
                    help="Plot Policy Output after each Episode.",
                    action='store_true')
parser.add_argument("-ta",
                    "--TrueAction",
                    help="Plot Action as seen by the Robot.",
                    action='store_true')
parser.add_argument(
    "-save",
    "--SaveData",
    help="Save the Policy Output to a .npy file in the results folder.",
    action='store_true')
parser.add_argument("-s", "--Seed", help="Seed (Default: 0).")
ARGS = parser.parse_args()

def print_space_details(space, space_type):
    print(f"{space_type} Space Details:")
    if isinstance(space, gym.spaces.Box):
        print("Type: Box")
        print("Shape:", space.shape)
        print("Low values:", space.low)
        print("High values:", space.high)
    elif isinstance(space, gym.spaces.Discrete):
        print("Type: Discrete")
        print("Number of discrete values:", space.n)
    elif isinstance(space, gym.spaces.Dict):
        print("Type: Dict")
        for key, subspace in space.spaces.items():
            print(f"  Key: {key}")
            print("    Shape:", subspace.shape)
            if isinstance(subspace, gym.spaces.Box):
                print("    Low values:", subspace.low)
                print("    High values:", subspace.high)
            elif isinstance(subspace, gym.spaces.Discrete):
                print("    Number of discrete values:", subspace.n)
    else:
        print("Unknown space type")

def drop_bouncing_ball_from_side(env, ball_mass=5.0, ball_radius=0.02, side_position=-0.4, drop_height=0.25, restitution=0.9):
   
    ball_start_position = [0, side_position, drop_height]  
    ball_start_orientation = p.getQuaternionFromEuler([0, 0.5, 0])

    ball_visual_shape_id = p.createVisualShape(shapeType=p.GEOM_SPHERE, 
                                               radius=ball_radius, 
                                               rgbaColor=[0, 0, 1, 1])  
    ball_collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_SPHERE, 
                                                     radius=ball_radius)
    
    ball_id = p.createMultiBody(baseMass=ball_mass,
                                baseCollisionShapeIndex=ball_collision_shape_id,
                                baseVisualShapeIndex=ball_visual_shape_id,
                                basePosition=ball_start_position,
                                baseOrientation=ball_start_orientation)
    
    p.changeDynamics(ball_id, -1, restitution=restitution, lateralFriction=0.5, spinningFriction=0.5, rollingFriction=0.1)

    p.resetBaseVelocity(ball_id, linearVelocity=[0, 3, 0])  

    print(f"Stronger side ball created with ID {ball_id} at position {side_position} meters with velocity towards the robot.")
    return ball_id



def main():
    """ The main() function. """

    print("STARTING MINITAUR ARS")

    # TRAINING PARAMETERS
    # env_name = "MinitaurBulletEnv-v0"
    seed = 0
    if ARGS.Seed:
        seed = ARGS.Seed

    max_timesteps = 4e6
    file_name = "spot_ars_"

    if ARGS.DebugRack:
        on_rack = True
    else:
        on_rack = False

    if ARGS.DebugPath:
        draw_foot_path = True
    else:
        draw_foot_path = False

    if ARGS.HeightField:
        height_field = True
    else:
        height_field = False

    if ARGS.NoContactSensing:
        contacts = False
    else:
        contacts = True

    if ARGS.DontRender:
        render = False
    else:
        render = True

    env_randomizer = SpotEnvRandomizer()


    

    # Find abs path to this file
    my_path = os.path.abspath(os.path.dirname(__file__))
    results_path = os.path.join(my_path, "../results")
    if contacts:
        models_path = os.path.join(my_path, "../models/contact")
    else:
        models_path = os.path.join(my_path, "../models/no_contact")

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    if not os.path.exists(models_path):
        os.makedirs(models_path)

    
    env = spotBezierEnv(render=render,
                        on_rack=on_rack,
                        height_field=height_field,
                        draw_foot_path=draw_foot_path,
                        contacts=contacts,
                        env_randomizer=env_randomizer)

   
   # Set seeds
    env.seed(seed)
    np.random.seed(seed)

   
     # Print details of the observation space
    obs_space = env.observation_space
    print_space_details(obs_space, "Observation")

    # Print details of the action space
    action_space = env.action_space
    print_space_details(action_space, "Action")

    foot_joint_indices = {
        'back_left_leg_foot': 9,
        'front_left_leg_foot': 17,
        'front_right_leg_foot': 5,
        'back_right_leg_foot': 13
    }

    # Enable joint force/torque sensors
    for foot_joint_index in foot_joint_indices.values():
        p.enableJointForceTorqueSensor(env.spot.quadruped, foot_joint_index, enableSensor=True)

    print("Sensors have been initialized.")

    # Set seeds
    env.seed(seed)
    np.random.seed(seed)

    state_dim = env.observation_space.shape[0]
    print("STATE DIM: {}".format(state_dim))
    action_dim = env.action_space.shape[0]
    print("ACTION DIM: {}".format(action_dim))
    max_action = float(env.action_space.high[0])

    env.reset()

    spot = SpotModel()

    bz_step = BezierStepper(dt=env._time_step)
    bzg = BezierGait(dt=env._time_step)

    # Initialize Normalizer
    normalizer = Normalizer(state_dim)

    # Initialize Policy
    policy = Policy(state_dim, action_dim)

    # to GUI or not to GUI
    if ARGS.GUI:
        gui = True
    else:
        gui = False

    # Initialize Agent with normalizer, policy and gym env
    agent = ARSAgent(normalizer, policy, env, bz_step, bzg, spot, gui)
    agent_num = 0
    if ARGS.AgentNum:
        agent_num = ARGS.AgentNum
    if os.path.exists(models_path + "/" + file_name + str(agent_num) +
                      "_policy"):
        print("Loading Existing agent")
        agent.load(models_path + "/" + file_name + str(agent_num))
        agent.policy.episode_steps = np.inf
        policy = agent.policy

  
    env.reset()
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    t = 0
    
    ball_id = None
    side_ball_id = None

    # Define simulation parameters
    time_to_drop = 2 
    timestep = env._time_step 
    drop_timesteps = int(time_to_drop / timestep)

    print("STARTED MINITAUR TEST SCRIPT")
     
    

  
    while t < (int(max_timesteps)):

        #ball_id = drop_bouncing_ball(env)
        #print(f"Bouncing ball with ID {ball_id} has been dropped from above.")

       
        side_ball_id = drop_bouncing_ball_from_side(env)
        print(f"Side ball with ID {side_ball_id} has been dropped from the side.")

        episode_reward, episode_timesteps = agent.deployTG()
  
        t += episode_timesteps
        # episode_reward = agent.train()
        # +1 to account for 0 indexing.
        # +0 on ep_timesteps since it will increment +1 even if done=True
        print("Total T: {} Episode Num: {} Episode T: {} Reward: {}".format(
            t, episode_num, episode_timesteps, episode_reward))
        episode_num += 1
    

    env.close()

    
if __name__ == '__main__':
    main()
