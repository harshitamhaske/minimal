import pybullet as p
import time
import pybullet_data

p.connect(p.GUI)

p.setAdditionalSearchPath(pybullet_data.getDataPath())

robot_id = p.loadURDF("")

num_joints = p.getNumJoints(robot_id)

print(f"Number of joints in the robot: {num_joints}")

for joint_index in range(num_joints):
    joint_info = p.getJointInfo(robot_id, joint_index)
    joint_id = joint_info[0] 
    joint_name = joint_info[1].decode('utf-8') 
    print(f"Joint Index: {joint_index}, Joint ID: {joint_id}, Joint Name: {joint_name}")

time.sleep(10)

p.disconnect()

