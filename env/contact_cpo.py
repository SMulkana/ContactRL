"""

Date: 2025-10-01

ContactCPO — Constrained Policy Optimization Safe RL Env baseline (Gymnasium + PyBullet)

This environment frames safe contact reaching as a CMDP and exposes explicit safety
**costs** for use with Constrained Policy Optimization (CPO). The agent controls a UR3e
arm (with Robotiq gripper) in PyBullet to approach a hand-shaped target while keeping
contact forces below a comfort limit.

What's modeled here:
- Observation/Action: simple Box observation (EE pose/velocity + target) and 3-DoF EE deltas.
- Reward: reach shaping + optional smoothness/proximity terms; success bonus on contact.
- Safety: per-step binary cost when measured contact force exceeds
  `safety_margin * force_limit` (tightened threshold).
- Accounting: tracks undiscounted episode cost (`episode_cost`) and discounted cost
  (`discounted_episode_cost`) with factor `gamma` for CPO updates.
- Extras: jerk estimation, basic collision checks, and episode statistics.
- Rendering: runs in `p.DIRECT` by default; camera/render helpers included.

"""

import gymnasium as gym
from gymnasium import error, spaces, utils
from gymnasium.utils import seeding
import time
import os
import pybullet as p
import pybullet_data
import math
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
import cv2

MAX_EPISODE_LEN = 150
MODE = p.DIRECT# p.GUI or p.DIRECT - with or without rendering
DIM_OBS = 9 # no. of dimensions in observation space
DIM_ACT = 3 # no. of dimensions in action space 

class ContactCPO(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None, timestep=1/60, output_dir='/path/to/folder/'):
        self.step_counter = 0
        self.cumulative_reward = 0
        self.force_limit = 50
        self.gamma = 0.99
        self.safety_margin = 0.9
        self.ep_cost = 0
        self.disc_cost = 0
        self.discount = 1
        self.mean_cumulative_reward = 0
        self.episode_rewards = []
        self.episodic_reward = []
        self.timestep_counter = []
        self.Reward_reach = []
        self.Reward_safe = []
        self.episode_counter = 0
        self.start_time = time.time()
        self.reward_list = []
        self.episode_end_condition = []
        self.episode_end_list = []
        self.episode_end_reason = []
        self.interaction_forceB_counter = []
        self.contact_counter = []
        self.distance_ee_objB = []
        self.ee_increment = []
        self.total_reward = []
        self.Reward_smooth = []
        self.Reward_proximity = []
        self.TT = []
        self.reach_success = []
        self.Average_episode_reward = []
        self.acc_history = []
        self.jerk_mag = []
        self.rms_jerk = []
        self.acceleration_ee = []
        self.av_reward = 0
        self.prev_action = 0
        self.timestep = timestep  # Adjust the timestep here

        self.forceB = 0
        self.output_dir = output_dir  # Directory to save frames
        
        if not os.path.exists(self.output_dir):
           os.makedirs(self.output_dir)
        
        p.connect(MODE)
        p.setTimeStep(self.timestep)  # Set the simulation timestep
        p.resetDebugVisualizerCamera(cameraDistance=0.5, cameraYaw=0, cameraPitch=-10, cameraTargetPosition=[0.2, -1.5, 0.1])
        self.down_quat = p.getQuaternionFromEuler([math.pi, 0, 0])
        assets = "/path/to/folder/"
        # tell PyBullet where to find all mesh files:
        p.setAdditionalSearchPath(os.path.join(assets, "meshes"))
        self.action_space = spaces.Box(np.array([-0.1]*DIM_ACT), np.array([0.1]*DIM_ACT), dtype=np.float32)
                # Define observation space as a dictionary for HER compatibility
        self.observation_space = spaces.Box(np.array([-1]*DIM_OBS), np.array([1]*DIM_OBS), dtype=np.float32)
      
    def compute_jerk(self, acc_history, dt):

        acc_history = np.array(acc_history)
        if len(acc_history) < 3:
            return [], 0.0  # Not enough data

        # Compute jerk as derivative of acceleration
        jerk = np.diff(acc_history, axis=0) / dt
        jerk_mags = np.linalg.norm(jerk, axis=1)
        rms_jerk = np.sqrt(np.mean(jerk_mags**2))
        return jerk_mags, rms_jerk
    
    def step(self, action):
        reach_success = 0
        reach_bonus = 0
        terminated = False
        truncated = False
        self.step_counter += 1
        hB = 0.02 #height of cubeB which is the hand
        hand_x = self.state_objectB[0]
        hand_y = self.state_objectB[1]
        hand_z = self.state_objectB[2]

        F_N_objB_desirable_max = 50 #N maximumm allowable force on the hand to lift object without causing causing discomfort
  
        reward_reach = 0
        reward_safe = 0
        
        dx, dy, dz = action
        self.x_dir = dx
        self.y_dir = dy
        self.z_dir = dz
                # ─── 1.  build the current 3‑D step vector ──────────────────────────────
        delta_xyz   = np.array([dx, dy, dz])                 # [m]
        pos_ee = np.array(p.getLinkState(self.robot_id, self.ee_link_idx)[0])
        new_pos = pos_ee + np.array([dx, dy, dz])
        orn = p.getQuaternionFromEuler([math.pi, 0, 0])
        full_sol = p.calculateInverseKinematics(self.robot_id, self.ee_link_idx, new_pos.tolist(), orn)

        grip_cmd = 0.2
        arm_targets = list(full_sol[:6])
        info = p.getJointInfo(self.robot_id, self.gripper_joints[0])
        low, high = info[8], info[9]
        frac = (grip_cmd + 1.0) * 0.5
        finger_target = low + frac * (high - low)
        #~~~~~~~ Arm Movement ~~~~~~~~~~~#
        p.setJointMotorControlArray(
        bodyUniqueId    = self.robot_id,
        jointIndices    = self.arm_joints,     # 6 indices
        controlMode     = p.POSITION_CONTROL,
        targetPositions = arm_targets
    )
         #~~~~~~~ Gripper Movement ~~~~~~~~~~~#
        p.setJointMotorControl2(
        bodyUniqueId    = self.robot_id,
        jointIndex      = self.gripper_joints[0],
        controlMode     = p.POSITION_CONTROL,
        targetPosition  = finger_target
    )
        p.stepSimulation()
       
    

# ─── 2.  smoothness: penalise the change in the *vector* itself ─────────
# keep a history of the previous action; reset() should set self.prev_delta = np.zeros(3)
        diff_vec    = delta_xyz - self.prev_delta            # jerk‑like term
        diff_norm   = np.linalg.norm(diff_vec)               # magnitude of that change
        reward_smooth = -100.0 * diff_norm                  # scale factor is up to you

        self.prev_delta = delta_xyz.copy()                   # update for next step

# ─── 3.  proximity: penalise large steps relative to max reach distance ─
        Reward_proximity = -100.0 * (diff_norm / self.max_dist)

        dist_ee_to_objB = math.sqrt((new_pos[0] - hand_x)**2 +(new_pos[1] - hand_y)**2 + (new_pos[2] - hand_z)**2)
        self.Distance = dist_ee_to_objB
        self.distance_ee_objB.append(dist_ee_to_objB)

 
        col_table = p.getContactPoints(self.robot_id, self.table)
        if col_table: 
             terminated = True #terinate if collision with table occurs

        #Contact with hand
        
        reward_reach = -100*(dist_ee_to_objB)

        contact_gripper_objectB = p.getContactPoints(self.robot_id, self.objectBUid) 
        if contact_gripper_objectB:
            finger_object_collision = any((c[3] in self.finger_links) or (c[4] in self.finger_links)for c in contact_gripper_objectB)
            if finger_object_collision:
                 for point in contact_gripper_objectB:
                     self.forceB = point[9]  # Contact normal force
                     reach_bonus = 1000
                #print(f"Contact normal force B: {self.forceB}")
                     #reward_reach = 1000
                     if self.forceB <= F_N_objB_desirable_max:
                         reward_safe = 50*(1-self.forceB/50)
                         self.contact_counter.append('safe_contact')
                     else:
                         reward_safe = -50*(self.forceB-50)/50          
            else:

                 reward_safe = 0 
                 self.forceB = 0
                 self.contact_counter.append('no contact')
        else:

            reward_safe = 0 
            self.forceB = 0
            self.contact_counter.append('no contact')

        if self.step_counter > 3:   
             self_collisions = p.getContactPoints(self.robot_id, self.robot_id)

             if len(self_collisions) > 0:
                 gripper_self_collision = any((c[3] in self.gripper_joints) and (c[4] in self.gripper_joints) for c in self_collisions)
                 gripper_links_collision = any((c[3] in self.gripper_links) and (c[4] in self.gripper_links) for c in self_collisions)
                 if gripper_self_collision:
                     pass 
                 
                 elif gripper_links_collision: 
                      pass

                 else: 
                     terminated = True


        if contact_gripper_objectB:
            finger_object_collision = any((c[3] in self.finger_links) or (c[4] in self.finger_links)for c in contact_gripper_objectB)
            if finger_object_collision:
                terminated = True
                reach_success = 1

        self.reach_success.append(reach_success)
        self.Reward_reach.append(reward_reach)
        self.Reward_safe.append(reward_safe)
        self.Reward_smooth.append(reward_smooth)
        self.Reward_proximity.append(Reward_proximity)
        self.interaction_forceB_counter.append(self.forceB)
        #Capture frame
        if p.getConnectionInfo()['isConnected']:
           self.capture_frame()

        self.timestep_counter.append(self.step_counter)  
        reward_efficiency = Reward_proximity + reward_smooth
        reward_time = - self.step_counter       
        reward = reward_reach + 0*reward_smooth + 0*Reward_proximity + 0*reward_safe  + reach_bonus + 0*reward_time
        
        self.total_reward.append(reward)  # Accumulate rewards
        self.reward_list.append(reward)
        self.cumulative_reward += reward
        
        tight = self.safety_margin * self.force_limit
        cost_force     = 1.0 if self.forceB > tight else 0.0   # binary
        step_cost      = cost_force                           # add other costs here
    # track discounted & undiscounted totals
        self.ep_cost   += step_cost
        self.disc_cost += self.discount * step_cost
        self.discount     *= self.gamma
        info = {}
        info["cost"]                  = float(step_cost)      # per‑step
        info["episode_cost"]          = float(self.ep_cost)   # undiscounted
        info["discounted_episode_cost"] = float(self.disc_cost)
        info["force"]                 = float(self.forceB)

        if self.step_counter >= MAX_EPISODE_LEN:
            truncated = True  
        else:
            truncated = False
        
        
        if terminated or truncated:
            self.av_reward = sum(self.total_reward) / self.step_counter
            self.total_reward = []
            self.episode_rewards.append(self.cumulative_reward)
            self.cumulative_reward = 0
            self.TT.append(self.step_counter)
            self.mean_cumulative_reward = np.mean(self.episode_rewards)
            episode_duration = time.time() - self.start_time
            self.episode_counter += 1
            info['episode_length'] = self.step_counter
            info['episode_number'] = self.episode_counter
            info['episode_duration'] = episode_duration
        else:
            self.av_reward = 0 
            self.TT.append(0)
        self.Average_episode_reward.append(self.av_reward)
        self.episodic_reward.append(self.mean_cumulative_reward)
        
        if terminated: 
            self.episode_end_condition = 'terminated'
        elif truncated: 
            self.episode_end_condition  = 'truncated'
        else:
            self.episode_end_condition = 'episode not end'
        
        self.episode_end_reason.append(self.episode_end_condition)

        
        ee_pos = p.getLinkState(self.robot_id, self.ee_link_idx)[0]
        lin_vel_ee = p.getLinkState(self.robot_id, self.ee_link_idx, computeLinkVelocity=1)[6]
        self.velocity_ee = lin_vel_ee
        if hasattr(self, 'prev_vel_ee'):
            dt = self.timestep
            acc = (np.array(lin_vel_ee) - np.array(self.prev_vel_ee)) / dt
            self.acc_history.append(acc)
        self.prev_vel_ee = lin_vel_ee
        jerk_mags, rms_jerk = self.compute_jerk(self.acc_history, self.timestep)
        self.jerk_mag.append(jerk_mags)
        self.rms_jerk.append(rms_jerk)
        self.acceleration_ee.append(acc[2])
        observation    = np.array(list(ee_pos) + list(self.B) + list(lin_vel_ee), dtype=np.float32) 
        
        pos_ee = np.array(p.getLinkState(self.robot_id, self.ee_link_idx)[0])

        return observation, reward, terminated, truncated, info


    def reset(self, seed=None, options=None):

        super().reset(seed=seed)
        self.acc_history = []
        self.prev_vel_ee = np.zeros(3)
        self.step_counter      = 0
        self.cumulative_reward = 0
        self.forceB            = 0
        self.start_time        = time.time()
        self.ep_cost = 0
        self.disc_cost = 0
        self.discount = 1
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        p.resetSimulation()
        p.setTimeStep(self.timestep)
        p.setGravity(0, 0, -9.81)

        urdfRootPath=pybullet_data.getDataPath()

    # 4) Load plane and table from pybullet_data
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf", basePosition=[0,0,-0.65])
        self.table = p.loadURDF("table/table.urdf", basePosition=[0.5,0,-0.65])      
        
        self.down_quat = p.getQuaternionFromEuler([math.pi, 0, 0])
        assets = "/path/to/folder/"
        p.setAdditionalSearchPath(os.path.join(assets, "meshes"))
        p.setAdditionalSearchPath(assets)
        flags = (
                p.URDF_USE_INERTIA_FROM_FILE
  | p.URDF_USE_SELF_COLLISION
  | p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS
)

        self.robot_id = p.loadURDF(
        os.path.join(assets, "ur3e_robotiq.urdf"),
        basePosition=[0,0,0],
        useFixedBase=True,
        flags=flags
    )

        n_joints = p.getNumJoints(self.robot_id)
        self.arm_joints     = []
        self.gripper_joints = []
        self.gripper_links = []
        for i in range(n_joints):
            info = p.getJointInfo(self.robot_id, i)
            name = info[1].decode("utf-8")
            # UR3e arm joints:
            if name in [
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint"
            ]:
                self.arm_joints.append(i)
            # Robotiq single finger joint:
            if name == "robotiq_finger_joint":
                self.gripper_joints.append(i)

        for i in range(n_joints):
             info = p.getJointInfo(self.robot_id, i)
             link_name = info[12].decode("utf-8")   # index 12 is the link name
             # adjust the condition to match your URDF’s naming
             if "finger" in link_name.lower() or "knuckle" in link_name.lower():
                 self.gripper_links.append(i)

        # combine into one control list (6 arm + 1 finger)
        self.control_joints = self.arm_joints + self.gripper_joints
        
        self.finger_links = [i for i in range(n_joints)
                if 'finger' in p.getJointInfo(self.robot_id,i)[12].decode().lower()]
        self.knuckle_links = [i for i in range(n_joints)
                 if 'knuckle' in p.getJointInfo(self.robot_id,i)[12].decode().lower()]
 
        def get_quaternion_90_x():
    # Roll 90 degrees (π/2 radians), pitch and yaw remain 0
            return p.getQuaternionFromEuler([np.pi / 2, 0, 0])
        #hand
        hB = 0.02 

        objectB = [0.4,0.1, random.uniform(0.03, 0.08)]
        
        # Set orientation to 90 degrees roll along x-axis
        orientation_B = get_quaternion_90_x()
        self.objectBUid = p.loadURDF("/path/to/folder/objectB.urdf", basePosition=objectB, baseOrientation=orientation_B, useFixedBase=True)
        self.state_objectB = p.getBasePositionAndOrientation(self.objectBUid)[0] #returns a tuple where the first element is position and second element is orientation
        
        self.ee_link_idx    = self.arm_joints[-1]  # wrist_3_link
    
        height_offset = 0.40   # e.g. 10 cm above
        objB_z = self.state_objectB[2]
        # set your gripper’s start position
        start_pos = [0.4, 0, objB_z + height_offset]

        lowers, uppers = [], []
        for j in self.arm_joints:
            info = p.getJointInfo(self.robot_id, j)
            lowers.append(info[8])
            uppers.append(info[9])
 
        q_init = [0, -1.57, 1.57, -1.57, -1.57, 1.57]
        for j, q in zip(self.arm_joints, q_init):
             p.resetJointState(self.robot_id, j, q)

        g_low, g_high = p.getJointInfo(self.robot_id, self.gripper_joints[0])[8:10]
        g_mid = 0.5*(g_low+g_high)
        p.resetJointState(self.robot_id, self.gripper_joints[0], g_mid)
     
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        self.step_counter = 0
    # 5) re-initialize your state variables from these reset angles
        self.state_ee       = start_pos + np.array([0,0,0.153])
        self.state_gripper  = g_mid
        self.ee             = list(start_pos)
        self.B = self.state_objectB
        #print('Position of hand:', self.B) 
        self.prev_delta = np.zeros(3, dtype=np.float32)
        lin_vel_ee = p.getLinkState(self.robot_id, self.ee_link_idx, computeLinkVelocity=1)[6]
    # 6) return your observation (e.g. [x,y,z, gripper_pos]) and info dict
        observation = np.array(list(self.state_ee) + list(self.B) + list(lin_vel_ee), dtype=np.float32)
        self.max_dist = math.dist(self.state_ee, self.state_objectB)

        info = {}

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    
    def get_distance(self):
        #setting camera's position, orientation and field of view
        camera_position = [0.1, 0, 0.05]  # 10 cm forward from end effector
        camera_orientation = p.getQuaternionFromEuler([0, np.pi, 0])  # Facing forward from end effector
        camera_target = [0, 0, 0.5]
        up_vector = [0, 0, 1]
        fov = 60  # Field of view
        aspect = 1.0
        near = 0.01
        far = 2
        width, height = 640, 480

        # Camera matrices
        view_matrix = p.computeViewMatrix(camera_position, camera_target, up_vector)
        projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)
        
        image = p.getCameraImage(width, height, view_matrix, projection_matrix, renderer=p.ER_TINY_RENDERER)
        rgb_image = np.reshape(image[2], (height, width, 4))[:, :, :3]  # Only RGB channels
        depth_buffer = np.reshape(image[3], (height, width))
        depth_image = far * near / (far - (far - near) * depth_buffer)
 # Convert RGB image to grayscale and apply simple threshold for object detection
        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        _, thresholded = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
# Find contours to detect objects
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            # Calculate bounding box of the object
            x, y, w, h = cv2.boundingRect(contour)
            # Calculate the center of the bounding box
            center_x, center_y = x + w // 2, y + h // 2
            
            # Get distance from the depth image at the bounding box center
            distance_to_object = depth_image[center_y, center_x]
            
            # Draw bounding box and distance on RGB image
            cv2.rectangle(rgb_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(rgb_image, f"{distance_to_object:.2f} m", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Display the processed image
        cv2.imshow("Detected Object with Distance", rgb_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return(distance_to_object)
    
    def render(self, mode='human'):
        view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.2,-1.5,0.1],
                                                            distance=0.5,
                                                            yaw=0,
                                                            pitch=-10,
                                                            roll=0,
                                                            upAxisIndex=2)
 
        proj_matrix = p.computeProjectionMatrixFOV(fov=60,
                                                     aspect=float(960) /720,
                                                     nearVal=0.1,
                                                     farVal=100.0)
        (_, _, px, _, _) = p.getCameraImage(width=960,
                                              height=720,
                                              viewMatrix=view_matrix,
                                              projectionMatrix=proj_matrix,
                                              renderer=p.ER_BULLET_HARDWARE_OPENGL)

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (720,960, 4))

        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def _get_state(self):
       return self.observation

    def capture_frame(self):
      frame = self.render(mode='rgb_array')
      img = Image.fromarray(frame)
      img.save(f"{self.output_dir}/frame_{self.step_counter:04d}.png")

    def close(self):
        p.disconnect()


