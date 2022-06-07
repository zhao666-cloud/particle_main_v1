import os
import time
import cv2
import numpy as np
import pybullet as p
import math
from math import tan,cos,sin
import torch
import matplotlib.pyplot as plt
from collections import namedtuple
import torch.nn.functional as F
from kernel.args import get_args
PATH = os.path.dirname(__file__)
Up = [0, 0, 1]
pixelWidth = 256
pixelHeight = 192
aspect = pixelWidth / pixelHeight
nearPlane = 0.01
farPlane = 1000
fov = 90

class RobotBase(object):
    """
    The base class for robots
    """

    def __init__(self, pos, ori):
        """
        Arguments:
            pos: [x y z]
            ori: [r p y]

        Attributes:
            id: Int, the ID of the robot
            eef_id: Int, the ID of the End-Effector
            arm_num_dofs: Int, the number of DoFs of the arm
                i.e., the IK for the EE will consider the first `arm_num_dofs` controllable (non-Fixed) joints
            joints: List, a list of joint info
            controllable_joints: List of Ints, IDs for all controllable joints
            arm_controllable_joints: List of Ints, IDs for all controllable joints on the arm (that is, the first `arm_num_dofs` of controllable joints)

            ---
            For null-space IK
            ---
            arm_lower_limits: List, the lower limits for all controllable joints on the arm
            arm_upper_limits: List
            arm_joint_ranges: List
            arm_rest_poses: List, the rest position for all controllable joints on the arm

            gripper_range: List[Min, Max]
        """
        self.args = get_args()
        self.base_pos = pos
        self.base_ori = p.getQuaternionFromEuler(ori)

    def load(self):
        self.__init_robot__()
        self.__parse_joint_info__()
        self.__post_load__()
        #print(self.joints)

    def step_simulation(self):
        raise RuntimeError('`step_simulation` method of RobotBase Class should be hooked by the environment.')

    def __parse_joint_info__(self):
        numJoints = p.getNumJoints(self.id)
        jointInfo = namedtuple('jointInfo', 
            ['id','name','type','damping','friction','lowerLimit','upperLimit','maxForce','maxVelocity','controllable'])
        self.joints = []
        self.controllable_joints = []
        for i in range(numJoints):
            info = p.getJointInfo(self.id, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = info[2]  # JOINT_REVOLUTE, JOINT_PRISMATIC, JOINT_SPHERICAL, JOINT_PLANAR, JOINT_FIXED
            jointDamping = info[6]
            jointFriction = info[7]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            controllable = (jointType != p.JOINT_FIXED)
            if controllable:
                self.controllable_joints.append(jointID)
                p.setJointMotorControl2(self.id, jointID, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
            info = jointInfo(jointID,jointName,jointType,jointDamping,jointFriction,jointLowerLimit,
                            jointUpperLimit,jointMaxForce,jointMaxVelocity,controllable)
            self.joints.append(info)

        assert len(self.controllable_joints) >= self.arm_num_dofs
        self.arm_controllable_joints = self.controllable_joints[:self.arm_num_dofs]

        self.arm_lower_limits = [info.lowerLimit for info in self.joints if info.controllable][:self.arm_num_dofs]
        self.arm_upper_limits = [info.upperLimit for info in self.joints if info.controllable][:self.arm_num_dofs]
        self.arm_joint_ranges = [info.upperLimit - info.lowerLimit for info in self.joints if info.controllable][:self.arm_num_dofs]

    def __init_robot__(self):
        raise NotImplementedError
    
    def __post_load__(self):
        pass

    def reset_robot(self):
        self.reset_arm()
        self.reset_gripper()

    def reset_arm(self):
        """
        reset to rest poses
        """
        for rest_pose, joint_id in zip(self.arm_rest_poses, self.arm_controllable_joints):
            p.resetJointState(self.id, joint_id, rest_pose)

        # Wait for a few steps
        for _ in range(10):
            self.step_simulation()

    def reset_gripper(self):
        self.open_gripper()

    def open_gripper(self):
        self.move_gripper(self.gripper_range[1])

    def close_gripper(self):
        self.move_gripper(self.gripper_range[0])

    def move_ee(self, action):
        steps = 0
        x, y, z, roll, pitch, yaw = action
        pos = (x, y, z)
        orn = p.getQuaternionFromEuler((roll, pitch, yaw))
        while steps < 1000:
            joint_poses = p.calculateInverseKinematics(self.id, self.eef_id, pos, orn,
                                                       self.arm_lower_limits, self.arm_upper_limits, self.arm_joint_ranges, self.arm_rest_poses)
            # arm
            for i, joint_id in enumerate(self.arm_controllable_joints):
                p.setJointMotorControl2(self.id, joint_id, p.POSITION_CONTROL, joint_poses[i],
                                    force=self.joints[joint_id].maxForce, maxVelocity=self.joints[joint_id].maxVelocity/10)
            for _ in range(120):
                p.stepSimulation()
            steps += 1
            ls = self.get_joint_obs()
            curr_pos = np.array(ls[0])
            target_pos = np.array(pos)
            distance = np.linalg.norm(target_pos - curr_pos)
            ori_dis = np.linalg.norm(np.array(ls[1]) - np.array(orn))
            if distance < 0.002 and ori_dis < 0.005:
                break

    def move_gripper(self, open_length):
        raise NotImplementedError

    def get_joint_obs(self):
        ee_pos = p.getLinkState(self.id, self.eef_id)[0]
        ee_ori = p.getLinkState(self.id,self.eef_id)[1]
        return ee_pos,ee_ori


class UR5Robotiq85(RobotBase):
    def __init_robot__(self):
        self.eef_id = 7
        self.arm_num_dofs = 6
        self.arm_rest_poses = [0, -1.5446774605904932, 1.343946009733127, -1.3708613585093699,
                               -1.5707970583733368, 0.0009377758247187636]
        self.pixelWidth = 256
        self.pixelHeight = 192
        self.id = p.loadURDF(f'{PATH}/urdf/ur5_robotiq_85.urdf', self.base_pos, self.base_ori,
                             useFixedBase=True, flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
        self.gripper_range = [0, 0.085]
    
    def __post_load__(self):
        # To control the gripper
        mimic_parent_name = 'finger_joint'
        mimic_children_names = {'right_outer_knuckle_joint': 1,
                                'left_inner_knuckle_joint': 1,
                                'right_inner_knuckle_joint': 1,
                                'left_inner_finger_joint': -1,
                                'right_inner_finger_joint': -1}
        self.__setup_mimic_joints__(mimic_parent_name, mimic_children_names)

    def __setup_mimic_joints__(self, mimic_parent_name, mimic_children_names):
        self.mimic_parent_id = [joint.id for joint in self.joints if joint.name == mimic_parent_name][0]
        self.mimic_child_multiplier = {joint.id: mimic_children_names[joint.name] for joint in self.joints if joint.name in mimic_children_names}

        for joint_id, multiplier in self.mimic_child_multiplier.items():
            c = p.createConstraint(self.id, self.mimic_parent_id,
                                   self.id, joint_id,
                                   jointType=p.JOINT_GEAR,
                                   jointAxis=[0, 1, 0],
                                   parentFramePosition=[0, 0, 0],
                                   childFramePosition=[0, 0, 0])
            p.changeConstraint(c, gearRatio=-multiplier, maxForce=100, erp=1)  # Note: the mysterious `erp` is of EXTREME importance

    def move_gripper(self, open_length):
        # open_length = np.clip(open_length, *self.gripper_range)
        open_angle = 0.715 - math.asin((open_length - 0.010) / 0.1143)  # angle calculation
        # Control the mimic gripper joint(s)
        p.setJointMotorControl2(self.id, self.mimic_parent_id, p.POSITION_CONTROL, targetPosition=open_angle,
                                force=self.joints[self.mimic_parent_id].maxForce, maxVelocity=self.joints[self.mimic_parent_id].maxVelocity)
    def load_obj(self,scale = 1.0):
        obj_dirs = os.listdir(f'{PATH}/{self.args.objects_dir}')
        self.obj_list = []
        self.obj_id_list = []
        for obj_dir in obj_dirs:
            if "_vhacd" not in obj_dir:
                self.obj_list.append(obj_dir)
        self.obj_list.sort()
        list = np.random.choice(len(self.obj_list), self.args.num_obj, replace=False)#num
        obj_pos = np.random.random(2) * 0.01 + 0.5
        obj_pos = np.concatenate((obj_pos,[0.3]))
        obj_ori = p.getQuaternionFromEuler([90,90,90])
        for i in list:
            obj_path = f"{PATH}/{self.args.objects_dir}/{self.obj_list[i]}/textured.obj"#{self.obj_list[i]}
            vhacd_obj_path = obj_path.replace(".obj","_vhacd.obj")
            if not os.path.exists(vhacd_obj_path):
                p.vhacd(obj_path, vhacd_obj_path, "vhacd_log.txt",alpha=0.04, resolution=100000)
            collisionShapedId = p.createCollisionShape(shapeType=p.GEOM_MESH,
                                                       fileName=vhacd_obj_path,
                                                       meshScale=[scale for i in range(3)])
            self.obj_id = p.createMultiBody(baseMass=1,
                              baseCollisionShapeIndex = collisionShapedId,
                              basePosition=obj_pos,
                              baseOrientation=obj_ori)
            for _ in range(2000):
                p.stepSimulation()
            box = np.array(p.getAABB(self.obj_id))
            p.removeBody(self.obj_id)
            for _ in range(2000):
                p.stepSimulation()
            # make sure object is not too big
            diagnal = np.linalg.norm(box[1] - box[0])
            scale = pow(np.tan(2 * diagnal - np.pi / 2 - 0.5), -1) + 1
            visualShapeId = p.createVisualShape(shapeType=p.GEOM_MESH,
                                                fileName=obj_path,
                                                meshScale=[scale for i in range(3)])
            collisionShapedId = p.createCollisionShape(shapeType=p.GEOM_MESH,
                                                       fileName=vhacd_obj_path,
                                                       meshScale=[scale for i in range(3)])
            self.obj_id = p.createMultiBody(baseMass=1,
                              baseCollisionShapeIndex = collisionShapedId,
                              baseVisualShapeIndex= visualShapeId,
                              basePosition=obj_pos,
                              baseOrientation=obj_ori)
            self.obj_id_list.append(self.obj_id)
            for _ in range(2000):
                p.stepSimulation()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

    def show_particle(self,particles,rgbaColor=[0,170,20,1]):
        show_points = particles.clone()
        show_points[:, 0] += 0.2
        show_points[:, 1] += 0.2
        show_points[:, 2] += 0.1
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)
        visualShapedId = p.createVisualShape(shapeType=p.GEOM_SPHERE,rgbaColor=rgbaColor,radius=0.001)
        particle_id_list = []
        for points in show_points:
            obj_id = p.createMultiBody(baseMass=0,
                                       baseCollisionShapeIndex=-1,
                                       baseVisualShapeIndex=visualShapedId,
                                       basePosition=points[:3])
            particle_id_list.append(obj_id)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)
        p.removeAllUserDebugItems()
        time.sleep(0.1)
        return particle_id_list
    def show_part_particle(self,particles,seg_sign,rgbaColor=[0,170,20,1]):
        color_bar = {
            0 : [230/255,95/255,84/255,1],
            1 : [155/255,101/255,240/255,1],
            2 : [102/255,203/255,217/255,1],
            3 : [140/255,240/255,101/255,1],
            4 : [230/255,193/255,96/255,1]
        }
        show_points = particles.clone()
        unique_sign = np.unique(seg_sign)
        color_sign = {}
        for i,sign in enumerate(unique_sign):
            color_sign[sign] = color_bar[i]
        show_points[:, 0] += 0.2
        show_points[:, 1] += 0.2
        show_points[:, 2] += 0.1
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)
        visualShapedId = p.createVisualShape(shapeType=p.GEOM_SPHERE,rgbaColor=rgbaColor,radius=0.001)
        particle_id_list = []
        for i,points in enumerate(show_points):
            obj_id = p.createMultiBody(baseMass=0,
                                       baseCollisionShapeIndex=-1,
                                       baseVisualShapeIndex=visualShapedId,
                                       basePosition=points[:3])
            particle_id_list.append(obj_id)
            p.changeVisualShape(obj_id,-1,rgbaColor = color_sign[seg_sign[i]])
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)
        p.removeAllUserDebugItems()
        time.sleep(0.1)
        return particle_id_list
    def hide_particle(self,particle_id_list):
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)
        for i in particle_id_list:
            p.removeBody(i)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)
    def get_image(self,camera_pos,cam_target_pos):
        view_matrix = p.computeViewMatrix(camera_pos, cam_target_pos,Up)
        projection_matrix = p.computeProjectionMatrixFOV(fov,aspect,nearPlane,farPlane)
        img = p.getCameraImage(self.pixelWidth,self.pixelHeight,view_matrix,projection_matrix,
                               renderer=p.ER_BULLET_HARDWARE_OPENGL)
        rgb = img[2].reshape(self.pixelHeight, self.pixelWidth, 4)[:, :, :3].astype(np.float32) / 255
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        seg = ((img[4]==1)^(img[4])).reshape(self.pixelHeight,self.pixelWidth).astype(np.float32)
        seg = ((seg!=-1)&(seg!=0)).astype(np.float32)
        return rgb, seg
    def get_top_iamge(self,center):
        cam_target_pos = center.copy()
        center[2] += 0.866
        center[1] += 0.001
        center[0] += 0.001
        camera_pos = center
        view_matrix = p.computeViewMatrix(camera_pos, cam_target_pos, Up)
        projection_matrix = p.computeProjectionMatrixFOV(60, 1, nearPlane, farPlane)
        img = p.getCameraImage(256,256,view_matrix,projection_matrix,
                               renderer=p.ER_BULLET_HARDWARE_OPENGL)
        seg = ((img[4]==1)^(img[4])).reshape(256,256).astype(np.float32)
        seg = ((seg!=-1)&(seg!=0)).astype(np.float32)
        return seg

    def calc_extrinsic(self,camTargetPos, cameraPos, Up):
        C = torch.tensor(cameraPos).float()
        p = torch.tensor(camTargetPos).float()
        u = torch.tensor(Up).float()

        L = p - C
        L = F.normalize(L, dim=0, p=2)

        s = torch.cross(L, u)
        s = F.normalize(s, dim=0, p=2)
        u2 = torch.cross(s, L)
        r_mat = torch.tensor([
            [s[0], s[1], s[2]],
            [u2[0], u2[1], u2[2]],
            [-L[0], -L[1], -L[2]],
        ])

        m_r = torch.zeros((4, 4))
        m_r[:3, :3] = r_mat
        m_r[3, 3] = 1

        t = - torch.matmul(r_mat, C)
        m_t = torch.zeros((3, 4))
        m_t[:3, :3] = torch.eye(3)
        m_t[:3, 3] = t

        return m_r.numpy(), m_t.numpy()
    def calc_intrinsic(self):
        intrinsic = np.zeros((3, 3))
        intrinsic[0, 0] = -pixelWidth / (2 * tan(fov / 2 / 180 * math.pi)) / aspect
        intrinsic[1, 1] = pixelHeight / (2 * tan(fov / 2 / 180 * math.pi))
        intrinsic[0, 2] = pixelWidth / 2
        intrinsic[1, 2] = pixelHeight / 2
        intrinsic[0, 1] = 0  # skew
        intrinsic[2, 2] = 1
        return intrinsic
    def get_viode(self,x0,y0,step=4):
        r = 0.18
        base = [x0,y0,0.001]
        x,y,z = base
        m_r_list = []
        m_t_list = []
        rgb_list = []
        seg_list = []
        intrinsic =self.calc_intrinsic()
        for theta in range(0,360,step):
            x_ = x + r*cos(np.radians(theta))
            y_ = y + r*sin(np.radians(theta))
            m_r,m_t = self.calc_extrinsic([x, y, z], [x_, y_, r],[0,0,1])
            m_r_list.append(m_r)
            m_t_list.append(m_t)
            rgb,seg = self.get_image([x_, y_, r],[x,y,z])
            rgb_list.append(rgb)
            seg_list.append(seg)


        return rgb_list,seg_list,m_r_list,m_t_list,intrinsic







