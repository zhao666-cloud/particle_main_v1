import time
from gym import spaces,core
import numpy as np
import pybullet as p
import pybullet_data
from .robot import UR5Robotiq85
pixelWidth = 256
pixelHeight = 192
class grasp(core.Env,UR5Robotiq85):

    SIMULATION_STEP_DELAY = 1/ 240.
#TODO GUI
    def __init__(self,GUI=True,object=True):
        super(UR5Robotiq85,self).__init__([0,0,0],[0,0,0])
        #self.robot = robot
        self.GUI = GUI
        self.low_state = np.float32(np.ones((pixelWidth, pixelHeight, 3)) * 0)
        self.high_state = np.float32(np.ones((pixelWidth, pixelHeight, 3)) * 255)
        self.action_space = spaces.Discrete(6) #no use
        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state, dtype=np.uint8)
        self.physicsClient = p.connect(p.GUI if self.GUI else p.DIRECT)
        #self.robot.step_simulation = self.step_simulation
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        p.resetDebugVisualizerCamera(1.5, 150, -30, (0.1, 0.1, 0))
        self.planeID = p.loadURDF("plane.urdf")
        self.load()
        if object:
            self.load_obj()


    def step_simulation(self):
        p.stepSimulation()
        if self.GUI:
            time.sleep(self.SIMULATION_STEP_DELAY)
    def reset(self):
        self.reset_robot()
    def step(self,action:np.ndarray):
        prepare_action = np.copy(action)
        prepare_action[2] += 0.15
        self.open_gripper()
        self.move_ee(prepare_action)
        time.sleep(0.5)
        self.move_ee(action)
        self.close_gripper()
        self.move_ee(action)
        time.sleep(0.5)
        self.move_ee(prepare_action)
        #TODO gym reward




