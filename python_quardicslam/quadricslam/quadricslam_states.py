'''定义了表示系统状态的几个类'''

from spatialmath import SE3
from typing import Dict, List, Optional, Union
import gtsam
import numpy as np


# 创建一个带有名称 'q' 或 'x' 的符号（symbol）对象，并将其转换为整数。
# q是对偶二次曲线，x是位姿。
def qi(i: int) -> int:
    return int(gtsam.symbol('q', i))
def xi(i: int) -> int:
    return int(gtsam.symbol('x', i))


# 这个类表示从图像中检测到的一个物体，它包含了：
# 该物体的标签、边界框（bounds）、位姿关键字（pose_key）和对偶二次曲面关键字（quadric_key）。
class Detection:
    def __init__(self,
                 label: str,
                 bounds: np.ndarray,
                 pose_key: int,
                 quadric_key: Optional[int] = None) -> None:
        self.label = label
        self.bounds = bounds
        self.pose_key = pose_key
        self.quadric_key = quadric_key


# 这个类表示"一步"，它包含了一个整数i和一个位姿关键字pose_key，
# 还有一些可选的数据成员，如RGB、深度图像、里程计信息、检测到的物体等。
class StepState:
    def __init__(self, i: int) -> None:
        self.i = i
        self.pose_key = xi(i)

        self.rgb: Optional[np.ndarray] = None
        self.depth: Optional[np.ndarray] = None
        self.odom: Optional[SE3] = None

        self.detections: List[Detection] = []
        self.new_associated: List[Detection] = [] # 表示在这一步中新关联上的物体。


# 这个类表示"系统状态"，它包含了初始位姿、各种噪声模型、优化器类型和参数等。
# 包含了一个字典，将位姿关键字映射到标签。
# 包含了一个非线性因子图（NonlinearFactorGraph）和一个变量值集合（Values）对象，用于表示系统状态。
class SystemState:
    def __init__(
        self,
        initial_pose: SE3, 
        noise_prior: np.ndarray,
        noise_odom: np.ndarray,
        noise_boxes: np.ndarray,
        optimiser_batch: bool,
        optimiser_params: Union[gtsam.ISAM2Params,
                                gtsam.LevenbergMarquardtParams,
                                gtsam.GaussNewtonParams],
    ) -> None:
        self.initial_pose = gtsam.Pose3(initial_pose.A) # Pose3 类型的初始位姿，表示 SLAM 系统的初始姿态
        self.noise_prior = gtsam.noiseModel.Diagonal.Sigmas(noise_prior) # 表示系统先验噪声模型
        self.noise_odom = gtsam.noiseModel.Diagonal.Sigmas(noise_odom) # 表示里程计测量的噪声模型
        self.noise_boxes = gtsam.noiseModel.Diagonal.Sigmas(noise_boxes)# 表示物体检测边界框测量的噪声模型

        self.optimiser_batch = optimiser_batch # 表示优化器是否使用批处理方法；
        self.optimiser_params = optimiser_params
        self.optimiser_type = (
            gtsam.ISAM2 if type(optimiser_params) == gtsam.ISAM2Params else
            gtsam.GaussNewtonOptimizer if type(optimiser_params)
            == gtsam.GaussNewtonParams else gtsam.LevenbergMarquardtOptimizer)
        # 优化器参数对象，可以是 ISAM2Params、LevenbergMarquardtParams 或 GaussNewtonParams；

        self.associated: List[Detection] = [] # Detection 类型的列表，表示已经被关联的物体
        self.unassociated: List[Detection] = [] # Detection 类型的列表，表示还没有被关联的物体

        self.labels: Dict[int, str] = {} # 物体标签

        self.graph = gtsam.NonlinearFactorGraph() # 表示非线性因子图
        self.estimates = gtsam.Values() # 表示优化器的估计值

        self.optimiser = None

        self.calib_depth: Optional[float] = None # 表示深度摄像头的校准参数
        self.calib_rgb: Optional[np.ndarray] = None # 表示 RGB 摄像头的校准参数


class QuadricSlamState:

    def __init__(self, system: SystemState) -> None:
        self.system = system # 系统状态， SystemState 

        self.prev_step: Optional[StepState] = None # 上一步状态
        self.this_step: Optional[StepState] = None # 这一步状态
