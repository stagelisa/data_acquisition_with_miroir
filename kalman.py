import os
import numpy as np
import math
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as R
from time import sleep, time

from utils import *
from utils.imu import *
from utils.for_kalman import *


class Kalman_filter:
    def __init__(self, data, mov_avg=False):
        self.imu = IMU(data)
        self.mov_avg = mov_avg

    def cal_raw_data(self):
        [self.acc_x, self.acc_y, self.acc_z] = self.imu.get_acc()
        if self.mov_avg:
            self.acc_x = movingaverage(self.acc_x, 10)
            self.acc_y = movingaverage(self.acc_y, 10)
            self.acc_z = movingaverage(self.acc_z, 10)

        # Get gyro measurements
        [self.gyr_x, self.gyr_y, self.gyr_z] = self.imu.get_gyro()
        if self.mov_avg:
            self.gyr_x = movingaverage(self.gyr_x, 10)
            self.gyr_y = movingaverage(self.gyr_y, 10)
            self.gyr_z = movingaverage(self.gyr_z, 10)

        # Get mag measurements
        [self.mag_x, self.mag_y, self.mag_z] = self.imu.get_mag()
        if self.mov_avg:
            self.mag_x = movingaverage(self.mag_x, 10)
            self.mag_y = movingaverage(self.mag_y, 10)
            self.mag_z = movingaverage(self.mag_z, 10)


    def sys_init(self):
        self.t = self.imu.get_t()
        self.t = self.t - self.t[0]

        self.p_0 = np.zeros((3,))
        self.v_0 = np.zeros((3,))
        self.q_0 = self.imu.get_initial_orientation(angle=False)
        self.a_b_0 = np.zeros((3,))
        self.omega_b_0 = np.zeros((3,))
        self.g_0 = self.imu.get_g0()
        print(f'g value detected: {self.g_0}')

        self.V_i_0 = np.identity(3) * 0.5**2
        self.Theta_i_0 = np.identity(3) * (0.5 * np.pi / 180)**2
        self.A_i_0 = np.identity(3) * 0.01**2
        self.Omega_i_0 = np.identity(3) * (0.1*np.pi/180)**2
        self.V = np.identity(3) * 1e-4
        self.V_momentary = np.identity(1)*1e-4

        self.x_hats = np.zeros((self.imu.imu_data.shape[0] - 1, 19))
        self.delta_x_hats  = np.zeros((self.imu.imu_data.shape[0] - 1, 18))

        self.x = np.concatenate([self.p_0, self.v_0, self.q_0, self.a_b_0, self.omega_b_0, self.g_0])
        self.delta_x = np.zeros((18,))
        self.P_theta_x = np.identity(18)
        self.P_theta_x[0:3,0:3] = 0.5**2*np.identity(3)
        self.P_theta_x[3:6,3:6] = 0.5**2*np.identity(3)
        self.P_theta_x[6:9,6:9] = (0.1*np.pi/180)**2*np.identity(3)
        self.P_theta_x[9:12,9:12] = 1e-4*np.identity(3)
        self.P_theta_x[12:15,12:15] = 1e-4*np.identity(3)
        self.P_theta_x[15:,15:] = 1e-4*np.identity(3)

        self.shoe_detector = SHOE(self.imu.imu_data[:, [1,2,3,7,8,9]], self.g_0)

    
    def run(self):
        self.cal_raw_data()
        self.sys_init()
        for i in range(self.imu.imu_data.shape[0] - 1):
            # Sampling time
            dt = self.t[i+1] - self.t[i]
            input_data = np.array([self.acc_x[i+1], self.acc_y[i+1], self.acc_z[i+1], self.gyr_x[i+1], self.gyr_y[i+1], self.gyr_z[i+1]])
            
            # Prediction
            self.x = nominal_state_predict(self.x, input_data, dt)
            self.delta_x, self.P_theta_x = error_state_predict(self.delta_x, self.P_theta_x, self.x, input_data, dt, self.V_i_0, self.Theta_i_0, self.A_i_0, self.Omega_i_0)

            if self.shoe_detector[i+1]:
                self.delta_x, self.P_theta_x = zero_velocity_update(self.x, self.P_theta_x, self.V, np.array([0, 0, 0]))
                self.x = injection_obs_err_to_nominal_state(self.x, self.delta_x)
                self.delta_x, self.P_theta_x = ESKF_reset(self.delta_x, self.P_theta_x)

            self.x_hats[i, :] = self.x
            
            self.delta_x_hats[i, :] = self.delta_x
        
        ori = np.zeros((self.x_hats.shape[0] + 1, 4))
        ori[:-1] = self.x_hats[:, [9, 6, 7, 8]]
        ori[-1] = ori[-2]
        return ori