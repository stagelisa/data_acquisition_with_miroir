import numpy as np
import pandas as pd
import math

from time import sleep
from scipy.spatial.transform import Rotation as R


class IMU:
        
    def __init__(self, data):
        self.imu_data = data.to_numpy()
    
    def get_t(self):
        return self.imu_data[:, 0]         
        
    # rad/s
    def get_gyro(self):
        return [self.imu_data[:, 7], self.imu_data[:, 8], self.imu_data[:, 9]]        

    def get_mag(self):
        return [self.imu_data[:, 4], self.imu_data[:, 5], self.imu_data[:, 6]]    
        
    # m/s^2
    def get_acc(self):
        return [self.imu_data[:, 1], self.imu_data[:, 2], self.imu_data[:, 3]]


    def get_initial_orientation(self, N = 20, angle=True):
        [ax, ay, az] = self.get_acc()
        ax, ay, az = np.mean(ax[:N]), np.mean(ay[:N]), np.mean(az[:N])

        phi = np.arctan2(ay, np.sqrt(ax ** 2.0 + az ** 2.0))
        theta = np.arctan2(-ax, np.sqrt(ay ** 2.0 + az ** 2.0))

        [mag_x, mag_y, mag_z] = self.get_mag()
        mag_x, mag_y, mag_z = np.mean(mag_x[:N]), np.mean(mag_y[:N]), np.mean(mag_z[:N])

        a = -mag_y * np.cos(phi) + mag_z * np.sin(phi)
        b = mag_x * np.cos(theta)
        c = mag_y * np.sin(theta) * np.sin(phi)
        d = mag_z * np.sin(theta) * np.cos(phi)
        gamma = np.arctan2(a,  (b + c + d)) + np.pi/2

        if angle:
            return [phi, theta, gamma]
        else:
            return R.from_euler(seq='xyz', angles=[phi, theta, gamma]).as_quat()


    def get_g0(self, N = 20):
        [ax, ay, az] = self.get_acc()
        [mag_x, mag_y, mag_z] = self.get_mag()

        g = np.zeros((N, 3))
        for i in range(N):
            phi = np.arctan2(ay[i], np.sqrt(ax[i] ** 2.0 + az[i] ** 2.0))
            theta = np.arctan2(-ax[i], np.sqrt(ay[i] ** 2.0 + az[i] ** 2.0))

            a = -mag_y[i] * np.cos(phi) + mag_z[i] * np.sin(phi)
            b = mag_x[i] * np.cos(theta)
            c = mag_y[i] * np.sin(theta) * np.sin(phi)
            d = mag_z[i] * np.sin(theta) * np.cos(phi)
            gamma = np.arctan2(a,  (b + c + d)) + np.pi/2

            g[i] = R.from_euler('xyz', [phi, theta, gamma]).as_matrix() @ np.array([ax[i], ay[i], az[i]])

        return -np.mean(g, axis=0)

    
    # rad
    def get_acc_angles(self):
        [ax, ay, az] = self.get_acc()
        phi = np.arctan2(ay, np.sqrt(ax ** 2.0 + az ** 2.0))
        theta = np.arctan2(-ax, np.sqrt(ay ** 2.0 + az ** 2.0))
        return [phi, theta]

def get_mag_yaw(mag_x, mag_y, mag_z, prev_angle):
    a = -mag_y * np.cos(prev_angle[0]) + mag_z * np.sin(prev_angle[0])
    b = mag_x * np.cos(prev_angle[1])
    c = mag_y * np.sin(prev_angle[1]) * np.sin(prev_angle[0])
    d = mag_z * np.sin(prev_angle[1]) * np.cos(prev_angle[0])
    
    return (np.arctan2(a,  (b + c + d)) + np.pi/2)