# all the linear positions

import numpy as np

class R_KalmanFilter:
    def __init__(self, F, P_k, Q, R, H, x_k, dt):
        self.x_k = x_k   
        self.P_k = P_k 
        self.Q = Q
        self.F = F
        self.H = H
        self.R = R
        self.dt = dt
    
        self.x_priori = 0
        self.P_priori = 0
        self.current_time = 0

  
    def priori(self):
        self.x_priori = self.F @ self.x_k
        self.P_priori = (self.F @ self.P_k @ self.F.T) + self.Q

    def update(self, z):
        K = (self.P_priori @ self.H.T) @ np.linalg.inv(self.H @ self.P_priori @ self.H.T + self.R)
        self.x_k = self.x_priori + K @ (z - self.H @ self.x_priori)
        self.P_k = (np.eye(len(K)) - K @ self.H) @ self.P_priori
        self.current_time += self.dt

    def sef_F(self, F):
        self.F = F

    def get_state(self):
        return self.x_k