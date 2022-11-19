from abc import ABCMeta, abstractmethod
import this
from scipy.spatial.transform import Rotation as R
import numpy as np
from sympy import sec, tan, cos, sin, pi

class KF(object):
    __metaclass__ = ABCMeta

    def __init__(self, n, m, dt=0.1, pval=None, qval=None, rval=None, init_x=None, z=None):

        self.dt = dt #
        self.P_pre = None #k-1 的 P阵
        self.n = n  #X维度
        self.m = m  #Z维度

        self.pval = pval
        self.qval = qval
        self.rval = rval
        
        self.z_pre = z
        if init_x is not None:
            self.x = np.array(init_x).reshape(n, 1)
        else:
            self.x = np.zeros((n, 1))

        self.P_post = self.getP0()
        self.Q = self.getQ()  
        self.R = self.getR()  

        self.I = np.eye(n)

        self.Phi = None
        self.H = self.getH(self.x)

    def step(self, z, dt=0.1):
        self.Phi = self.getPhi(self.x)

        self.x = self.Phi*self.x #状态一步预测
        self.P_pre = self.Phi * self.P_post * self.Phi.T + self.Q  #一步预测均方差方程
        K_k = self.P_pre * self.H.T * np.linalg.inv(self.H * self.P_pre * self.H.T + self.R)  #滤波增益方程
        self.x += (K_k * (np.array(z) - self.H*(self.x.T))).T # 状态估计计算方程
        self.P_post = (self.I - np.matmul(K_k, self.H)) * self.P_pre + K_k*self.R*K_k.T  #估计均方差方程
        return self.x.reshape(self.n)

    def predict(self, dt):
        xPre = self.f(self.x, dt)
        return xPre.reshape(self.n)

    #计算phi阵
    @abstractmethod
    def getPhi(self, x):
        raise NotImplementedError()
    
    #计算H阵
    @abstractmethod
    def getH(self, x):
        raise NotImplementedError()

    #计算Q阵
    @abstractmethod
    def getQ(self):
        raise NotImplementedError()

    #计算P_0阵
    @abstractmethod
    def getP0(self):
        raise NotImplementedError()
    
    #计算R阵
    @abstractmethod
    def getR(self):
        raise NotImplementedError()


class our_KF(KF):
    def __init__(self, n, m, dt=0.1, pval=None, qval=None, rval=None, init_x=None, z=None):
        KF.__init__(self, n, m, dt=0.1, pval=None, qval=None, rval=None, init_x=None, z=None)
        

    #计算phi阵
    def getPhi(self, x):
        w_ie = 7.292e-5
        
        fE, fN, fU, posL, posl, posH, velE, velN, velU = getPosVel(x=x, z=this.z)
        sinL, cosL, tanL, secL = getL(posL)
        RmCur, RnCur = getRmRn(sinL, posH)

        f12 = w_ie * sinL + velE * tanL / RnCur
        f13 = -(w_ie * cosL + velE / RnCur)
        f15 = -1 / RmCur
        f21 = -w_ie * sinL - velE * tanL / RnCur
        f23 = -velN / RmCur
        f24 = velN / RnCur
        f27 = -w_ie * sinL
        f31 = w_ie * cosL + velE / RnCur
        f32 = velN / RmCur
        f34 = tanL / RnCur
        f37 = w_ie * cosL + velE * (secL**2) / RnCur
        f42 = -fU
        f43 = fN
        f44 = (velN * tanL - velU) / RmCur
        f45 = 2 * w_ie * sinL + velE * tanL / RnCur
        f46 = -(2 * w_ie * cosL + velE / RnCur)
        f47 = 2 * w_ie * cosL * velN + velE * velN * (secL**2) / RnCur + 2 * w_ie * sinL * velU
        f51 = fU
        f53 = -fE
        f54 = -(2 * w_ie * sinL + velE * tanL / RnCur)
        f55 = -velU / RmCur
        f56 = -velN / RmCur
        f57 = -(2 * w_ie * cosL + velE * (secL**2) / RnCur) * velE
        f61 = -fN
        f62 = fE
        f64 = 2 * (w_ie * cosL + velE / RnCur)
        f65 = 2 * velN / RmCur
        f67 = -2 * velE * w_ie * sinL
        f75 = 1 / RmCur
        f84 = secL / RnCur
        f87 = velE * secL * tanL / RnCur
        f96 = 1
        F = np.matrix([
            [0.0, f12, f13, 0.0, f15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [f21, 0.0, f23, f24, 0.0, 0.0, f27, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [f31, f32, 0.0, f34, 0.0, 0.0, f37, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, f42, f43, f44, f45, f46, f47, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [f51, 0.0, f53, f54, f55, f56, f57, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [f61, f62, f64, f65, f67, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, f75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, f84, 0.0, f87, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, f96, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ])
        return np.eye(self.n) + F

    #计算H阵
    def getH(self, x):
        raise NotImplementedError()

    #计算Q阵
    def getQ(self):
        return np.diag(self.qval)

    #计算P_0阵
    def getP0(self):
        return np.diag(self.pval)
    
    #计算R阵
    def getR(self):
        return np.diag(self.rval)