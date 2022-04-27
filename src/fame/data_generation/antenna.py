import math
import numpy as np
import tensorflow as tf
from fame.utilities.pspoint import PSpoint

dilog = lambda x: tf.math.special.spence(1-x)
frac = lambda x, y : x / y

class Antenna():
    def __init__(self, mu, alpha_s=0.118, C_F=4/3, C_A=3, N_c=3, N_f=5):
        self.alpha_s = alpha_s
        self.C_F = C_F
        self.C_A = C_A
        self.N_c = N_c
        self.N_f = N_f
        self.beta = (11*N_c-2*N_f)/6
        self.mu = mu
        self.eg = np.euler_gamma
        self.Pi = np.pi
        self.S = 8*self.Pi**2
        self.quark_indices = [1, 2]
        self.gluon_indices = [3, 4, 5]


    def X_3_1(self, i, j, k, corrections="lc"):
        self.set_indices(i, j, k)
        if (i in self.quark_indices and k in self.quark_indices):
            X_3_1 = self.N_c*self.A_3_1_lc(i, j, k)
            if corrections == "slc":
                X_3_1 += -1/self.N_c*self.A_3_1_slc(i, j, k)
            elif corrections == "ql":
                X_3_1 +=  self.N_f*self.A_3_1_ql(i, j, k)
            elif corrections == "all":
                X_3_1 += -1/self.N_c*self.A_3_1_slc(i, j, k) + self.N_f*self.A_3_1_ql(i, j, k)
        elif (i in self.quark_indices and j in self.gluon_indices):
            X_3_1 = self.N_c*self.D_3_1_lc(i, j, k)/2
            if corrections == "ql":
                X_3_1 += self.N_f*self.D_3_1_ql(i, j, k)/2
            elif corrections == "all":
                X_3_1 += self.N_f*self.D_3_1_ql(i, j, k)/2
        elif (i in self.gluon_indices and j in self.gluon_indices):
            X_3_1 = self.N_c*self.F_3_1_lc(i, j, k)/2
            if corrections == "ql":
                X_3_1 += self.N_f*self.F_3_1_ql(i, j, k)/2
            elif corrections == "all":
                X_3_1 += self.N_f*self.F_3_1_ql(i, j, k)/2
        return X_3_1


    def X_3_0(self, i, j, k):
        self.set_indices(i, j, k)
        if (i in self.quark_indices and k in self.quark_indices):
            X_3_0 = self.N_c*self.A_3_0(i, j, k)
        elif (i in self.quark_indices and j in self.gluon_indices):
            X_3_0 = self.N_c*self.D_3_0(i, j, k)/2
        elif (i in self.gluon_indices and j in self.gluon_indices):
            X_3_0 = self.N_c*self.F_3_0(i, j, k)/2
        return X_3_0


    def A_3_1_slc(self, i, j, k):
        self.set_indices(i, j, k)
        return ((3*tf.math.log(self.s_jk/self.s_ijk)*self.s_jk)/(self.s_ij + self.s_ik) + (12*tf.math.log(self.s_ik/self.s_ijk)*tf.math.pow(self.s_ik,2))/tf.math.pow(self.s_ij + self.s_jk,2) + self.s_ij*((-3 - tf.math.pow(self.Pi,2) + 9*tf.math.log(self.s_ik) - 6*self.eg*tf.math.log(self.s_ik) + 12*tf.math.log(2.0)*tf.math.log(self.s_ik) + 6*tf.math.log(self.Pi)*tf.math.log(self.s_ik) - 6*tf.math.log(tf.math.pow(self.mu,-2))*tf.math.log(self.s_ik) - 3*tf.math.pow(tf.math.log(self.s_ik),2) - 9*tf.math.log(self.s_ik/self.s_ijk) + 3*tf.math.log(tf.math.pow(-1 + self.s_ik/self.s_ijk,2))*tf.math.log(self.s_ik/self.s_ijk) + 3*tf.math.log(tf.math.pow(-1 + self.s_jk/self.s_ijk,2))*tf.math.log(self.s_jk/self.s_ijk) - 6*tf.math.log(self.s_ik/self.s_ijk)*tf.math.log(self.s_jk/self.s_ijk) - 9*tf.math.log(self.s_ijk) + 6*self.eg*tf.math.log(self.s_ijk) - 12*tf.math.log(2.0)*tf.math.log(self.s_ijk) - 6*tf.math.log(self.Pi)*tf.math.log(self.s_ijk) + 6*tf.math.log(tf.math.pow(self.mu,-2))*tf.math.log(self.s_ijk) + 3*tf.math.pow(tf.math.log(self.s_ijk),2) + 6*dilog((self.s_ik/self.s_ijk)) + 6*dilog((self.s_jk/self.s_ijk)))/self.s_jk + (3*tf.math.log(self.s_ij/self.s_ijk)*(2*self.s_ik + self.s_jk))/tf.math.pow(self.s_ik + self.s_jk,2)) + self.s_ik*(-((3 - 2*tf.math.pow(self.Pi,2) + 6*tf.math.log(tf.math.pow(-1 + self.s_ij/self.s_ijk,2))*tf.math.log(self.s_ij/self.s_ijk) + 6*tf.math.log(tf.math.pow(-1 + self.s_ik/self.s_ijk,2))*tf.math.log(self.s_ik/self.s_ijk) - 12*tf.math.log(self.s_ij/self.s_ijk)*tf.math.log(self.s_ik/self.s_ijk) + 12*dilog((self.s_ij/self.s_ijk)) + 12*dilog((self.s_ik/self.s_ijk)))/self.s_jk) + (3*tf.math.log(self.s_jk/self.s_ijk)*self.s_jk)/tf.math.pow(self.s_ij + self.s_ik,2) + 3*(1/(self.s_ij + self.s_ik) + (4*tf.math.log(self.s_jk/self.s_ijk))/(self.s_ij + self.s_ik) + 4/(self.s_ij + self.s_jk) + (8*tf.math.log(self.s_ik/self.s_ijk))/(self.s_ij + self.s_jk) + 1/(self.s_ik + self.s_jk) + (4*tf.math.log(self.s_ij/self.s_ijk))/(self.s_ik + self.s_jk))) + (-((3 + tf.math.pow(self.Pi,2) - 9*tf.math.log(self.s_ik) + 6*self.eg*tf.math.log(self.s_ik) - 12*tf.math.log(2.0)*tf.math.log(self.s_ik) - 6*tf.math.log(self.Pi)*tf.math.log(self.s_ik) + 6*tf.math.log(tf.math.pow(self.mu,-2))*tf.math.log(self.s_ik) + 3*tf.math.pow(tf.math.log(self.s_ik),2) - 3*tf.math.log(tf.math.pow(-1 + self.s_ij/self.s_ijk,2))*tf.math.log(self.s_ij/self.s_ijk) + 9*tf.math.log(self.s_ik/self.s_ijk) - 3*tf.math.log(tf.math.pow(-1 + self.s_ik/self.s_ijk,2))*tf.math.log(self.s_ik/self.s_ijk) + 6*tf.math.log(self.s_ij/self.s_ijk)*tf.math.log(self.s_ik/self.s_ijk) + 9*tf.math.log(self.s_ijk) - 6*self.eg*tf.math.log(self.s_ijk) + 12*tf.math.log(2.0)*tf.math.log(self.s_ijk) + 6*tf.math.log(self.Pi)*tf.math.log(self.s_ijk) - 6*tf.math.log(tf.math.pow(self.mu,-2))*tf.math.log(self.s_ijk) - 3*tf.math.pow(tf.math.log(self.s_ijk),2) - 6*dilog((self.s_ij/self.s_ijk)) - 6*dilog((self.s_ik/self.s_ijk)))*tf.math.pow(self.s_jk,2)) + self.s_ik*((-3 + 2*tf.math.pow(self.Pi,2) - 6*tf.math.log(tf.math.pow(-1 + self.s_ik/self.s_ijk,2))*tf.math.log(self.s_ik/self.s_ijk) - 6*tf.math.log(tf.math.pow(-1 + self.s_jk/self.s_ijk,2))*tf.math.log(self.s_jk/self.s_ijk) + 12*tf.math.log(self.s_ik/self.s_ijk)*tf.math.log(self.s_jk/self.s_ijk) - 12*dilog((self.s_ik/self.s_ijk)) - 12*dilog((self.s_jk/self.s_ijk)))*self.s_jk - 2*(2*tf.math.pow(self.Pi,2) - 9*tf.math.log(self.s_ik) + 6*self.eg*tf.math.log(self.s_ik) - 12*tf.math.log(2.0)*tf.math.log(self.s_ik) - 6*tf.math.log(self.Pi)*tf.math.log(self.s_ik) + 6*tf.math.log(tf.math.pow(self.mu,-2))*tf.math.log(self.s_ik) + 3*tf.math.pow(tf.math.log(self.s_ik),2) - 3*tf.math.log(tf.math.pow(-1 + self.s_ij/self.s_ijk,2))*tf.math.log(self.s_ij/self.s_ijk) + 9*tf.math.log(self.s_ik/self.s_ijk) - 6*tf.math.log(tf.math.pow(-1 + self.s_ik/self.s_ijk,2))*tf.math.log(self.s_ik/self.s_ijk) + 6*tf.math.log(self.s_ij/self.s_ijk)*tf.math.log(self.s_ik/self.s_ijk) - 3*tf.math.log(tf.math.pow(-1 + self.s_jk/self.s_ijk,2))*tf.math.log(self.s_jk/self.s_ijk) + 6*tf.math.log(self.s_ik/self.s_ijk)*tf.math.log(self.s_jk/self.s_ijk) + 9*tf.math.log(self.s_ijk) - 6*self.eg*tf.math.log(self.s_ijk) + 12*tf.math.log(2.0)*tf.math.log(self.s_ijk) + 6*tf.math.log(self.Pi)*tf.math.log(self.s_ijk) - 6*tf.math.log(tf.math.pow(self.mu,-2))*tf.math.log(self.s_ijk) - 3*tf.math.pow(tf.math.log(self.s_ijk),2) - 6*dilog((self.s_ij/self.s_ijk)) - 12*dilog((self.s_ik/self.s_ijk)) - 6*dilog((self.s_jk/self.s_ijk)))*self.s_ijk))/(self.s_ij*self.s_jk))/(48.*tf.math.pow(self.Pi,2)*self.s_ijk)

    
    def A_3_1_ql(self, i, j, k):
        self.set_indices(i, j, k)
        return -0.041666666666666664*((self.eg + tf.math.log(self.s_ijk/(4.*self.Pi*tf.math.pow(self.mu,2))))*(tf.math.pow(self.s_ij,2) + tf.math.pow(self.s_jk,2) + 2*self.s_ik*self.s_ijk))/(tf.math.pow(self.Pi,2)*self.s_ij*self.s_jk*self.s_ijk)

    
    def A_3_1_lc(self, i, j, k):
        self.set_indices(i, j, k)

        R_ij_kj = self.R(self.y_ij, self.y_jk)
        a = R_ij_kj + 5/3*(tf.math.log(self.y_ij)+tf.math.log(self.y_jk))
        b = frac(1, self.s_ijk) + frac(self.s_ik + self.s_jk, 2*self.s_ijk*self.s_ij) + frac(self.s_ik + self.s_ij, 2*self.s_ijk*self.s_jk) - frac(self.s_ij, 2*self.s_ijk*(self.s_ik + self.s_ij)) - frac(self.s_jk, 2*self.s_ijk*(self.s_ik + self.s_jk))
        c = tf.math.log(self.y_ij)*(2-0.5*frac(self.s_ij*self.s_jk, (self.s_ik + self.s_jk)**2)+2*frac(self.s_ij-self.s_jk, self.s_ik+self.s_jk))/self.s_ijk
        d = tf.math.log(self.y_jk)*(2-0.5*frac(self.s_ij*self.s_jk, (self.s_ik+self.s_ij)**2)+2*frac(self.s_jk-self.s_ij, self.s_ik+self.s_ij))/self.s_ijk
        finite_from_pole = -(-22*self.eg + 6*tf.math.pow(self.eg,2) - 7*tf.math.pow(self.Pi,2) + 24*tf.math.pow(tf.math.log(2.0),2) + 12*tf.math.log(4.0)*tf.math.log(self.Pi) + 6*tf.math.pow(tf.math.log(self.Pi),2) + 22*tf.math.log(4*self.Pi) - 12*self.eg*tf.math.log(4*self.Pi) + 6*tf.math.pow(tf.math.log(tf.math.pow(self.mu,2)),2) + 4*(-5 + 3*self.eg - 3*tf.math.log(4*self.Pi))*tf.math.log(self.s_ij) + 6*tf.math.pow(tf.math.log(self.s_ij),2) - 20*tf.math.log(self.s_jk) + 12*self.eg*tf.math.log(self.s_jk) - 12*tf.math.log(4*self.Pi)*tf.math.log(self.s_jk) + 6*tf.math.pow(tf.math.log(self.s_jk),2) - 2*tf.math.log(tf.math.pow(self.mu,2))*(-11 + 6*self.eg - 6*tf.math.log(4*self.Pi) + 6*tf.math.log((self.s_ij*self.s_jk)/self.s_ijk)) + 18*tf.math.log(self.s_ijk) - 12*self.eg*tf.math.log(self.s_ijk) + 12*tf.math.log(4*self.Pi)*tf.math.log(self.s_ijk) - 6*tf.math.pow(tf.math.log(self.s_ijk),2))/(96.*tf.math.pow(self.Pi,2))
    
        A_3_0 = self.A_3_0(i, j, k)
        A_3_1 = (-a*A_3_0+b+c+d)/self.S + self.T*A_3_0/self.S + finite_from_pole*A_3_0
        return A_3_1


    def A_3_0(self, i, j, k):
        self.set_indices(i, j, k)

        a = frac(self.s_ij, self.s_jk)
        b = frac(self.s_ik*self.s_ijk, self.s_ij*self.s_jk)

        A_3_0 = (a + 1/a + 2*b)/self.s_ijk
        return tf.convert_to_tensor([A_3_0])


    def D_3_1_ql(self, i, j, k):
        self.set_indices(i, j, k)
        return (tf.math.pow(self.s_ij,2)*((-4*self.eg + tf.math.log(256.0) + 4*tf.math.log(self.Pi) + 8*tf.math.log(self.mu) - 4*tf.math.log(self.s_ijk))*tf.math.pow(self.s_ik,2) + (-2*self.eg + tf.math.log(16.0) + 2*tf.math.log(self.Pi) + 4*tf.math.log(self.mu) - 2*tf.math.log(self.s_ijk))*tf.math.pow(self.s_jk,2) + (-4*self.eg + tf.math.log(256.0) + 4*tf.math.log(self.Pi) + 8*tf.math.log(self.mu) - 4*tf.math.log(self.s_ijk))*tf.math.pow(self.s_ijk,2)) + self.s_ij*((-2*self.eg + tf.math.log(16.0) + 2*tf.math.log(self.Pi) + 4*tf.math.log(self.mu) - 2*tf.math.log(self.s_ijk))*tf.math.pow(self.s_jk,3) + self.s_ik*((-2*self.eg + tf.math.log(16.0) + 2*tf.math.log(self.Pi) + 4*tf.math.log(self.mu) - 2*tf.math.log(self.s_ijk))*tf.math.pow(self.s_jk,2) + (-10*self.eg + tf.math.log(1048576) + 10*tf.math.log(self.Pi) + 20*tf.math.log(self.mu) - 10*tf.math.log(self.s_ijk))*self.s_jk*self.s_ijk - 16*tf.math.pow(self.Pi,2)*tf.math.pow(self.s_ijk,2))) + self.s_ik*((-2*self.eg + tf.math.log(16.0) + 2*tf.math.log(self.Pi) + 4*tf.math.log(self.mu) - 2*tf.math.log(self.s_ijk))*tf.math.pow(self.s_jk,3) + self.s_ik*((-2*self.eg + tf.math.log(16.0) + 2*tf.math.log(self.Pi) + 4*tf.math.log(self.mu) - 2*tf.math.log(self.s_ijk))*tf.math.pow(self.s_jk,2) + (-4*self.eg + tf.math.log(256.0) + 4*tf.math.log(self.Pi) + 8*tf.math.log(self.mu) - 4*tf.math.log(self.s_ijk))*tf.math.pow(self.s_ijk,2))))/(48.*tf.math.pow(self.Pi,2)*self.s_ij*self.s_ik*self.s_jk*tf.math.pow(self.s_ijk,2))


    def D_3_1_lc(self, i, j, k):
        self.set_indices(i, j, k)

        R_ij_jk = self.R(self.y_ij, self.y_jk)
        R_ik_jk = self.R(self.y_ik, self.y_jk)
        R_ij_ik = self.R(self.y_ij, self.y_ik)

        a = R_ij_jk + R_ik_jk + R_ij_ik
        b = 5/3*(tf.math.log(self.y_ij) + tf.math.log(self.y_ik)) + 11/6*tf.math.log(self.y_jk)
        c = 1/(3*self.s_jk)
        finite_from_pole = -((-22*self.eg + 6*tf.math.pow(self.eg,2) - 7*tf.math.pow(self.Pi,2) + 24*tf.math.pow(tf.math.log(2.0),2) + 12*tf.math.log(4.0)*tf.math.log(self.Pi) + 6*tf.math.pow(tf.math.log(self.Pi),2) + 22*tf.math.log(4*self.Pi) - 12*self.eg*tf.math.log(4*self.Pi) + 6*tf.math.pow(tf.math.log(tf.math.pow(self.mu,2)),2) + 4*(-5 + 3*self.eg - 3*tf.math.log(4*self.Pi))*tf.math.log(self.s_ij) + 6*tf.math.pow(tf.math.log(self.s_ij),2) - 20*tf.math.log(self.s_ik) + 12*self.eg*tf.math.log(self.s_ik) - 12*tf.math.log(4*self.Pi)*tf.math.log(self.s_ik) + 6*tf.math.pow(tf.math.log(self.s_ik),2) - 22*tf.math.log(self.s_jk) + 12*self.eg*tf.math.log(self.s_jk) - 12*tf.math.log(4*self.Pi)*tf.math.log(self.s_jk) + 6*tf.math.pow(tf.math.log(self.s_jk),2) - 2*tf.math.log(tf.math.pow(self.mu,2))*(-11 + 6*self.eg - 6*tf.math.log(4*self.Pi) + 6*tf.math.log(self.s_ij) + 6*tf.math.log(self.s_ik) + 6*tf.math.log(self.s_jk) - 12*tf.math.log(self.s_ijk)) + 40*tf.math.log(self.s_ijk) - 24*self.eg*tf.math.log(self.s_ijk) + 24*tf.math.log(4*self.Pi)*tf.math.log(self.s_ijk) - 12*tf.math.pow(tf.math.log(self.s_ijk),2))/(96.*tf.math.pow(self.Pi,2)))

        D_3_0 = self.D_3_0(i, j, k)
        D_3_1 = (-(a+b)*D_3_0 + c)/self.S + self.T*D_3_0/self.S + finite_from_pole*D_3_0
        return D_3_1


    def D_3_0(self, i, j, k):
        self.set_indices(i, j, k)

        d_ij = 2*frac(self.s_ijk**2*self.s_ij, self.s_ik*self.s_jk) + frac(self.s_ik*self.s_jk + self.s_jk**2, self.s_ij)
        d_ik = 2*frac(self.s_ijk**2*self.s_ik, self.s_ij*self.s_jk) + frac(self.s_ij*self.s_jk + self.s_jk**2, self.s_ik)
        d_ijk = 2*frac(self.s_ij*self.s_ik, self.s_jk) + 5*self.s_ijk + self.s_jk

        D_3_0 = (d_ij + d_ik + d_ijk)/self.s_ijk**2
        return tf.convert_to_tensor([D_3_0])


    def F_3_1_ql(self, i, j, k):
        self.set_indices(i, j, k)
        return -0.041666666666666664*(self.s_ik*self.s_jk*tf.math.pow(self.s_ijk,2) - (-2*self.eg + 2*tf.math.log(4*self.Pi) + 4*tf.math.log(self.mu) - 2*tf.math.log(self.s_ijk))*tf.math.pow(self.s_jk,2)*tf.math.pow(self.s_ijk,2) + (2*self.eg - 2*tf.math.log(4*self.Pi) - 4*tf.math.log(self.mu) + 2*tf.math.log(self.s_ijk))*tf.math.pow(self.s_ik,2)*(tf.math.pow(self.s_jk,2) + tf.math.pow(self.s_ijk,2)) + (2*self.eg - 2*tf.math.log(4*self.Pi) - 4*tf.math.log(self.mu) + 2*tf.math.log(self.s_ijk))*tf.math.pow(self.s_ij,2)*(tf.math.pow(self.s_ik,2) + tf.math.pow(self.s_jk,2) + tf.math.pow(self.s_ijk,2)) + self.s_ij*self.s_ijk*(self.s_jk*self.s_ijk + self.s_ik*((1 + 8*self.eg - 8*tf.math.log(4*self.Pi) - 16*tf.math.log(self.mu) + 8*tf.math.log(self.s_ijk))*self.s_jk + self.s_ijk)))/(tf.math.pow(self.Pi,2)*self.s_ij*self.s_ik*self.s_jk*tf.math.pow(self.s_ijk,2))


    def F_3_1_lc(self, i, j, k):
        self.set_indices(i, j, k)

        R_ij_ik = self.R(self.y_ij, self.y_ik)
        R_ik_jk = self.R(self.y_ik, self.y_jk)
        R_ij_jk = self.R(self.y_ij, self.y_jk)
        R = R_ij_ik + R_ik_jk + R_ij_jk

        Y = 11/6 * (tf.math.log(self.y_ij) + tf.math.log(self.y_ik) + tf.math.log(self.y_jk))
        S = frac(1, self.s_ij) + frac(1, self.s_ik) + frac(1, self.s_jk) + frac(1, self.s_ijk)
        finite_from_pole = -0.010416666666666666*(-22*self.eg + 6*tf.math.pow(self.eg,2) - 7*tf.math.pow(self.Pi,2) + 24*tf.math.pow(tf.math.log(2.0),2) + 12*tf.math.log(4.0)*tf.math.log(self.Pi) + 6*tf.math.pow(tf.math.log(self.Pi),2) + 22*tf.math.log(4*self.Pi) - 12*self.eg*tf.math.log(4*self.Pi) + 6*tf.math.pow(tf.math.log(tf.math.pow(self.mu,2)),2) + 2*(-11 + 6*self.eg - 6*tf.math.log(4*self.Pi))*tf.math.log(self.s_ij) + 6*tf.math.pow(tf.math.log(self.s_ij),2) - 22*tf.math.log(self.s_ik) + 12*self.eg*tf.math.log(self.s_ik) - 12*tf.math.log(4*self.Pi)*tf.math.log(self.s_ik) + 6*tf.math.pow(tf.math.log(self.s_ik),2) - 22*tf.math.log(self.s_jk) + 12*self.eg*tf.math.log(self.s_jk) - 12*tf.math.log(4*self.Pi)*tf.math.log(self.s_jk) + 6*tf.math.pow(tf.math.log(self.s_jk),2) - 2*tf.math.log(tf.math.pow(self.mu,2))*(-11 + 6*self.eg - 6*tf.math.log(4*self.Pi) + 6*tf.math.log(self.s_ij) + 6*tf.math.log(self.s_ik) + 6*tf.math.log(self.s_jk) - 12*tf.math.log(self.s_ijk)) + 44*tf.math.log(self.s_ijk) - 24*self.eg*tf.math.log(self.s_ijk) + 24*tf.math.log(4*self.Pi)*tf.math.log(self.s_ijk) - 12*tf.math.pow(tf.math.log(self.s_ijk),2))/tf.math.pow(self.Pi,2)
        
        F_3_0 = self.F_3_0(i, j, k)
        F_3_1 = -(R+Y)*F_3_0/self.S + S/3/self.S + self.T*F_3_0/self.S + finite_from_pole*F_3_0
        return F_3_1


    def F_3_0(self, i, j, k):
        self.set_indices(i, j, k)

        f_ij = frac(self.s_ijk**2*self.s_ij, self.s_ik*self.s_jk) + frac(self.s_ik*self.s_jk, self.s_ij)
        f_ik = frac(self.s_ijk**2*self.s_ik, self.s_ij*self.s_jk) + frac(self.s_ij*self.s_jk, self.s_ik)
        f_jk = frac(self.s_ijk**2*self.s_jk, self.s_ij*self.s_ik) + frac(self.s_ij*self.s_ik, self.s_jk)

        F_3_0 = 2/self.s_ijk**2*(f_ij+f_ik+f_jk+4*self.s_ijk)
        return tf.convert_to_tensor([F_3_0])


    def R(self, y, z):
        a = tf.math.log(y)*tf.math.log(z)
        b = tf.math.log(y)*tf.math.log(1-y)
        c = tf.math.log(z)*tf.math.log(1-z)
        d = self.Pi**2/6
        e = dilog(y) + dilog(z)
        return a-b-c+d-e


    def kosower_mapping(self, i, j, k):
        self.set_indices(i, j, k)

        r = self.s_jk / (self.s_ij + self.s_jk)
        rho = math.sqrt(1 + 4*r*(1-r)*self.s_ij*self.s_jk/(self.s_ijk*self.s_ik))
        
        q_ij = ((1+rho)*self.s_ijk-2*self.s_jk*r)*self.pi/(2*(self.s_ijk-self.s_jk)) + ((1-rho)*self.s_ijk-2*self.s_ij*r)*self.pk/(2*(self.s_ijk-self.s_ij)) + r*self.pj
        q_jk = ((1-rho)*self.s_ijk-2*self.s_jk*(1-r))*self.pi/(2*(self.s_ijk-self.s_jk)) + ((1+rho)*self.s_ijk-2*self.s_ij*(1-r))*self.pk/(2*(self.s_ijk-self.s_ij)) + (1-r)*self.pj

        # delete p_j
        q = np.delete(self.p, j+1, axis=0)
        # insert p_ij
        if i >= j+1:
            q[i] = q_ij
        else:
            q[i+1] = q_ij
        # insert p_jk
        if k > j:
            q[k] = q_jk
        else:
            q[k+1] = q_jk

        return q


    def set_indices(self, i, j, k):
        """
        Set indices of antenna legs.
        """
        self.pi = self.p[i+1]
        self.pj = self.p[j+1]
        self.pk = self.p[k+1]
        ij = self._get_pair_string(i, j)
        ik = self._get_pair_string(i, k)
        jk = self._get_pair_string(j, k)
        self.s_ij = self.s[ij]
        self.s_ik = self.s[ik]
        self.s_jk = self.s[jk]
        self.s_ijk = self.s_ij + self.s_ik + self.s_jk
        self.y_ij = self._y(self.s_ij)
        self.y_ik = self._y(self.s_ik)
        self.y_jk = self._y(self.s_jk)
        self.T = self.beta*(tf.math.log(self.mu**2/self.s_ijk))/(8*self.Pi**2)


    def set_momenta(self, p):
        """
        Set momenta of choice.
        """
        self.p = p
        self.num_jets = self.p.shape[0]-2
        self.w = p[0][0] + p[1][0]
        # quicker to calculate all s_ij here as we can re-use them for all permutations
        self.s = PSpoint(p, self.num_jets, self.w, 0.0).sij


    def _y(self, s):
        return s/self.s_ijk


    def _get_pair_string(self, i, j):
        return "".join(map(str, sorted([i, j])))
