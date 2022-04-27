import tensorflow as tf
from fame.data_generation.antenna import Antenna

dilog = lambda x: tf.math.special.spence(1-x)
frac = lambda x, y : x / y

class SubAntenna(Antenna):
    def X_3_1(self, i, j, k):
        self.set_indices(i, j, k)
        if (i in self.quark_indices and k in self.quark_indices):
            X_3_1 = self.N_c*self.A_3_1(i, j, k)
        elif (i in self.quark_indices and j in self.gluon_indices):
            X_3_1 = self.N_c*self.D_3_1(i, j, k)/2
        elif (i in self.gluon_indices and j in self.gluon_indices):
            X_3_1 = self.N_c*self.F_3_1(i, j, k)/2
        return X_3_1

    def X_3_0(self, i, j, k):
        self.set_indices(i, j, k)
        if (i in self.quark_indices and k in self.quark_indices):
            X_3_0 = self.N_c*self.all_a_3_0(i, j, k)
        elif (i in self.quark_indices and j in self.gluon_indices):
            X_3_0 = self.N_c*self.all_d_3_0(i, j, k)/2
        elif (i in self.gluon_indices and j in self.gluon_indices):
            X_3_0 = self.N_c*self.all_f_3_0(i, j, k)/2
        return X_3_0

    def A_3_1(self, i, j, k):
        self.set_indices(i, j, k)

        R_ij_kj = self.R(self.y_ij, self.y_jk)
        a = R_ij_kj + 5/3*(tf.math.log(self.y_ij)+tf.math.log(self.y_jk))
        b = frac(1, self.s_ijk) + frac(self.s_ik + self.s_jk, 2*self.s_ijk*self.s_ij) + frac(self.s_ik + self.s_ij, 2*self.s_ijk*self.s_jk) - frac(self.s_ij, 2*self.s_ijk*(self.s_ik + self.s_ij)) - frac(self.s_jk, 2*self.s_ijk*(self.s_ik + self.s_jk))
        c = tf.math.log(self.y_ij)*(2-0.5*frac(self.s_ij*self.s_jk, (self.s_ik + self.s_jk)**2)+2*frac(self.s_ij-self.s_jk, self.s_ik+self.s_jk))/self.s_ijk
        d = tf.math.log(self.y_jk)*(2-0.5*frac(self.s_ij*self.s_jk, (self.s_ik+self.s_ij)**2)+2*frac(self.s_jk-self.s_ij, self.s_ik+self.s_ij))/self.s_ijk
        finite_from_pole = -(-22*self.eg + 6*tf.math.pow(self.eg,2) - 7*tf.math.pow(self.Pi,2) + 24*tf.math.pow(tf.math.log(2.0),2) + 12*tf.math.log(4.0)*tf.math.log(self.Pi) + 6*tf.math.pow(tf.math.log(self.Pi),2) + 22*tf.math.log(4*self.Pi) - 12*self.eg*tf.math.log(4*self.Pi) + 6*tf.math.pow(tf.math.log(tf.math.pow(self.mu,2)),2) + 4*(-5 + 3*self.eg - 3*tf.math.log(4*self.Pi))*tf.math.log(self.s_ij) + 6*tf.math.pow(tf.math.log(self.s_ij),2) - 20*tf.math.log(self.s_jk) + 12*self.eg*tf.math.log(self.s_jk) - 12*tf.math.log(4*self.Pi)*tf.math.log(self.s_jk) + 6*tf.math.pow(tf.math.log(self.s_jk),2) - 2*tf.math.log(tf.math.pow(self.mu,2))*(-11 + 6*self.eg - 6*tf.math.log(4*self.Pi) + 6*tf.math.log((self.s_ij*self.s_jk)/self.s_ijk)) + 18*tf.math.log(self.s_ijk) - 12*self.eg*tf.math.log(self.s_ijk) + 12*tf.math.log(4*self.Pi)*tf.math.log(self.s_ijk) - 6*tf.math.pow(tf.math.log(self.s_ijk),2))/(96.*tf.math.pow(self.Pi,2))
    
        A_3_0 = self.all_a_3_0(i, j, k)
        A_3_1 = (-a*A_3_0+b/2+c/2+d/2)/self.S + self.T*A_3_0/self.S + finite_from_pole*A_3_0
        return A_3_1
    
    def A_3_0(self, i, j, k):
        return tf.reduce_sum(self.all_a_3_0(i, j, k), axis=0)
    
    def all_a_3_0(self, i, j, k):
        return tf.convert_to_tensor([self.a_3_0(i, j, k), self.a_3_0(k, j, i)])
    
    def a_3_0(self, i, j, k):
        self.set_indices(i, j, k)

        a = frac(self.s_jk, self.s_ij)
        b = frac(self.s_ik*self.s_ijk, (self.s_ij+self.s_jk)*self.s_ij)

        a_3_0 = (a + 2*b)/self.s_ijk
        return a_3_0

    def D_3_1(self, i, j, k):
        self.set_indices(i, j, k)

        R_ij_jk = self.R(self.y_ij, self.y_jk)
        R_ik_jk = self.R(self.y_ik, self.y_jk)
        R_ij_ik = self.R(self.y_ij, self.y_ik)

        a = R_ij_jk + R_ik_jk + R_ij_ik
        b = 5/3*(tf.math.log(self.y_ij) + tf.math.log(self.y_ik)) + 11/6*tf.math.log(self.y_jk)
        c = 1/(3*self.s_jk)
        finite_from_pole = -((-22*self.eg + 6*tf.math.pow(self.eg,2) - 7*tf.math.pow(self.Pi,2) + 24*tf.math.pow(tf.math.log(2.0),2) + 12*tf.math.log(4.0)*tf.math.log(self.Pi) + 6*tf.math.pow(tf.math.log(self.Pi),2) + 22*tf.math.log(4*self.Pi) - 12*self.eg*tf.math.log(4*self.Pi) + 6*tf.math.pow(tf.math.log(tf.math.pow(self.mu,2)),2) + 4*(-5 + 3*self.eg - 3*tf.math.log(4*self.Pi))*tf.math.log(self.s_ij) + 6*tf.math.pow(tf.math.log(self.s_ij),2) - 20*tf.math.log(self.s_ik) + 12*self.eg*tf.math.log(self.s_ik) - 12*tf.math.log(4*self.Pi)*tf.math.log(self.s_ik) + 6*tf.math.pow(tf.math.log(self.s_ik),2) - 22*tf.math.log(self.s_jk) + 12*self.eg*tf.math.log(self.s_jk) - 12*tf.math.log(4*self.Pi)*tf.math.log(self.s_jk) + 6*tf.math.pow(tf.math.log(self.s_jk),2) - 2*tf.math.log(tf.math.pow(self.mu,2))*(-11 + 6*self.eg - 6*tf.math.log(4*self.Pi) + 6*tf.math.log(self.s_ij) + 6*tf.math.log(self.s_ik) + 6*tf.math.log(self.s_jk) - 12*tf.math.log(self.s_ijk)) + 40*tf.math.log(self.s_ijk) - 24*self.eg*tf.math.log(self.s_ijk) + 24*tf.math.log(4*self.Pi)*tf.math.log(self.s_ijk) - 12*tf.math.pow(tf.math.log(self.s_ijk),2))/(96.*tf.math.pow(self.Pi,2)))

        D_3_0 = self.all_d_3_0(i, j, k)
        D_3_1 = (-(a+b)*D_3_0 + c/2)/self.S + self.T*D_3_0/self.S + finite_from_pole*D_3_0
        return D_3_1
    
    def D_3_0(self, i, j, k):
        return tf.reduce_sum(self.all_d_3_0(i, j, k), axis=0)
    
    def all_d_3_0(self, i, j, k):
        return tf.convert_to_tensor([self.d_3_0(i, j, k), self.d_3_0(i, k, j)])
    
    def d_3_0(self, i, j, k):
        self.set_indices(i, j, k)
        
        d_ik = 2*frac(self.s_ijk**2*self.s_ik, self.s_ij*self.s_jk) + frac(self.s_ij*self.s_jk + self.s_jk**2, self.s_ik)
        d_ijk = frac(self.s_ij*self.s_ik, self.s_jk) + 5/2*self.s_ijk + 1/2*self.s_jk

        d_3_0 = (d_ik + d_ijk)/self.s_ijk**2
        return d_3_0

    def F_3_1(self, i, j, k):
        self.set_indices(i, j, k)

        R_ij_ik = self.R(self.y_ij, self.y_ik)
        R_ik_jk = self.R(self.y_ik, self.y_jk)
        R_ij_jk = self.R(self.y_ij, self.y_jk)
        R = R_ij_ik + R_ik_jk + R_ij_jk

        Y = 11/6 * (tf.math.log(self.y_ij) + tf.math.log(self.y_ik) + tf.math.log(self.y_jk))
        S = frac(1, self.s_ij) + frac(1, self.s_ik) + frac(1, self.s_jk) + frac(1, self.s_ijk)
        finite_from_pole = -0.010416666666666666*(-22*self.eg + 6*tf.math.pow(self.eg,2) - 7*tf.math.pow(self.Pi,2) + 24*tf.math.pow(tf.math.log(2.0),2) + 12*tf.math.log(4.0)*tf.math.log(self.Pi) + 6*tf.math.pow(tf.math.log(self.Pi),2) + 22*tf.math.log(4*self.Pi) - 12*self.eg*tf.math.log(4*self.Pi) + 6*tf.math.pow(tf.math.log(tf.math.pow(self.mu,2)),2) + 2*(-11 + 6*self.eg - 6*tf.math.log(4*self.Pi))*tf.math.log(self.s_ij) + 6*tf.math.pow(tf.math.log(self.s_ij),2) - 22*tf.math.log(self.s_ik) + 12*self.eg*tf.math.log(self.s_ik) - 12*tf.math.log(4*self.Pi)*tf.math.log(self.s_ik) + 6*tf.math.pow(tf.math.log(self.s_ik),2) - 22*tf.math.log(self.s_jk) + 12*self.eg*tf.math.log(self.s_jk) - 12*tf.math.log(4*self.Pi)*tf.math.log(self.s_jk) + 6*tf.math.pow(tf.math.log(self.s_jk),2) - 2*tf.math.log(tf.math.pow(self.mu,2))*(-11 + 6*self.eg - 6*tf.math.log(4*self.Pi) + 6*tf.math.log(self.s_ij) + 6*tf.math.log(self.s_ik) + 6*tf.math.log(self.s_jk) - 12*tf.math.log(self.s_ijk)) + 44*tf.math.log(self.s_ijk) - 24*self.eg*tf.math.log(self.s_ijk) + 24*tf.math.log(4*self.Pi)*tf.math.log(self.s_ijk) - 12*tf.math.pow(tf.math.log(self.s_ijk),2))/tf.math.pow(self.Pi,2)
        
        F_3_0 = self.all_f_3_0(i, j, k)
        F_3_1 = -(R+Y)*F_3_0/self.S + S/3/self.S/3 + self.T*F_3_0/self.S + finite_from_pole*F_3_0
        return F_3_1
    
    def F_3_0(self, i, j, k):
        return tf.reduce_sum(self.all_f_3_0(i, j, k), axis=0)
    
    def all_f_3_0(self, i, j, k):
        return tf.convert_to_tensor([self.f_3_0(i, k, j), self.f_3_0(k, j, i), self.f_3_0(j, i, k)])
    
    def f_3_0(self, i, j, k):
        self.set_indices(i, j, k)
        
        f_ik = 2*frac(self.s_ijk**2*self.s_ik, self.s_ij*self.s_jk) + frac(self.s_ik*self.s_ij, self.s_jk) + frac(self.s_ik*self.s_jk, self.s_ij) + 8/3*self.s_ijk

        f_3_0 = f_ik / self.s_ijk**2
        return f_3_0
