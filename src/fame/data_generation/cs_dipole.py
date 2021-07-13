import numpy as np

class CS_dipole():
    """
    Functions to calculate Catani-Seymour 'dipoles' which are fed as inputs to neural network.
    """
    def __init__(self, mode='gluon', alpha_s=0.118, C_F=4/3, C_A=3, T_R=1/2):
        self.mode = mode
        self.alpha_s = alpha_s
        self.C_F = C_F
        self.C_A = C_A
        self.T_R = T_R
        
    def calculate_D_ijk(self, i, j, k):
        """
        Catani-Seymour dipole without reduced matrix element.
        """
        self.set_indices(i, j, k)
        s_ij = 2*self.dot(self.pi, self.pj)

        quark_indices = [1, 2]
        if self.mode == 'gluon':
            gluon_indices = range(3, self.num_jets+1)
        elif self.mode == 'quark':
            quark_indices.extend([3, 4])
            gluon_indices = [5]
        if (i in quark_indices and j in gluon_indices) or (i in gluon_indices and j in quark_indices):
            V_ijk = self.calculate_V_qg(i, j, k)
        elif i in gluon_indices and j in gluon_indices:
            V_ijk = self.calculate_V_gg(i, j, k)
        elif i in quark_indices and j in quark_indices:
            V_ijk = self.calculate_V_qq(i, j, k)
        D_ijk = 1 / s_ij * V_ijk
        return D_ijk
        
    def calculate_V_qg(self, i, j, k):
        """
        Helicity averaged splitting kernel for quark-gluon splitting in d=4 dimensions.
        Taken from Black Book.
        Tends to DGLAP splitting kernel in collinear limit.
        """
        self.set_indices(i, j, k)
        prefactor = 8*np.pi*self.alpha_s*self.C_F
        zi = self.calculate_zi()
        y_ijk = self.calculate_y()

        return prefactor * (2 / (1-zi*(1-y_ijk)) - (1+zi))
    
    def calculate_V_gg(self, i, j, k):
        """
        Helicity averaged splitting kernel for gluon-gluon splitting in d=4 dimensions.
        Taken from Black Book.
        Tends to eikonal in soft limit.
        """
        self.set_indices(i, j, k)
        prefactor = 16*np.pi*self.alpha_s*self.C_A
        zi = self.calculate_zi()
        zj = self.calculate_zj()
        y_ijk = self.calculate_y()
        
        return prefactor * (1 / (1-zi*(1-y_ijk)) + 1 / (1-zj*(1-y_ijk)) - 2 + zi*zj)

    def calculate_V_qq(self, i, j, k):
        """
        Helicity averaged splitting kernel for quark-quark splitting in d=4 dimensions.
        Taken from Black Book.
        """
        self.set_indices(i, j, k)
        prefactor = 8*np.pi*self.alpha_s*self.T_R
        zi = self.calculate_zi()
        zj = self.calculate_zj()

        return prefactor * (1 - 2*zi*zj)
    
    def map_momenta_inplace(self, i, j, k):
        self.set_indices(i, j, k)
        y_ijk = self.calculate_y()
        
        q_ij = self.pi + self.pj - y_ijk / (1 - y_ijk) * self.pk
        q_k = self.pk / (1 - y_ijk)
        
        q = np.delete(self.p, j+1, axis=0)
        if i >= j+1:
            q[i] = q_ij
        else:
            q[i+1] = q_ij
        if k > j:
            q[k] = q_k
        else:
            q[k+1] = q_k
        return q

    def map_momenta(self):
        """
        Maps m+1 momenta to m momenta using Catani-Seymour mapping.
        Conserves momenta and keeps resulting momenta on-shell.
        """
        y_ijk = self.calculate_y()
        
        q_ij = self.pi + self.pj - y_ijk / (1 - y_ijk) * self.pk
        q_k = self.pk / (1 - y_ijk)
        return q_ij, q_k

    def calculate_y(self):
        """
        Calculate recoil factor for momenta mapping.
        """
        pij = self.dot(self.pi, self.pj)
        pik = self.dot(self.pi, self.pk)
        pjk = self.dot(self.pj, self.pk)
        return pij / (pij + pik + pjk)
    
    def calculate_zi(self):
        """
        Calculate splitting variable for emitter.
        """
        pik = self.dot(self.pi, self.pk)
        pjk = self.dot(self.pj, self.pk)
        return pik / (pik + pjk)
    
    def calculate_zj(self):
        """
        Calculate splitting variable for emitted parton.
        """
        return 1 - self.calculate_zi()
    
    def set_indices(self, i, j, k):
        """
        Set indices of emitter (i), emitted (j), and spectator (k).
        """
        self.pi = self.p[i+1]
        self.pj = self.p[j+1]
        self.pk = self.p[k+1]
    
    def set_momenta(self, p):
        """
        Set momenta of choice.
        """
        self.p = p
        self.num_jets = self.p.shape[0]-2
        if self.num_jets <= 3 and self.mode != 'gluon':
            raise Exception("Less than 4 jets has to be in gluon mode.")
    
    def dot(self, p1, p2):
        """
        Minkowski dot product of two momenta
        """
        return p1[0]*p2[0] - p1[1]*p2[1] - p1[2]*p2[2] - p1[3]*p2[3]
