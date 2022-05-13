import numpy as np
from fame_pp.utilities import utility_functions

class CS_dipole():
    """
    Functions to calculate Catani-Seymour 'dipoles' which are fed as inputs to neural network.
    """
    def __init__(self, alpha_s=0.118, C_F=4/3, C_A=3, T_R=1/2):
        self.alpha_s = alpha_s
        self.C_F = C_F
        self.C_A = C_A
        self.T_R = T_R
        self.incoming_indices = [1, 2]
        self.outgoing_indices = [5, 6, 7, 8]
        self.quarks = [7, 8]
        self.gluons = [1, 2, 5, 6]
        
    def calculate_D(self, i, j, k):
        """
        Catani-Seymour dipole without reduced matrix element.
        """
        self.set_indices(i, j, k)
        self.calculate_invariants()
        s_ij = 2*self.dot(self.pi, self.pj)
        if self.mode == "FF":
            if (i in self.quarks and j in self.gluons) or (i in self.gluons and j in self.quarks):
                V = self.calculate_V_qg_FF(i, j, k)
            elif i in self.gluons and j in self.gluons:
                V = self.calculate_V_gg_FF(i, j, k)
            elif i in self.quarks and j in self.quarks:
                V = self.calculate_V_qq_FF(i, j, k)
            D = V / s_ij
        elif self.mode == "FI":
            if (i in self.quarks and j in self.gluons) or (i in self.gluons and j in self.quarks):
                V = self.calculate_V_qg_FI(i, j, k)
            elif i in self.gluons and j in self.gluons:
                V = self.calculate_V_gg_FI(i, j, k)
            elif i in self.quarks and j in self.quarks:
                V = self.calculate_V_qq_FI(i, j, k)
            # already normalised by x
            D = V / s_ij
        elif self.mode == "IF":
            if (i in self.quarks and j in self.gluons):
                V = self.calculate_V_qg_IF(i, j, k)
            elif (i in self.gluons and j in self.quarks):
                V = self.calculate_V_gqbar_IF(i, j, k)
            elif i in self.gluons and j in self.gluons:
                V = self.calculate_V_gg_IF(i, j, k)
            elif i in self.quarks and j in self.quarks:
                V = self.calculate_V_qq_IF(i, j, k)
            # already normalised by x
            D = V / s_ij
        elif self.mode == "II":
            if (i in self.quarks and j in self.gluons):
                V = self.calculate_V_qg_II(i, j, k)
            elif (i in self.gluons and j in self.quarks):
                V = self.calculate_V_gqbar_II(i, j, k)
            elif i in self.gluons and j in self.gluons:
                V = self.calculate_V_gg_II(i, j, k)
            elif i in self.quarks and j in self.quarks:
                V = self.calculate_V_qq_II(i, j, k)
            # already normalised by x
            D = V / s_ij
        return D

    ############### II ###################
    def calculate_V_qg_II(self, i, j, k):
        """
        Helicity averaged splitting kernel for II quark-gluon splitting in d=4 dimensions.
        Tends to DGLAP splitting kernel in collinear limit.
        CS 4.19
        """
        self.set_indices(i, j, k)
        prefactor = 8*np.pi*self.alpha_s*self.C_F
        rf = self.calculate_recoil_factor()

        return prefactor * (1 + (1-rf)**2)/rf**2
    
    def calculate_V_gg_II(self, i, j, k):
        """
        Helicity averaged splitting kernel for II gluon-gluon splitting in d=4 dimensions.
        Tends to eikonal in soft limit.
        CS 4.21
        """
        self.set_indices(i, j, k)
        prefactor = 16*np.pi*self.alpha_s*self.C_A
        rf = self.calculate_recoil_factor()
        
        return prefactor * (rf/(1-rf) + (1-rf)/rf + rf*(1-rf)) / rf

    def calculate_V_qq_II(self, i, j, k):
        """
        Helicity averaged splitting kernel for II quark-quark splitting in d=4 dimensions.
        CS 4.18
        """
        self.set_indices(i, j, k)
        prefactor = 8*np.pi*self.alpha_s*self.C_F
        rf = self.calculate_recoil_factor()

        return prefactor * (1+rf**2)/(1-rf) / rf

    def calculate_V_gqbar_II(self, i, j, k):
        """
        Helicity averaged splitting kernel for II gluon-antiquark splitting in d=4 dimensions.
        CS 4.20
        """
        self.set_indices(i, j, k)
        prefactor = 8*np.pi*self.alpha_s*self.T_R
        rf = self.calculate_recoil_factor()

        return prefactor * (1 - 2*rf*(1-rf)) / rf

    ############### IF ###################
    def calculate_V_qg_IF(self, i, j, k):
        """
        Helicity averaged splitting kernel for IF quark-gluon splitting in d=4 dimensions.
        Tends to DGLAP splitting kernel in collinear limit.
        CS 5.77
        """
        self.set_indices(i, j, k)
        prefactor = 8*np.pi*self.alpha_s*self.C_F
        ui = self.calculate_ui()
        rf = self.calculate_recoil_factor()

        return prefactor * (2 / (1-rf+ui) - (1+rf)) / rf
    
    def calculate_V_gg_IF(self, i, j, k):
        """
        Helicity averaged splitting kernel for IF gluon-gluon splitting in d=4 dimensions.
        Tends to eikonal in soft limit.
        CS 5.79
        """
        self.set_indices(i, j, k)
        prefactor = 16*np.pi*self.alpha_s*self.C_A
        ui = self.calculate_ui()
        rf = self.calculate_recoil_factor()
        
        return prefactor * (1 / (1-rf+ui) + (1-rf) / rf - 1 + rf*(1-rf)) / rf

    def calculate_V_qq_IF(self, i, j, k):
        """
        Helicity averaged splitting kernel for IF quark-quark splitting in d=4 dimensions.
        CS 5.80
        """
        self.set_indices(i, j, k)
        prefactor = 8*np.pi*self.alpha_s*self.C_F
        rf = self.calculate_recoil_factor()

        return prefactor * (rf + 2*(1-rf)/rf) / rf

    def calculate_V_gqbar_IF(self, i, j, k):
        """
        Helicity averaged splitting kernel for IF gluon-antiquark splitting in d=4 dimensions.
        CS 5.78
        """
        self.set_indices(i, j, k)
        prefactor = 8*np.pi*self.alpha_s*self.T_R
        rf = self.calculate_recoil_factor()

        return prefactor * (1 - 2*rf*(1-rf)) / rf
        
    ############### FI ###################
    def calculate_V_qg_FI(self, i, j, k):
        """
        Helicity averaged splitting kernel for FI quark-gluon splitting in d=4 dimensions.
        Tends to DGLAP splitting kernel in collinear limit.
        CS 5.54
        """
        self.set_indices(i, j, k)
        prefactor = 8*np.pi*self.alpha_s*self.C_F
        zi = self.calculate_zi()
        rf = self.calculate_recoil_factor()

        return prefactor * (2 / (1-zi+(1-rf)) - (1+zi)) / rf


    def calculate_V_gg_FI(self, i, j, k):
        """
        Helicity averaged splitting kernel for FI gluon-gluon splitting in d=4 dimensions.
        Tends to eikonal in soft limit.
        CS 5.56
        """
        self.set_indices(i, j, k)
        prefactor = 16*np.pi*self.alpha_s*self.C_A
        zi = self.calculate_zi()
        zj = self.calculate_zj()
        rf = self.calculate_recoil_factor()
        
        return prefactor * (1 / (zj+(1-rf)) + 1 / (zi+(1-rf)) - 2 + zi*zj) / rf

    def calculate_V_qq_FI(self, i, j, k):
        """
        Helicity averaged splitting kernel for FI quark-quark splitting in d=4 dimensions.
        CS 5.55
        """
        self.set_indices(i, j, k)
        prefactor = 8*np.pi*self.alpha_s*self.T_R
        zi = self.calculate_zi()
        zj = self.calculate_zj()
        rf = self.calculate_recoil_factor()

        return prefactor * (1 - 2*zi*zj) / rf

    ############### FF ###################
    def calculate_V_qg_FF(self, i, j, k):
        """
        Helicity averaged splitting kernel for FF quark-gluon splitting in d=4 dimensions.
        Tends to DGLAP splitting kernel in collinear limit.
        CS 5.29
        """
        self.set_indices(i, j, k)
        prefactor = 8*np.pi*self.alpha_s*self.C_F
        zi = self.calculate_zi()
        rf = self.calculate_recoil_factor()

        return prefactor * (2 / (1-zi*(1-rf)) - (1+zi))

    def calculate_V_gg_FF(self, i, j, k):
        """
        Helicity averaged splitting kernel for FF gluon-gluon splitting in d=4 dimensions.
        Tends to eikonal in soft limit.
        CS 5.31
        """
        self.set_indices(i, j, k)
        prefactor = 16*np.pi*self.alpha_s*self.C_A
        zi = self.calculate_zi()
        zj = self.calculate_zj()
        rf = self.calculate_recoil_factor()
        
        return prefactor * (1 / (1-zi*(1-rf)) + 1 / (1-zj*(1-rf)) - 2 + zi*zj)

    def calculate_V_qq_FF(self, i, j, k):
        """
        Helicity averaged splitting kernel for FF quark-quark splitting in d=4 dimensions.
        CS 5.30
        """
        self.set_indices(i, j, k)
        prefactor = 8*np.pi*self.alpha_s*self.T_R
        zi = self.calculate_zi()
        zj = self.calculate_zj()

        return prefactor * (1 - 2*zi*zj)

    def map_momenta_inplace(self, i, j, k):
        self.set_indices(i, j, k)
        self.calculate_invariants()

        if self.mode == "FF":
            q_ij, q_k = self.map_momenta_FF()
        elif self.mode == "FI":
            q_ij, q_k = self.map_momenta_FI()
        elif self.mode == "IF":
            q_ij, q_k = self.map_momenta_IF()
        elif self.mode == "II":
            q_ij, q_k = self.map_momenta_II()
        else:
            raise utility_functions.myException("Mode not set.")

        q = np.delete(self.p, j-1, axis=0)
        if self.mode in ["FF", "FI", "IF"]:
            if i >= j-1:
                q[i-2] = q_ij
            else:
                q[i-1] = q_ij
            if k > j:
                q[k-2] = q_k
            else:
                q[k-1] = q_k
        elif self.mode == "II":
            q[i-1] = q_ij
            q[2:] = q_k
        return q

    def map_momenta_II(self):
        rf = self.calculate_recoil_factor()

        q_ij = rf*self.pi
        K = self.pi + self.pk - self.pj
        K2 = self.dot(K, K)
        Ktilde = q_ij + self.pk
        Ks = K + Ktilde
        Ks2 = self.dot(Ks, Ks)

        k = np.delete(self.p[2:], self.j-3, axis=0)
        q_k = np.zeros_like(k)
        for ik, kj in enumerate(k):
            q_k[ik] = kj - 2*self.dot(kj, Ks)*Ks/Ks2 + 2*self.dot(kj, K)*Ktilde/K2
        return q_ij, q_k

    def map_momenta_IF(self):
        rf = self.calculate_recoil_factor()

        q_ij = rf*self.pi
        q_k = self.pk + self.pj - (1-rf)*self.pi
        return q_ij, q_k

    def map_momenta_FI(self):
        rf = self.calculate_recoil_factor()

        q_ij = self.pi + self.pj - (1-rf)*self.pk
        q_k = rf*self.pk
        return q_ij, q_k

    def map_momenta_FF(self):
        """
        Maps m+1 momenta to m momenta using Catani-Seymour mapping.
        Conserves momenta and keeps resulting momenta on-shell.
        """
        rf = self.calculate_recoil_factor()
        
        q_ij = self.pi + self.pj - rf / (1 - rf) * self.pk
        q_k = self.pk / (1 - rf)
        return q_ij, q_k

    def calculate_recoil_factor(self):
        """
        Calculate recoil factor depending on spectator incoming or outgoing.
        """
        if self.mode == "FF":
            recoil = self.calculate_y_FF()
        elif self.mode == "FI":
            recoil = self.calculate_x_FI()
        elif self.mode == "IF":
            recoil = self.calculate_x_IF()
        elif self.mode == "II":
            recoil = self.calculate_x_II()
        self.recoil = recoil
        return recoil

    def calculate_x_II(self):
        """
        Calculate II recoil factor for momenta mapping.
        II: i = a, k = b
        """
        return (self.pik - self.pij - self.pjk) / self.pik

    def calculate_x_IF(self):
        """
        Calculate IF recoil factor for momenta mapping. 
        IF: i = a
        """
        return (self.pik + self.pij - self.pjk) / (self.pij + self.pik)

    def calculate_x_FI(self):
        """
        Calculate FI recoil factor for momenta mapping. 
        FI: k = a
        """
        return (self.pik + self.pjk - self.pij) / (self.pik + self.pjk)

    def calculate_y_FF(self):
        """
        Calculate FF recoil factor for momenta mapping.
        """
        return self.pij / (self.pij + self.pik + self.pjk)

    def calculate_ui(self):
        """
        Calculate splitting variable for I emitter F spectator.
        IF: i = a
        """
        return self.pij / (self.pij + self.pik)

    def calculate_uk(self):
        """
        Calculate splitting variable for I emitted F spectator.
        """
        return 1 - self.calculate_ui()

    def calculate_zi(self):
        """
        Calculate splitting variable for FF and FI.
        """
        return self.pik / (self.pik + self.pjk)
    
    def calculate_zj(self):
        """
        Calculate splitting variable for FF and FI.
        """
        return 1 - self.calculate_zi()

    def calculate_invariants(self):
        self.pij = self.dot(self.pi, self.pj)
        self.pik = self.dot(self.pi, self.pk)
        self.pjk = self.dot(self.pj, self.pk)
    
    def set_indices(self, i, j, k):
        """
        Set indices of emitter (i), emitted (j), and spectator (k) and mode between:
            FF: final-state splitter, final-state spectator
            FI: final-state splitter, initial-state spectator
            IF: initial-state splitter, final-state spectator
            II: initial-state splitter, initial-state spectator.
        """
        self.i = i
        self.j = j
        self.k = k
        self.pi = self.p[i-1]
        self.pj = self.p[j-1]
        self.pk = self.p[k-1]
        if i in self.outgoing_indices and j in self.outgoing_indices and k in self.outgoing_indices:
            self.mode = "FF"
        elif i in self.outgoing_indices and j in self.outgoing_indices and k in self.incoming_indices:
            self.mode = "FI"
        elif i in self.incoming_indices and j in self.incoming_indices and k in self.outgoing_indices:
            self.mode = "IF"
        elif i in self.incoming_indices and j in self.outgoing_indices and k in self.outgoing_indices:
            self.mode = "IF"
        elif i in self.incoming_indices and j in self.incoming_indices and k in self.incoming_indices:
            self.mode = "II"
        elif i in self.incoming_indices and j in self.outgoing_indices and k in self.incoming_indices:
            self.mode = "II"
        else:
            self.mode = None
            raise utility_functions.myException("Indices don't correspond to valid configuration.")
    
    def set_momenta(self, p):
        """
        Set momenta of choice.
        """
        self.p = p
    
    def dot(self, p1, p2):
        """
        Minkowski dot product of two momenta
        """
        return p1[0]*p2[0] - p1[1]*p2[1] - p1[2]*p2[2] - p1[3]*p2[3]
