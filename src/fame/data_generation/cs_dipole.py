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
        """
        self.set_indices(i, j, k)
        prefactor = 8*np.pi*self.alpha_s*self.C_F
        rf = self.calculate_recoil_factor()

        return prefactor * (1 + (1-rf)**2)/rf
    
    def calculate_V_gg_II(self, i, j, k):
        """
        Helicity averaged splitting kernel for II gluon-gluon splitting in d=4 dimensions.
        Tends to eikonal in soft limit.
        """
        self.set_indices(i, j, k)
        prefactor = 16*np.pi*self.alpha_s*self.C_A
        rf = self.calculate_recoil_factor()
        
        return prefactor * (rf/(1-rf) + (1-rf)/rf + rf*(1-rf))

    def calculate_V_qq_II(self, i, j, k):
        """
        Helicity averaged splitting kernel for IF quark-quark splitting in d=4 dimensions.
        """
        self.set_indices(i, j, k)
        prefactor = 8*np.pi*self.alpha_s*self.C_F
        rf = self.calculate_recoil_factor()

        return prefactor * (1+rf**2)/(1-rf)

    def calculate_V_gqbar_II(self, i, j, k):
        """
        Helicity averaged splitting kernel for IF gluon-antiquark splitting in d=4 dimensions.
        """
        self.set_indices(i, j, k)
        prefactor = 8*np.pi*self.alpha_s*self.T_R
        rf = self.calculate_recoil_factor()

        return prefactor * (1 - 2*rf*(1-rf))

    ############### IF ###################
    def calculate_V_qg_IF(self, i, j, k):
        """
        Helicity averaged splitting kernel for IF quark-gluon splitting in d=4 dimensions.
        Tends to DGLAP splitting kernel in collinear limit.
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
        """
        self.set_indices(i, j, k)
        prefactor = 16*np.pi*self.alpha_s*self.C_A
        ui = self.calculate_ui()
        rf = self.calculate_recoil_factor()
        
        return prefactor * (1 / (1-rf+ui) + (1-rf) / rf - 1 + rf*(1-rf)) / rf

    def calculate_V_qq_IF(self, i, j, k):
        """
        Helicity averaged splitting kernel for IF quark-quark splitting in d=4 dimensions.
        """
        self.set_indices(i, j, k)
        prefactor = 8*np.pi*self.alpha_s*self.C_F
        rf = self.calculate_recoil_factor()

        return prefactor * (rf + 2*(1-rf)/rf) / rf

    def calculate_V_gqbar_IF(self, i, j, k):
        """
        Helicity averaged splitting kernel for IF gluon-antiquark splitting in d=4 dimensions.
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
        """
        self.set_indices(i, j, k)
        prefactor = 16*np.pi*self.alpha_s*self.C_A
        zi = self.calculate_zi()
        zj = self.calculate_zj()
        rf = self.calculate_recoil_factor()
        
        return prefactor * (1 / (1-zi+(1-rf)) + 1 / (1-zj+(1-rf)) - 2 + zi*zj) / rf

    def calculate_V_qq_FI(self, i, j, k):
        """
        Helicity averaged splitting kernel for FI quark-quark splitting in d=4 dimensions.
        """
        self.set_indices(i, j, k)
        prefactor = 8*np.pi*self.alpha_s*self.T_R
        zi = self.calculate_zi()
        zj = self.calculate_zj()

        return prefactor * (1 - 2*zi*zj) / rf

    ############### FF ###################
    def calculate_V_qg_FF(self, i, j, k):
        """
        Helicity averaged splitting kernel for FF quark-gluon splitting in d=4 dimensions.
        Tends to DGLAP splitting kernel in collinear limit.
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
        """
        self.set_indices(i, j, k)
        prefactor = 8*np.pi*self.alpha_s*self.T_R
        zi = self.calculate_zi()
        zj = self.calculate_zj()

        return prefactor * (1 - 2*zi*zj)

    def calculate_recoil_factor(self):
        """
        Calculate recoil factor depending on spectator incoming or outgoing.
        """
        if self.mode == "FF":
            recoil = self.calculate_y()
        elif self.mode == "FI":
            recoil = self.calculate_x_FI()
        elif self.mode == "IF":
            recoil = self.calculate_x_IF()
        elif self.mode == "II":
            recoil = self.calculate_x_II()
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
        self.pi = self.p[i-1]
        self.pj = self.p[j-1]
        self.pk = self.p[k-1]
        if i in self.outgoing_indices and j in self.outgoing_indices and k in self.outgoing_indices:
            self.mode = "FF"
        elif i in self.outgoing_indices and j in self.outgoing_indices and k in self.incoming_indices:
            self.mode = "FI"
        elif i in self.incoming_indices and j in self.incoming_indices and k in self.outgoing_indices:
            self.mode = "IF"
        elif i in self.incoming_indices and j in self.incoming_indices and k in self.incoming_indices:
            self.mode = "II"
    
    def set_momenta(self, p):
        """
        Set momenta of choice.
        """
        self.p = p
        # self.num_jets = self.p.shape[0]-2
    
    def dot(self, p1, p2):
        """
        Minkowski dot product of two momenta
        """
        return p1[0]*p2[0] - p1[1]*p2[1] - p1[2]*p2[2] - p1[3]*p2[3]


    # def map_momenta_inplace(self, i, j, k):
    #     self.set_indices(i, j, k)
    #     y_ijk = self.calculate_y()
        
    #     q_ij = self.pi + self.pj - y_ijk / (1 - y_ijk) * self.pk
    #     q_k = self.pk / (1 - y_ijk)
        
    #     q = np.delete(self.p, j+1, axis=0)
    #     if i >= j+1:
    #         q[i] = q_ij
    #     else:
    #         q[i+1] = q_ij
    #     if k > j:
    #         q[k] = q_k
    #     else:
    #         q[k+1] = q_k
    #     return q

    # def map_momenta(self):
    #     """
    #     Maps m+1 momenta to m momenta using Catani-Seymour mapping.
    #     Conserves momenta and keeps resulting momenta on-shell.
    #     """
    #     y_ijk = self.calculate_y()
        
    #     q_ij = self.pi + self.pj - y_ijk / (1 - y_ijk) * self.pk
    #     q_k = self.pk / (1 - y_ijk)
    #     return q_ij, q_k