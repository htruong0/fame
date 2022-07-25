import math
import numpy as np
from fame.utilities import utility_functions


class CS_dipole():
    """
    Functions to calculate Catani-Seymour 'dipoles' which are fed as inputs to neural network.
    """
    def __init__(self, alpha_s=0.118, C_F=4/3, C_A=3, T_R=1/2):
        self.alpha_s = alpha_s
        self.C_F = C_F
        self.C_A = C_A
        self.T_R = T_R

    def initialise_z_4j(self):
        """
        Initiliase settings for gg > emepggddx.
        """
        self.incoming_indices = [1, 2]
        self.outgoing_indices = [5, 6, 7, 8]
        self.quarks = [7, 8]
        self.gluons = [1, 2, 5, 6]
        self.massive_quarks = []

    def initialise_ttbar1(self):
        """
        Initiliase settings for uux > ttxgddx.
        """
        self.incoming_indices = [1, 2]
        self.outgoing_indices = [3, 4, 5, 6, 7]
        self.quarks = [1, 2, 3, 4, 6, 7]
        self.gluons = [5]
        self.massive_quarks = [3, 4]

    def initialise_ttbar2(self):
        """
        Initiliase settings for gg > ttxggg.
        """
        self.incoming_indices = [1, 2]
        self.outgoing_indices = [3, 4, 5, 6, 7]
        self.quarks = [3, 4]
        self.gluons = [1, 2, 5, 6, 7]
        self.massive_quarks = [3, 4]

    def calculate_D(self, i, j, k):
        self.set_indices(i, j, k)
        self.calculate_invariants()
        self.calculate_masses()
        if any(x in self.massive_quarks for x in [i, j, k]):
            D = self.calculate_D_massive(i, j, k)
        else:
            D = self.calculate_D_massless(i, j, k)
        return D

    def calculate_RF(self, i, j, k):
        self.set_indices(i, j, k)
        self.calculate_invariants()
        self.calculate_masses()
        if any(x in self.massive_quarks for x in [i, j, k]):
            x = self.calculate_massive_recoil_factor()
        else:
            x = self.calculate_recoil_factor()
        return x

    def map_momenta(self, i, j, k):
        self.set_indices(i, j, k)
        self.calculate_invariants()
        self.calculate_masses()
        if any(x in self.massive_quarks for x in [i, j, k]):
            p_map = self.map_massive_momenta_inplace(i, j, k)
        else:
            p_map = self.map_momenta_inplace(i, j, k)
        return p_map
        
    def calculate_D_massless(self, i, j, k):
        """
        Massless Catani-Seymour dipole without reduced matrix element.
        """
        s_ij = 2*self.dot(self.pi, self.pj)
        if self.mode == "FF":
            if (i in self.quarks and j in self.gluons) or (i in self.gluons and j in self.quarks):
                V = self.calculate_V_qg_FF()
            elif i in self.gluons and j in self.gluons:
                V = self.calculate_V_gg_FF()
            elif i in self.quarks and j in self.quarks:
                V = self.calculate_V_qq_FF()
            D = V / s_ij
        elif self.mode == "FI":
            if (i in self.quarks and j in self.gluons) or (i in self.gluons and j in self.quarks):
                V = self.calculate_V_qg_FI()
            elif i in self.gluons and j in self.gluons:
                V = self.calculate_V_gg_FI()
            elif i in self.quarks and j in self.quarks:
                V = self.calculate_V_qq_FI()
            # already normalised by x
            D = V / s_ij
        elif self.mode == "IF":
            if (i in self.quarks and j in self.gluons):
                V = self.calculate_V_qg_IF()
            elif (i in self.gluons and j in self.quarks):
                V = self.calculate_V_gqbar_IF()
            elif i in self.gluons and j in self.gluons:
                V = self.calculate_V_gg_IF()
            elif i in self.quarks and j in self.quarks:
                V = self.calculate_V_qq_IF()
            # already normalised by x
            D = V / s_ij
        elif self.mode == "II":
            if (i in self.quarks and j in self.gluons):
                V = self.calculate_V_qg_II()
            elif (i in self.gluons and j in self.quarks):
                V = self.calculate_V_gqbar_II()
            elif i in self.gluons and j in self.gluons:
                V = self.calculate_V_gg_II()
            elif i in self.quarks and j in self.quarks:
                V = self.calculate_V_qq_II()
            # already normalised by x
            D = V / s_ij
        return D

    ############### II ###################
    def calculate_V_qg_II(self):
        """
        Helicity averaged splitting kernel for II quark-gluon splitting in d=4 dimensions.
        Tends to DGLAP splitting kernel in collinear limit.
        CS 4.19
        """
        prefactor = 8*math.pi*self.alpha_s*self.C_F
        rf = self.calculate_recoil_factor()

        return prefactor * (1 + (1-rf)**2)/rf**2
    
    def calculate_V_gg_II(self):
        """
        Helicity averaged splitting kernel for II gluon-gluon splitting in d=4 dimensions.
        Tends to eikonal in soft limit.
        CS 4.21
        """
        prefactor = 16*math.pi*self.alpha_s*self.C_A
        rf = self.calculate_recoil_factor()
        
        return prefactor * (rf/(1-rf) + (1-rf)/rf + rf*(1-rf)) / rf

    def calculate_V_qq_II(self):
        """
        Helicity averaged splitting kernel for II quark-quark splitting in d=4 dimensions.
        CS 4.18
        """
        prefactor = 8*math.pi*self.alpha_s*self.C_F
        rf = self.calculate_recoil_factor()

        return prefactor * (1+rf**2)/(1-rf) / rf

    def calculate_V_gqbar_II(self):
        """
        Helicity averaged splitting kernel for II gluon-antiquark splitting in d=4 dimensions.
        CS 4.20
        """
        prefactor = 8*math.pi*self.alpha_s*self.T_R
        rf = self.calculate_recoil_factor()

        return prefactor * (1 - 2*rf*(1-rf)) / rf

    ############### IF ###################
    def calculate_V_qg_IF(self):
        """
        Helicity averaged splitting kernel for IF quark-gluon splitting in d=4 dimensions.
        Tends to DGLAP splitting kernel in collinear limit.
        CS 5.77
        """
        prefactor = 8*math.pi*self.alpha_s*self.C_F
        ui = self.calculate_ui()
        rf = self.calculate_recoil_factor()

        return prefactor * (2 / (1-rf+ui) - (1+rf)) / rf
    
    def calculate_V_gg_IF(self):
        """
        Helicity averaged splitting kernel for IF gluon-gluon splitting in d=4 dimensions.
        Tends to eikonal in soft limit.
        CS 5.79
        """
        prefactor = 16*math.pi*self.alpha_s*self.C_A
        ui = self.calculate_ui()
        rf = self.calculate_recoil_factor()
        
        return prefactor * (1 / (1-rf+ui) + (1-rf) / rf - 1 + rf*(1-rf)) / rf

    def calculate_V_qq_IF(self):
        """
        Helicity averaged splitting kernel for IF quark-quark splitting in d=4 dimensions.
        CS 5.80
        """
        prefactor = 8*math.pi*self.alpha_s*self.C_F
        rf = self.calculate_recoil_factor()

        return prefactor * (rf + 2*(1-rf)/rf) / rf

    def calculate_V_gqbar_IF(self):
        """
        Helicity averaged splitting kernel for IF gluon-antiquark splitting in d=4 dimensions.
        CS 5.78
        """
        prefactor = 8*math.pi*self.alpha_s*self.T_R
        rf = self.calculate_recoil_factor()

        return prefactor * (1 - 2*rf*(1-rf)) / rf
        
    ############### FI ###################
    def calculate_V_qg_FI(self):
        """
        Helicity averaged splitting kernel for FI quark-gluon splitting in d=4 dimensions.
        Tends to DGLAP splitting kernel in collinear limit.
        CS 5.54
        """
        prefactor = 8*math.pi*self.alpha_s*self.C_F
        zi = self.calculate_zi()
        rf = self.calculate_recoil_factor()

        return prefactor * (2 / (1-zi+(1-rf)) - (1+zi)) / rf


    def calculate_V_gg_FI(self):
        """
        Helicity averaged splitting kernel for FI gluon-gluon splitting in d=4 dimensions.
        Tends to eikonal in soft limit.
        CS 5.56
        """
        prefactor = 16*math.pi*self.alpha_s*self.C_A
        zi = self.calculate_zi()
        zj = self.calculate_zj()
        rf = self.calculate_recoil_factor()
        
        return prefactor * (1 / (zj+(1-rf)) + 1 / (zi+(1-rf)) - 2 + zi*zj) / rf

    def calculate_V_qq_FI(self):
        """
        Helicity averaged splitting kernel for FI quark-quark splitting in d=4 dimensions.
        CS 5.55
        """
        prefactor = 8*math.pi*self.alpha_s*self.T_R
        zi = self.calculate_zi()
        zj = self.calculate_zj()
        rf = self.calculate_recoil_factor()

        return prefactor * (1 - 2*zi*zj) / rf

    ############### FF ###################
    def calculate_V_qg_FF(self):
        """
        Helicity averaged splitting kernel for FF quark-gluon splitting in d=4 dimensions.
        Tends to DGLAP splitting kernel in collinear limit.
        CS 5.29
        """
        prefactor = 8*math.pi*self.alpha_s*self.C_F
        zi = self.calculate_zi()
        rf = self.calculate_recoil_factor()

        return prefactor * (2 / (1-zi*(1-rf)) - (1+zi))

    def calculate_V_gg_FF(self):
        """
        Helicity averaged splitting kernel for FF gluon-gluon splitting in d=4 dimensions.
        Tends to eikonal in soft limit.
        CS 5.31
        """
        prefactor = 16*math.pi*self.alpha_s*self.C_A
        zi = self.calculate_zi()
        zj = self.calculate_zj()
        rf = self.calculate_recoil_factor()
        
        return prefactor * (1 / (1-zi*(1-rf)) + 1 / (1-zj*(1-rf)) - 2 + zi*zj)

    def calculate_V_qq_FF(self):
        """
        Helicity averaged splitting kernel for FF quark-quark splitting in d=4 dimensions.
        CS 5.30
        """
        prefactor = 8*math.pi*self.alpha_s*self.T_R
        zi = self.calculate_zi()
        zj = self.calculate_zj()

        return prefactor * (1 - 2*zi*zj)

    def map_momenta_inplace(self, i, j, k):
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
            if i < j:
                q[i-1] = q_ij
            elif i > j:
                q[i-2] = q_ij
            if k < j:
                q[k-1] = q_k
            elif k > j:
                q[k-2] = q_k
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


    ############# Massive dipoles ###############

    def calculate_D_massive(self, i, j, k):
        """
        Massive Catani-Seymour dipole without reduced matrix element.
        """
        if self.mode == "FF":
            if i in self.quarks and j in self.gluons:
                self.mij2 = self.mi2
                Vm = self.calculate_Vm_gQ_FF()
            elif i in self.gluons and j in self.quarks:
                self.mij2 = self.mj2
                Vm = self.calculate_Vm_gQ_FF()
            elif i in self.gluons and j in self.gluons:
                self.mij2 = 0
                Vm = self.calculate_Vm_gg_FF()
            elif i in self.quarks and j in self.quarks:
                self.mij2 = 0
                Vm = self.calculate_Vm_QQ_FF()
            pij = self.pi + self.pj
            d = self.dot(pij, pij) - self.mij2
            Dm = Vm / d
        elif self.mode == "FI":
            if i in self.quarks and j in self.gluons:
                self.mij2 = self.mi2
                Vm = self.calculate_Vm_gQ_FI()
            elif i in self.gluons and j in self.quarks:
                self.mij2 = self.mj2
                Vm = self.calculate_Vm_gQ_FI()
            elif i in self.quarks and j in self.quarks:
                self.mij2 = 0
                Vm = self.calculate_Vm_QQ_FI()
            pij = self.pi + self.pij
            d = self.dot(pij, pij) - self.mij2
            # already normalised by x
            Dm = Vm / d
        elif self.mode == "IF":
            if (i in self.quarks and j in self.gluons):
                self.mij2 = 0
                Vm = self.calculate_Vm_qg_IF()
            elif (i in self.gluons and j in self.quarks):
                self.mij2 = 0
                Vm = self.calculate_Vm_gqbar_IF()
            elif i in self.gluons and j in self.gluons:
                self.mij2 = 0
                Vm = self.calculate_Vm_gg_IF()
            elif i in self.quarks and j in self.quarks:
                self.mij2 = 0
                Vm = self.calculate_Vm_qq_IF()
            d = 2*self.dot(self.pi, self.pj)
            # already normalised by x
            Dm = Vm / d
        return Dm


    ############### IF (massive) ###################
    def calculate_Vm_qg_IF(self):
        """
        CS (massive) 5.81
        """
        x = self.calculate_massive_recoil_factor()
        # reuse u variable because it's the same as z for IF
        zj = self.calculate_uk()
        self.calculate_rescaled_masses_IF(x)
        prefactor = 8*math.pi*self.alpha_s*self.C_F

        return prefactor * (2/(2-x-zj) - 1 - x) / x


    def calculate_Vm_gqbar_IF(self):
        """
        CS (massive) 5.82
        """
        x = self.calculate_massive_recoil_factor()
        prefactor = 8*math.pi*self.alpha_s*self.T_R

        return prefactor * (1 - 2*x*(1-x)) / x


    def calculate_Vm_qq_IF(self):
        """
        CS (massive) 5.84
        """
        x = self.calculate_massive_recoil_factor()
        self.calculate_rescaled_masses(x)
        zi = self.calculate_ui()
        zj = self.calculate_uk()
        prefactor = 8*math.pi*self.alpha_s*self.C_F

        return prefactor * (x + 2*(1-x)/x - 2*self.muk2/x*zi/zj) / x


    def calculate_Vm_gg_IF(self):
        """
        CS (massive) 5.86
        """
        x = self.calculate_massive_recoil_factor()
        self.calculate_rescaled_masses(x)
        zi = self.calculate_ui()
        zj = self.calculate_uk()
        prefactor = 16*math.pi*self.alpha_s*self.C_A

        return prefactor * (1/(2-x-zj) - 1 + x*(1-x) + (1-x)/x - self.muk2/x*zi/zj) / x


    ############### FI (massive) ###################
    def calculate_Vm_gQ_FI(self):
        """
        CS (massive) 5.50
        """
        x = self.calculate_massive_recoil_factor()
        self.calculate_rescaled_masses(x)
        zj = self.calculate_zj()
        prefactor = 8*math.pi*self.alpha_s*self.C_F
        pipj = self.dot(self.pi, self.pj)
        return prefactor * (2 / (2-x-zj) - 1 - zj - self.mij2/pipj) / x


    def calculate_Vm_QQ_FI(self):
        """
        CS (massive) 5.52
        """
        x = self.calculate_massive_recoil_factor()
        self.calculate_rescaled_masses(x)
        zi = self.calculate_zi()
        prefactor = 8*math.pi*self.alpha_s*self.T_R

        a = 1 - x + self.muij2 + self.mui2 - self.muj2
        b = ((1 - x + self.muij2 - self.mui2 - self.muj2)**2 - 4*self.mui2*self.muj2)**0.5
        c = 2*(1 - x + self.muij2)
        zp = (a + b) / c
        zm = (a - b) / c
        return prefactor * (1 - 2*(zp-zi)*(zi-zm)) / x

    
    ############### FF (massive) ###################
    def calculate_Vm_gQ_FF(self):
        """
        CS (massive) 5.16
        """
        self.calculate_rescaled_masses()
        yijk = self.calculate_massive_recoil_factor()
        zj = self.calculate_zj()
        prefactor = 8*math.pi*self.alpha_s*self.C_F
        vt = (self.triangular(1, self.muij2, self.muk2))**0.5 / (1-self.muij2-self.muk2)
        a = (1-self.mui2-self.muj2-self.muk2)*(1-yijk)
        v = ((2*self.muk2+a)**2 - 4*self.muk2)**0.5 / a
        mQ2 = self.mj2
        pipj = self.dot(self.pi, self.pj)

        return prefactor * (2/(1-zj*(1-yijk)) - vt/v*(1+zj+mQ2/pipj))


    def calculate_Vm_QQ_FF(self):
        """
        CS (massive) 5.18
        """
        self.calculate_rescaled_masses()
        yijk = self.calculate_massive_recoil_factor()
        zi = self.calculate_zi()
        zj = self.calculate_zj()
        prefactor = 8*math.pi*self.alpha_s*self.T_R
        a = (1-self.mui2-self.muj2-self.muk2)*(1-yijk)
        b = (1-self.mui2-self.muj2-self.muk2)*yijk
        vijk = ((2*self.muk2+a)**2 - 4*self.muk2)**0.5 / a
        viji = (b**2 - 4*self.mui2*self.muj2)**0.5 / (2*self.mui2 + b)
        zp = (2*self.mui2 + b)*(1 + viji*vijk)/(2*(self.mui2 + self.muj2 + b))
        zm = (2*self.mui2 + b)*(1 - viji*vijk)/(2*(self.mui2 + self.muj2 + b))
        
        # kappa = 0
        return prefactor/vijk * (1 - 2*(zi*zj - zp*zm))


    def calculate_Vm_gg_FF(self):
        """
        CS (massive) 5.20
        """
        self.calculate_rescaled_masses()
        yijk = self.calculate_massive_recoil_factor()
        zi = self.calculate_zi()
        zj = self.calculate_zj()
        prefactor = 16*math.pi*self.alpha_s*self.C_A
        a = (1-self.mui2-self.muj2-self.muk2)*(1-yijk)
        b = (1-self.mui2-self.muj2-self.muk2)*yijk
        vijk = ((2*self.muk2+a)**2 - 4*self.muk2)**0.5 / a
        viji = (b**2 - 4*self.mui2*self.muj2)**0.5 / (2*self.mui2 + b)
        zp = (2*self.mui2 + b)*(1 + viji*vijk)/(2*(self.mui2 + self.muj2 + b))
        zm = (2*self.mui2 + b)*(1 - viji*vijk)/(2*(self.mui2 + self.muj2 + b))

        # kappa = 0
        return prefactor * (1/(1-zi*(1-yijk)) + 1/(1-zj*(1-yijk)) + (zi*zj-zp*zm-2)/vijk)

    def map_massive_momenta_inplace(self, i, j, k):
        if self.mode == "FF":
            if i in self.quarks and j in self.gluons:
                self.mij2 = self.mi2
            elif i in self.gluons and j in self.quarks:
                self.mij2 = self.mj2
            elif i in self.gluons and j in self.gluons:
                self.mij2 = 0
            elif i in self.quarks and j in self.quarks:
                self.mij2 = 0
            q_ij, q_k = self.map_massive_momenta_FF()
        elif self.mode == "FI":
            if i in self.quarks and j in self.gluons:
                self.mij2 = self.mi2
            elif i in self.gluons and j in self.quarks:
                self.mij2 = self.mj2
            elif i in self.quarks and j in self.quarks:
                self.mij2 = 0
            q_ij, q_k = self.map_massive_momenta_FI()
        elif self.mode == "IF":
            self.mij2 = 0
            q_ij, q_k = self.map_massive_momenta_IF()
        else:
            raise utility_functions.myException("Mode not set.")

        q = np.delete(self.p, j-1, axis=0)
        if self.mode in ["FF", "FI", "IF"]:
            if i < j:
                q[i-1] = q_ij
            elif i > j:
                q[i-2] = q_ij
            if k < j:
                q[k-1] = q_k
            elif k > j:
                q[k-2] = q_k
        elif self.mode == "II":
            q[i-1] = q_ij
            q[2:] = q_k
        return q


    def map_massive_momenta_IF(self):
        x = self.calculate_massive_recoil_factor()
        q_ij = x*self.pi
        q_k = self.pj + self.pk - (1-x)*self.pi
        return q_ij, q_k

    def map_massive_momenta_FI(self):
        x = self.calculate_massive_recoil_factor()
        q_ij = self.pi + self.pj - (1-x)*self.pk
        q_k = x*self.pk
        return q_ij, q_k

    def map_massive_momenta_FF(self):
        Q = self.pi + self.pj + self.pk
        Q2 = self.dot(Q, Q)
        pippj = self.pi + self.pj
        pipj2 = self.dot(pippj, pippj)
        lambda1 = (self.triangular(Q2, self.mij2, self.mk2))**0.5
        lambda2 = (self.triangular(Q2, pipj2, self.mk2))**0.5
        q_k = lambda1/lambda2*(self.pk - self.dot(Q, self.pk)*Q/Q2) + (Q2 + self.mk2 - self.mij2)*Q/(2*Q2)
        q_ij = Q - q_k
        return q_ij, q_k


    def calculate_rescaled_masses(self, x=None):
        """
        Calculate rescaled masses (mu) depending on spectator incoming or outgoing.
        """
        if self.mode == "FF":
            self.calculate_rescaled_masses_FF()
        elif self.mode == "FI":
            self.calculate_rescaled_masses_FI(x)
        elif self.mode == "IF":
            self.calculate_rescaled_masses_IF(x)


    def calculate_rescaled_masses_IF(self, x):
        pjt = self.pj + self.pk - (1-x)*self.pi
        norm = 2*self.dot(pjt, self.pi)
        self.mui2 = self.mi2 / norm
        self.muj2 = self.mj2 / norm
        self.muk2 = self.mk2 / norm
        self.muij2 = self.mij2 / norm


    def calculate_rescaled_masses_FI(self, x):
        pijt = self.pi + self.pj - (1-x)*self.pk
        norm = 2*self.dot(pijt, self.pk)
        self.mui2 = self.mi2 / norm
        self.muj2 = self.mj2 / norm
        self.muk2 = self.mk2 / norm
        self.muij2 = self.mij2 / norm


    def calculate_rescaled_masses_FF(self):
        Q = self.pi + self.pj + self.pk
        Q2 = self.dot(Q, Q)
        self.mui2 = self.mi2 / Q2
        self.muj2 = self.mj2 / Q2
        self.muk2 = self.mk2 / Q2
        self.muij2 = self.mij2 / Q2


    def calculate_massive_recoil_factor(self):
        """
        Calculate recoil factor depending on spectator incoming or outgoing.
        """
        if self.mode == "FF":
            recoil = self.calculate_ym_FF()
        elif self.mode == "FI":
            recoil = self.calculate_xm_FI()
        elif self.mode == "IF":
            recoil = self.calculate_xm_IF()
        self.recoil = recoil
        return recoil


    def calculate_xm_IF(self):
        # reuse x from massless case because it's the same
        x = self.calculate_x_IF()
        return x


    def calculate_xm_FI(self):
        return (self.pik + self.pjk - self.pij + 0.5*(self.mij2-self.mi2-self.mj2)) / (self.pik + self.pjk)


    def calculate_ym_FF(self):
        # reuse y from massless case because it's the same
        y = self.calculate_y_FF()
        return y


    def calculate_masses(self):
        """
        Calculate mass for partons i, j, k.
        """
        self.mi2 = self.dot(self.pi, self.pi)
        self.mj2 = self.dot(self.pj, self.pj)
        self.mk2 = self.dot(self.pk, self.pk)

    def triangular(self, x, y, z):
        """
        Triangular function for calculating rescaled masses.
        """
        return x**2 + y**2 + z**2 - 2*x*y - 2*x*z - 2*y*z
