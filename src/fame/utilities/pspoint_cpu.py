from ctypes import util
from itertools import combinations
from collections import OrderedDict
from fame.utilities import utility_functions

def idot(p1, p2):
    return p1[0]*p2[0] - p1[1]*p2[1] - p1[2]*p2[2] - p1[3]*p2[3]

class PSpoint():
    '''A phase-space point class which can be used to calculate all Mandelstam invariants.'''

    def __init__(self, four_moms, num_jets, w, y_global_cut):
        self.four_moms = four_moms
        self.E = four_moms[:, 0]
        self.px = four_moms[:, 1]
        self.py = four_moms[:, 2]
        self.pz = four_moms[:, 3]
        self.w = w
        self.num_jets = num_jets
        self.combs = list(combinations(range(1, self.num_jets+1), 2))
        self.keys = [''.join(str(comb[0]) + str(comb[1])) for comb in self.combs]
        self.sij = self.calculate_sij()
        self.minsij, self.minkey = min([(s[1], s[0]) for s in list(self.sij.items())])
        if self.minsij < y_global_cut*w**2:
            raise utility_functions.myException("min sij below global phase-space cut.")

        
    def __eq__(self, other):
        if type(self) == type(other):
            return self.four_moms == other.four_moms
        else:
            return False
        
    def __repr__(self):
        return "min sij = {:.1f} (s{})".format(self.minsij, self.minkey)
        
    def __getitem__(self, key):
        return self.four_moms[key]
    
    def __setitem__(self, key, item):
        self.four_moms[key] = item
    
    def __len__(self):
        return len(self.four_moms)
    
    def __neg__(self):
        negative = PSpoint(self.four_moms)
        negative.four_moms = -self.four_moms
        return negative
    
    def __add__(self, other):
        if len(self.four_moms) == len(other.four_moms):
            return self.four_moms + other.four_moms
        else:
            raise utility_functions.myException("Momenta don't have the same shape.")
            
    def __sub__(self, other):
        if len(self.four_moms) == len(other.four_moms):
            return self.four_moms - other.four_moms
        else:
            raise utility_functions.myException("Momenta don't have the same shape.")
            
    def __mul__(self, other):
        if len(self.four_moms) == len(other.four_moms):
            return self.four_moms * other.four_moms
        else:
            raise utility_functions.myException("Momenta don't have the same shape.")
            
    def __div__(self, other):
        if len(self.four_moms) == len(other.four_moms):
            return self.four_moms / other.four_moms
        else:
            raise utility_functions.myException("Momenta don't have the same shape.")
            
    def __truediv__(self, other):
        if len(self.four_moms) == len(other.four_moms):
            return self.four_moms / other.four_moms
        else:
            raise utility_functions.myException("Momenta don't have the same shape.")

            
    def check_momentum_conservation(self):
        sum_mom = np.sum(self.four_moms, axis=0)
        print(sum_mom)
        check_mom_cons = np.isclose(sum_mom, [2*self.w, 0, 0, 0],
                                    rtol=1E-10, atol=1E-10)
                
        if check_mom_cons.all() == True:
            print('\n######## Momentum conserved for this phase space point ##########\n')
        else:
            print('\n######## Momentum NOT conserved for this phase space point ##########\n')
         
        return None
    

    def check_onshell(self):
        masses = utility_functions.dot(self.four_moms, self.four_moms)
        check_onshell = np.isclose(masses, np.zeros_like(masses),
                                   rtol=1E-10, atol=1E-10)
        
        print(masses)
        
        if check_onshell.all() == True:
            print('\n################# All particles are on-shell ####################\n')
        else:
            print('\n################ Not all particles are on-shell #################\n')
         
        return None     


    def calculate_sij(self):
        sij = OrderedDict()
        for i, comb in enumerate(self.combs):
            pi = self.four_moms[comb[0]+1]
            pj = self.four_moms[comb[1]+1]
            sij[self.keys[i]] = 2*idot(pi, pj)
        return sij
