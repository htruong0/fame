import numpy as np
import tensorflow as tf
from itertools import combinations

from fame.utilities import utility_functions, pspoint

def calc_phi(p, i, j):
    '''
    Calculate an angle named phi that is the angle relative
    to our defined north on the plane perpendicular to the 
    averaged direction of the chosen indices i and j.
    '''
    pijv = p[:, i-1, 1:] + p[:, j-1, 1:]
    p2 = np.sum(pijv*pijv, axis=1)
    # take reference to be x direction because we can have initial state radiation now
    l = p2 / pijv[:, 0]
    ez = np.zeros((len(p), 3))
    ez[:, 0] = l
    nord = ez - pijv
    nord = nord / np.linalg.norm(nord, axis=1)[:, None]
    east = np.cross(nord, pijv)
    east = east / np.linalg.norm(east, axis=1)[:, None]
    assert np.allclose(np.sum(nord*pijv, axis=1), 0, atol=1E-7, rtol=1E-7)
    assert np.allclose(np.sum(east*pijv, axis=1), 0, atol=1E-7, rtol=1E-7)

    piv = p[:, i-1, 1:]
    ei = piv - (np.sum(pijv*piv, axis=1) / p2)[:, None]*pijv
    ei = ei / np.linalg.norm(ei, axis=1)[:, None]
    assert np.allclose(np.sum(ei*pijv, axis=1), 0, atol=1E-7, rtol=1E-7)

    cos = np.sum(ei*nord, axis=1)
    sin = np.sum(ei*east, axis=1)
    phi = np.arctan2(sin, cos)
    return phi

def calculate_cs_phis(p, CS, cast=False):
    '''Given momenta calculate cos(2phi) and sin(2phi) for all gluon pairs/quark pairs with relevant prefactors.'''
    alpha_s = 0.118
    C_A = 3
    T_R = 0.5
    gluon_prefactor = 16*np.pi*alpha_s*C_A
    quark_prefactor = 8*np.pi*alpha_s*T_R
    gluon_combs = list(combinations(CS.gluons, 2))
    # assume quarks-antiquark is coming in pairs
    quark_combs = [CS.quarks[i:i+2] for i in range(0, len(CS.quarks), 2)]
    # don't need initial state gluon/quark pair
    if (1, 2) in gluon_combs:
        gluon_combs.remove((1, 2))
    if [1, 2] in quark_combs:
        quark_combs.remove([1, 2])
    
    cos_2phis = []
    sin_2phis = []
    for comb in gluon_combs:
        i = comb[0]
        j = comb[1]
        phis = calc_phi(p, i, j)
        sij = utility_functions.mdot(p[:, i-1], p[:, j-1])
    
        cos_2phi = gluon_prefactor * np.cos(2*phis) / sij
        sin_2phi = gluon_prefactor * np.sin(2*phis) / sij
        
        cos_2phis.append(cos_2phi)
        sin_2phis.append(sin_2phi)
        
    for comb in quark_combs:
        i = comb[0]
        j = comb[1]
        phis = calc_phi(p, i, j)
        sij = utility_functions.mdot(p[:, i-1], p[:, j-1])
    
        cos_2phi = quark_prefactor * np.cos(2*phis) / sij
        sin_2phi = quark_prefactor * np.sin(2*phis) / sij
        
        cos_2phis.append(cos_2phi)
        sin_2phis.append(sin_2phi)

    cos_2phis = tf.transpose(tf.convert_to_tensor(cos_2phis))
    sin_2phis = tf.transpose(tf.convert_to_tensor(sin_2phis))

    if cast:
        cos_2phis = tf.cast(cos_2phis, tf.float32)
        sin_2phis = tf.cast(sin_2phis, tf.float32)

    return cos_2phis, sin_2phis

class ModelInputsGenerator():
    '''
    Class to compute inputs to neural network.
    '''
    
    def __init__(self, permutations, CS):
        self.permutations = permutations
        self.CS = CS
        
    def calculate_Ds(self, p):
        '''Calculate Catani-Seymour dipoles.'''
        self.CS.set_momenta(p)
        D = []
        for perm in self.permutations:
            D.append(self.CS.calculate_D(*perm))
        return tf.convert_to_tensor(D)
    
    def calculate_recoil_factors(self, p):
        '''Calculate Catani-Seymour recoil factors.'''
        self.CS.set_momenta(p)
        r = []
        for perm in self.permutations:
            rf = self.CS.calculate_RF(*perm)
            # this step makes the distributions of all recoils look more similar
            i, j, _ = perm
            if self.CS.mode == "FI":
                if self.CS.massive_quarks:
                    top = self.CS.massive_quarks[0]
                    antitop = self.CS.massive_quarks[1]
                    if (i == top and j == antitop) or (i == antitop and j == top):
                        r.append(rf)
                        continue
                r.append(1 - rf)
            elif self.CS.mode in ["IF", "II"]:
                r.append(1- rf)
            elif self.CS.mode == "FF":
                r.append(rf)
        return tf.convert_to_tensor(r)

    def calculate_mandelstam_invariants(self, p):
        s = list(pspoint.PSpoint(p).sij.values())
        return tf.convert_to_tensor(s)
    
    @tf.function
    def calculate_inputs(self, p_array, to_concat=None):
        '''Vectorise input computations.'''
        dipoles = tf.vectorized_map(self.calculate_Ds, p_array)
        if to_concat:
            dipoles = tf.concat([dipoles, *to_concat], axis=1)
        rfs = tf.vectorized_map(self.calculate_recoil_factors, p_array)
        sijs = tf.vectorized_map(self.calculate_mandelstam_invariants, p_array)
        return dipoles, rfs, sijs
