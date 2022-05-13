import numpy as np
import tensorflow as tf
from itertools import combinations

from fame_pp.utilities import utility_functions, pspoint
from fame_pp.data_generation import relevantDipoles, cs_dipole

CS = cs_dipole.CS_dipole()

def get_relevant_permutations(num_jets):
    perms = relevantDipoles.getPermutations(num_jets)
    perms = [perm for perm in perms if perm[1] not in [1, 2]]
    return perms

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
    assert np.allclose(np.sum(nord*pijv, axis=1), 0)
    assert np.allclose(np.sum(east*pijv, axis=1), 0)

    piv = p[:, i-1, 1:]
    ei = piv - (np.sum(pijv*piv, axis=1) / p2)[:, None]*pijv
    ei = ei / np.linalg.norm(ei, axis=1)[:, None]
    assert np.allclose(np.sum(ei*pijv, axis=1), 0)

    cos = np.sum(ei*nord, axis=1)
    sin = np.sum(ei*east, axis=1)
    phi = np.arctan2(sin, cos)
    return phi

def calculate_cs_phis(p, cast=False):
    '''Given momenta calculate cos(2phi) and sin(2phi) for all gluon pairs with relevant prefactors.'''
    alpha_s = 0.118
    C_A = 3
    prefactor = 16*np.pi*alpha_s*C_A
    # skip 2 initial state gluon splitting
    combs = list(combinations(CS.gluons, 2))[1:]
    
    cos_2phis = []
    sin_2phis = []
    for comb in combs:
        i = comb[0]
        j = comb[1]
        phis = calc_phi(p, i, j)
        sij = 2*utility_functions.dot(p[:, i-1], p[:, j-1])
    
        cos_2phi = prefactor * np.cos(2*phis) / sij
        sin_2phi = prefactor * np.sin(2*phis) / sij
        
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
            self.CS.set_indices(*perm)
            self.CS.calculate_invariants()
            rf = self.CS.calculate_recoil_factor()
            # this step makes the x recoil input more like the FF y_ijk
            if self.CS.mode in ["FI", "IF", "II"]:
                rf = rf / (1-rf)
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
