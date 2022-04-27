import numpy as np
import tensorflow as tf
from itertools import combinations

from fame.utilities import utility_functions
from fame.data_generation import relevantDipoles

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
    pijv = p[:, i+1, 1:] + p[:, j+1, 1:]
    p2 = np.sum(pijv*pijv, axis=1)
    l = p2 / pijv[:, 2]
    ez = np.zeros((len(p), 3))
    ez[:, 2] = l
    nord = ez - pijv
    nord = nord / np.linalg.norm(nord, axis=1)[:, None]
    east = np.cross(nord, pijv)
    east = east / np.linalg.norm(east, axis=1)[:, None]
    assert np.allclose(np.sum(nord*pijv, axis=1), 0)
    assert np.allclose(np.sum(east*pijv, axis=1), 0)

    piv = p[:, i+1, 1:]
    ei = piv - (np.sum(pijv*piv, axis=1) / p2)[:, None]*pijv
    ei = ei / np.linalg.norm(ei, axis=1)[:, None]
    assert np.allclose(np.sum(ei*pijv, axis=1), 0)

    cos = np.sum(ei*nord, axis=1)
    sin = np.sum(ei*east, axis=1)
    phi = np.arctan2(sin, cos)
    return phi

def calculate_cs_phis(p, num_jets, mode='gluon', cast=False):
    '''Given momenta calculate cos(2phi) and sin(2phi) for all gluon pairs with relevant prefactors.'''
    alpha_s = 0.118
    if mode == 'gluon':
        C_A = 3
        prefactor = 16*np.pi*alpha_s*C_A
        combs = list(combinations(range(3, num_jets+1), 2))
    elif mode == 'quark':
        T_R = 0.5
        prefactor = 8*np.pi*alpha_s*T_R
        # combinations for non-identical flavoured quarks in final state
#         combs = [(1, 2), (1, 4), (2, 3), (3, 4)]
        # combinations for identical flavoured quarks in final state
        combs = [(1, 2), (3, 4)]
    
    cos_2phis = []
    sin_2phis = []
    for comb in combs:
        i = comb[0]
        j = comb[1]
        phis = calc_phi(p, i, j)
        sij = 2*utility_functions.dot(p[:, i+1], p[:, j+1])
    
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
    
    def __init__(self, permutations, CS, cast):
        self.permutations = permutations
        self.CS = CS
        self.cast = cast
        
    def calculate_Ds(self, p):
        '''Calculate Catani-Seymour dipoles.'''
        self.CS.set_momenta(p)
        D = []
        for perm in self.permutations:
            D.append(self.CS.calculate_D_ijk(*perm))
        return tf.convert_to_tensor(D)
    
    def calculate_ys(self, p):
        '''Calculate Catani-Seymour recoil factors.'''
        self.CS.set_momenta(p)
        y = []
        for perm in self.permutations:
            self.CS.set_indices(*perm)
            y.append(self.CS.calculate_y())
        return tf.convert_to_tensor(y)
    
    @tf.function
    def calculate_inputs(self, p_array, to_concat=None):
        '''Vectorise input computations.'''
        dipoles = tf.vectorized_map(self.calculate_Ds, p_array)
        ys = tf.vectorized_map(self.calculate_ys, p_array)
        if self.cast:
            dipoles = tf.cast(dipoles, tf.float32)
            ys = tf.cast(ys, tf.float32)
        if to_concat:
            dipoles = tf.concat([dipoles, *to_concat], axis=1)
        return dipoles, ys
