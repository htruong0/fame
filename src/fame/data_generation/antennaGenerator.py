import tensorflow as tf
from fame.utilities.pspoint import PSpoint

class AntennaGenerator():
    '''
    Class to compute inputs to neural network.
    '''
    
    def __init__(self, permutations, antenna, mapper, cast=False):
        self.permutations = permutations
        self._antenna = antenna
        self._mapper = mapper
        self.cast = cast
        
    def calculate_map_variables(self, p):
        '''Calculate r and rho for momenta mapping.'''
        self._mapper.set_momenta(p)
        rs = []
        rhos = []
        for perm in self.permutations:
            r, rho = self._mapper.calculate_map_variables(*perm)
            rs.append(r)
            rhos.append(rho)
        return tf.concat([tf.convert_to_tensor(rs), tf.convert_to_tensor(rhos)], axis=0)
        
    def calculate_Xs_loop(self, p):
        '''Calculate antennae.'''
        self._antenna.set_momenta(p)
        X = []
        for perm in self.permutations:
            X.append(self._antenna.X_3_1(*perm))
        return tf.concat(X, axis=0)
        
    def calculate_Xs_born(self, p):
        '''Calculate antennae.'''
        self._antenna.set_momenta(p)
        X = []
        for perm in self.permutations:
            X.append(self._antenna.X_3_0(*perm))
        return tf.concat(X, axis=0)
    
    @tf.function
    def calculate_mapping(self, p_array):
        mapping_variables = tf.vectorized_map(self.calculate_map_variables, p_array)
        return mapping_variables
    
    @tf.function
    def calculate_inputs(self, p_array, mode="born", to_concat=None):
        '''Vectorise input computations.'''
        if mode == "born":
            antennae = tf.vectorized_map(self.calculate_Xs_born, p_array)
        elif mode == "loop":
            antennae = tf.vectorized_map(self.calculate_Xs_loop, p_array)
        if self.cast:
            antennae = tf.cast(antennae, tf.float32)
        if to_concat:
            antennae = tf.concat([antennae, *to_concat], axis=1)
        return antennae

class Mapper():        
    def calculate_map_variables(self, i, j, k):
        self.set_indices(i, j, k)
        r = self.s_jk / (self.s_ij + self.s_jk)
        rho = tf.math.sqrt(1 + 4*r*(1-r)*self.s_ij*self.s_jk/(self.s_ijk*self.s_ik))
        return r, rho
        
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
        
    def set_momenta(self, p):
        """
        Set momenta of choice.
        """
        self.p = p
        self.num_jets = self.p.shape[0]-2
        self.w = p[0][0] + p[1][0]
        # quicker to calculate all s_ij here as we can re-use them for all permutations
        self.s = PSpoint(p, self.num_jets, self.w, 0).sij
        
    def _get_pair_string(self, i, j):
        return "".join(map(str, sorted([i, j])))