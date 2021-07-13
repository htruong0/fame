import numpy as np
from tqdm.auto import tqdm

from fame.utilities import utility_functions
from fame.utilities import pspoint

class RAMBO():
    '''Vectorised RAMBO phase-space generator.'''
    
    def __init__(self, num_jets, w):
        self.num_jets = num_jets
        self.w = w
              
    
    def isotropic_moms(self, num_points):
        '''Create massless 4-mom with isotropic angular distribution.'''
        # first generate random numbers for momenta
        moms = np.random.uniform(0, 1, (num_points, self.num_jets, 4)) 
        ct = 2*moms[:, :, 0] - 1 # cos(theta)
        phi = 2*np.pi*moms[:, :, 1]
        q_0 = -np.log(moms[:, :, 2]*moms[:, :, 3])
        q_1 = q_0*np.sqrt(1-ct**2)*np.cos(phi) # sin(theta) * cos(phi)
        q_2 = q_0*np.sqrt(1-ct**2)*np.sin(phi) # sin(theta) * sin(phi)
        q_3 = q_0*ct
        
        return np.array([q_0.T, q_1.T, q_2.T, q_3.T]).T # (num_points, num_jets, 4) array
    
    
    def generate(self, num_points):
        '''Returns (num_points, num_jets, 4) array of boosted momenta.'''
        q = self.isotropic_moms(num_points)
        Q = np.sum(q, axis=1) # (num_points, 4) array
        M = np.sqrt(utility_functions.dot(Q, Q)).reshape(-1, 1) # (num_points, 1) array
        b = -Q[:, 1:] / M # (num_points, 3) array
        x = self.w / M # (num_points, 1) array
        gamma = (Q / M)[:, 0] # (num_points, 1) array
        a = 1 / (1 + gamma[:, None])
        c = np.einsum('ik,ijk->ij', b, q[:, :, 1:]) # (num_points, 4) array
    
        p_0 = x*(gamma[:, None]*q[:, :, 0] + c)
        # p_0 is a (num_points, num_jets) array with ith entries being the energies of the  nth jets
        p_space = x[:, None]*(q[:, :, 1:] + (b[:, None] * q[:, :, 0].reshape(-1, self.num_jets, 1)) + (b[:, None] * c.reshape(-1, self.num_jets, 1)) * a[:, None])
        # p_space is a (num_points, num_jets, 3) array with ith entries being the 3-momenta of the nth jets
        
        p_1 = np.array([self.w / 2, 0, 0, self.w / 2])
        p_2 = np.array([self.w / 2, 0, 0, -self.w / 2])
        p_inc = np.vstack([p_1, p_2])
        p_inc = np.tile(p_inc, (num_points, 1, 1))
        
        moms = np.dstack((p_0, p_space))
        moms = np.hstack([p_inc, moms])

        return moms


def generate_generic(num_jets, num_points, w, y_global_cut, num_cores):
    '''Generate generic phase-space points using RAMBO, then select points based on cuts.'''
    momenta = []

    pbar = tqdm(total = num_points, desc='Phase-space points')
    ps = RAMBO(num_jets, w)

    # 1% of s_com is lowest it should be
    collinear_cut = max(0.01, 2*y_global_cut)

    # generating PS points
    while len(momenta) < num_points:
        # number of points remaining
        remaining = num_points - len(momenta) 
        # since a lot of points get rejected in clustering, generate more here as it's cheap
        remaining2 = int((3 - 2 / num_points * remaining) * remaining) 
        # phase-space points from RAMBO
        boosted_moms = ps.generate(remaining2)
        # cluster in FastJet with kt algorithm, num_cores should be used for parallel clustering
        idx = utility_functions.clustering(boosted_moms, w, num_jets, collinear_cut, num_cores)
        for i in idx[:remaining]:
            try:
                # check for global phase-space cut
                point = pspoint.PSpoint(boosted_moms[i], num_jets, w, y_global_cut)
            except Exception:
                continue
            momenta.append(point.four_moms)
            pbar.update(1)
    pbar.close()
    print('######## Finished generating generic phase-space points #########')

    return np.array(momenta)
