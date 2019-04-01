import numpy as np
from numpy import linalg as LA
from scipy.spatial.transform import Rotation as R

from utils.curvature import *

def get_symmetry_vec(mesh, curvatures, p1, p2):
    '''
    Given two points v1,v2 on a mesh, compute the transformation T that maps v1 to v2.
    Input:
    - mesh: a triangular mesh
    - p1, p2: indices of vertices on the mesh, integer scalars
    Output:
    - T: (omega,u,s), a symmetry vector in 7-dimensional transformation space, of shape (7,)
    '''
    v1 = mesh.vertices[p1]; curs1 = curvatures[p1];
    v2 = mesh.vertices[p2]; curs2 = curvatures[p2];
    T = np.zeros(7)
    
    # T[:3] is omega
    r1_axis = np.cross(curs1.norm_dir, curs2.norm_dir); r1_axsi = r1_axis / LA.norm(r1_axis)
    r1 = R.from_rotvec(r1_axis * np.arccos(np.dot(curs1.norm_dir, curs2.norm_dir))) # rotation aligning the normal directions
    r2 = R.from_rotvec(curs2.norm_dir) # rotation aligning the two principle directions in the tangent space
    T[:3] = (r2 * r1).as_euler('xyz')
    
    # T[6] is s
    T[6] = np.sum(curs1.prin_curvatures / curs2.prin_curvatures) / 2
    
    # T[3:6] is u
    T[3:6] = v2 - T[6] * (r2 * r1).apply(v1)
    return T

def log_map(T):
    '''
    Logarithm map acting on the transformation space SIM(3).
    Input:
    - T: a symmetry vector in 7-dimensional transformation space, of shape (7,)
    Output:
    - logT: logarithm of T, of shape (7,)
    '''
    logT = np.zeros(7)
    return logT

def adjoint_invar_norm(logT, alpha=1, beta=1, gamma=1):
    '''
    Adjoint invariant norm of a 7-dimensional symmetry vector
    Input:
    - logT: logarithm of a symmetry vector in 7-dimensional transformation space, of shape (7,)
    - alpha: importance ratio of translation, non-negative scalar
    - beta: importance ratio of rotation, non-negative scalar
    - gamma: importance ration of scaling, non-negative scalar
    Output:
    - norm: ||logT||
    '''
    eps = 1e-6
    omega = logT[:3]; u = logT[3:6]; s = logT[6]
    omega_bar = omega / (LA.norm(omega) + eps)
    return alpha * np.sum(omega * omega)\
            + beta * ((1 - LA.norm(omega_bar)) * LA.norm(u) + LA.norm(omega_bar) * np.dot(omega_bar, u)) ** 2 \
            + gamma * s**2

def symmetry_dist(T1, T2, alpha=1, beta=1, gamma=1):
    '''
    Compute the distance between two symmetry vectors.
    Input:
    - T1, T2: symmetry vectors in 7-dimensional transformation space, of shape (7,)
    - alpha: importance ratio of translation, non-negative scalar
    - beta: importance ratio of rotation, non-negative scalar
    - gamma: importance ration of scaling, non-negative scalar
    Output:
    - dist: distance between T1 and T2, non-negative scalar
    '''
    return adjoint_invar_norm(np.dot(T1, LA.inv(T2)))
    