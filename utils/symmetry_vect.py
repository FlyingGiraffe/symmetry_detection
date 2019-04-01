import numpy as np
from numpy import linalg as LA
from scipy.spatial.transform import Rotation as R

from utils.curvature import *

def get_symmetry_vect(mesh, curvatures, p1, p2):
    '''
    Given two points v1,v2 on a mesh, compute the transformation T that maps v1 to v2.
    Input:
    - mesh: a triangular mesh
    - p1, p2: indices of vertices on the mesh, integer scalars
    Output:
    - T: (R,t,s), a symmetry vector in 7-dimensional transformation space, of shape (7,)
    '''
    v1 = mesh.vertices[p1]; curs1 = curvatures[p1];
    v2 = mesh.vertices[p2]; curs2 = curvatures[p2];
    T = np.zeros(7)
    
    # T[:3] is R
    r1_axis = np.cross(curs1.norm_dir, curs2.norm_dir);
    if not (r1_axis==0).all():
        r1_axsi = r1_axis / LA.norm(r1_axis)
    r1 = R.from_rotvec(r1_axis * np.arccos(np.dot(curs1.norm_dir, curs2.norm_dir))) # rotation aligning the normal directions
    r2_angle1 = np.arccos(np.dot(curs1.prin_dirs[:,0], curs2.prin_dirs[:,0]))
    r2_angle2 = np.arccos(np.dot(curs1.prin_dirs[:,0], curs2.prin_dirs[:,1]))
    r2 = R.from_rotvec(curs2.norm_dir * min(r2_angle1, r2_angle2)) # rotation aligning the two principle directions in the tangent space
    T[:3] = (r2 * r1).as_euler('xyz')
    
    # T[6] is s
    T[6] = np.sum(curs1.prin_curvatures / curs2.prin_curvatures) / 2
    
    # T[3:6] is t
    T[3:6] = v2 - T[6] * (r2 * r1).apply(v1)
    
    return T

def log_map(T):
    '''
    Logarithm map acting on the transformation space SIM(3).
    Input:
    - T: (R,t,s), a symmetry vector in 7-dimensional transformation space, of shape (7,)
    Output:
    - logT: (omega,u,lambda) logarithm of T, of shape (7,)
    '''
    logT = np.zeros(7)
    
    # logT[:3] is omega
    rot_mat = R.from_euler('xyz', T[:3]).as_dcm()
    theta = np.arccos((rot_mat.trace() - 1) / 2)
    if np.sin(theta) != 0:
        omega_hat = (rot_mat - rot_mat.transpose()) / np.sin(theta)
        logT[0] = omega_hat[2,1]; logT[1] = omega_hat[0,2]; logT[2] = omega_hat[1,0]
    
    # logT[6] is lambda
    logT[6] = np.log(T[6])
    
    # logT[3:6] is u
    a = logT[6]**2 / (logT[6]**2 + theta**2)
    b = (1/T[6] - 1 - logT[6]) / logT[6]**2
    c = (1 - np.cos(theta)) / theta**2 - logT[6] * (theta - np.sin(theta)) / theta**3
    d = (1 - logT[6] + logT[6]**2 - 1/T[6]) / logT[6]**2
    e = (theta - np.sin(theta)) / theta**3 - logT[6] * (np.cos(theta) - 1 + theta**2/2) / theta**4
    log_rot_mat = R.from_euler('xyz', logT[:3]).as_dcm()
    A = (1 - 1/T[6]) / logT[6]
    B = a * (b - logT[6]) + logT[6]
    C = a * (d - e) + e
    V = A * np.eye(3) + B * log_rot_mat + C * np.dot(log_rot_mat, log_rot_mat)
    logT[3:6] = np.dot(LA.inv(v), T[3:6])
    
    return logT

def adjoint_invar_norm(logT, alpha=1, beta=1, gamma=1):
    '''
    Adjoint invariant norm of a 7-dimensional symmetry vector.
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

def mult_sym_vect(T1, T2):
    '''
    Multiplication of two 7-dimensional symmetry vectors in SIM(3).
    Input:
    - T1, T2: symmetry vectors in 7-dimensional transformation space, of shape (7,)
    Output:
    - mult: multiplication of T1,T2, of shape (7,)
    '''
    mult = np.zeros(7)
    rot_mat1 = R.from_euler('xyz', T1[:3]).as_dcm()
    rot_mat2 = R.from_euler('xyz', T2[:3]).as_dcm()
    mult[:3] = R.from_dcm(np.dot(rot_mat1, rot_mat2)).as_euler('xyz')
    mult[3:6] = np.dot(rot_mat1, T2[3:6]) + T2[6] * T1[3:6]
    mult[6] = T1[6] * T2[6]
    return mult

def inv_sym_vect(T):
    '''
    Inverse of a 7-dimensional symmetry vector in SIM(3).
    Input:
    - T: symmetry vector in 7-dimensional transformation space, of shape (7,)
    Output:
    - inv: inverse of T, of shape (7,)
    '''
    inv = np.zeros(7)
    rot_mat = R.from_euler('xyz', T[:3]).as_dcm()
    inv[:3] = R.from_dcm(rot_mat.transpose()).as_euler('xyz')
    inv[3:6] = - np.dot(rot_mat.transpose(), T[3:6])/ T[6]
    inv[6] = 1 / T[6]
    return inv

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
    return adjoint_invar_norm(mult_sym_vect(T1, inv_sym_vect(T2)))
    