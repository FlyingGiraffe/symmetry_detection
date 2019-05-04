import numpy as np
from numpy import linalg as LA
from scipy.spatial.transform import Rotation as R

from utils.curvature import *

def get_symmetry_vect(mesh, curvatures, p1, p2):
    '''
    Given two points v1,v2 on a mesh, compute the transformation T that maps v1 to v2.
    Input:
    - mesh: a triangular mesh
    - curvatures: an array of Curvature object storing per-vertex second fundamental form, of shape (nv,) 
    - p1, p2: indices of vertices on the mesh, integers
    Output:
    - T: (R,t,s), a symmetry vector in 7-dimensional transformation space, of shape (7,)
    '''
    eps = 1e-6
    v1 = mesh.vertices[p1]; curs1 = curvatures[p1]; n1 = mesh.vertex_normals[p1];
    v2 = mesh.vertices[p2]; curs2 = curvatures[p2]; n2 = mesh.vertex_normals[p2];
    T = np.zeros(7)
    
    # T[:3] is R
    r1_axis = np.cross(n1, n2);
    if not (r1_axis==0).all():
        r1_axsi = r1_axis / LA.norm(r1_axis)
    cos = np.dot(n1, n2)
    cos = min(cos, 1); cos = max(cos, -1)
    r1 = R.from_rotvec(r1_axis * np.arccos(cos)) # rotation aligning the normal directions
    cos1 = np.dot(curs1.prin_dirs[:,0], curs2.prin_dirs[:,0])
    cos1 = min(cos1, 1); cos1 = max(cos1, -1)
    cos2 = np.dot(curs1.prin_dirs[:,0], curs2.prin_dirs[:,1])
    cos2 = min(cos2, 1); cos2 = max(cos2, -1)
    r2_angle1 = np.arccos(cos1)
    r2_angle2 = np.arccos(cos2)
    r2 = R.from_rotvec(n2 * min(r2_angle1, r2_angle2)) # rotation aligning the two principle directions in the tangent space
    T[:3] = (r2 * r1).as_euler('xyz')
    
    # T[6] is s
    T[6] = 1#abs(np.sum(curs1.prin_curvatures / (curs2.prin_curvatures + eps))) / 2
    
    # T[3:6] is t
    T[3:6] = v2 - T[6] * (r2 * r1).apply(v1)
    
    return T

def symmetry_test(mesh, curvatures, p1, p2, threshold=1e-6):
    '''
    Given two points v1,v2 on a mesh, test whether there might be a symmetry between them.
    Input:
    - mesh: a triangular mesh
    - curvatures: an array of Curvature object storing per-vertex second fundamental form, of shape (nv,) 
    - p1, p2: indices of vertices on the mesh, integers
    Output:
    - truth value indicating the existence of symmetry between p1 and p2, boolean
    '''
    eps = 1e-6
    #if curvatures[p1].prin_curvatures[0] * curvatures[p2].prin_curvatures[0] < 0 \
        #or curvatures[p1].prin_curvatures[1] * curvatures[p2].prin_curvatures[1] < 0:
        #return False
    #if abs(np.sum(curvatures[p1].prin_curvatures / (curvatures[p2].prin_curvatures + eps)) / 2 - 1) > 1.01:
    #    return False
    if LA.norm(curvatures[p1].prin_curvatures - curvatures[p2].prin_curvatures) > threshold:
        return False
    return True
    #if LA.norm(HKS[p1,:] - HKS[p2,:]) > 0.01:
    #    return False
    #return abs(curvatures[p1].prin_curvatures[0] * curvatures[p2].prin_curvatures[1]
               #- curvatures[p1].prin_curvatures[1] * curvatures[p2].prin_curvatures[0]) < threshold

def log_map(T):
    '''
    Logarithm map acting on the transformation space SIM(3).
    Input:
    - T: (R,t,s), a symmetry vector in 7-dimensional transformation space, of shape (7,)
    Output:
    - logT: (omega,u,lambda) logarithm of T, of shape (7,)
    '''
    eps = 1e-6
    logT = np.zeros(7)
    
    # logT[:3] is omega
    rot_mat = R.from_euler('xyz', T[:3]).as_dcm()
    theta = np.arccos((rot_mat.trace() - 1) / 2)
    if np.sin(theta) != 0:
        omega_hat = (rot_mat - rot_mat.transpose()) / (2 * np.sin(theta)) * theta
        logT[0] = omega_hat[2,1]; logT[1] = omega_hat[0,2]; logT[2] = omega_hat[1,0]
    
    # logT[6] is lambda
    logT[6] = np.log(T[6])
    
    # logT[3:6] is u
    a = logT[6]**2 / (logT[6]**2 + theta**2 + eps)
    b = (1/(T[6] + eps) - 1 - logT[6]) / (logT[6] + eps)**2
    c = (1 - np.cos(theta)) / (theta**2 + eps) - logT[6] * (theta - np.sin(theta)) / (theta**3 + eps)
    d = (1 - logT[6] + logT[6]**2 - 1/(T[6] + eps)) / (logT[6] + eps)**2
    e = (theta - np.sin(theta)) / (theta**3 + eps) - logT[6] * (np.cos(theta) - 1 + theta**2/2) / (theta**4 + eps)
    log_rot_mat = R.from_euler('xyz', logT[:3]).as_dcm()
    A = (1 - 1/(T[6] + eps)) / (logT[6] + eps)
    B = a * (b - logT[6]) + logT[6]
    C = a * (d - e) + e
    V = A * np.eye(3) + B * log_rot_mat + C * log_rot_mat ** 2
    logT[3:6] = np.dot(LA.inv(V), T[3:6])
    
    return logT

def adjoint_invar_norm(logT, alpha=1, beta=1, gamma=10):
    '''
    Adjoint invariant norm of a 7-dimensional symmetry vector.
    Input:
    - logT: logarithm of a symmetry vector in 7-dimensional transformation space, of shape (7,)
    - alpha: importance ratio of translation, non-negative number
    - beta: importance ratio of rotation, non-negative number
    - gamma: importance ration of scaling, non-negative number
    Output:
    - norm: ||logT||
    '''
    eps = 1e-6
    omega = logT[:3]; u = logT[3:6]; s = logT[6]
    omega_bar = omega / (LA.norm(omega) + eps)
    return alpha * np.sum(omega * omega)\
            + beta * ((1 - LA.norm(omega_bar)) * LA.norm(u) + LA.norm(omega_bar) * np.dot(omega_bar, u)) ** 2 \
            + gamma * s**2

def symmetry_dist(T1, T2, alpha=1, beta=1, gamma=10):
    '''
    Compute the distance between two symmetry vectors.
    Input:
    - T1, T2: symmetry vectors in 7-dimensional transformation space, of shape (7,)
    - alpha: importance ratio of translation, non-negative number
    - beta: importance ratio of rotation, non-negative number
    - gamma: importance ration of scaling, non-negative number
    Output:
    - dist: distance between T1 and T2, non-negative number
    '''
    return min(adjoint_invar_norm(log_map(T1) - log_map(T2), alpha, beta, gamma),
               adjoint_invar_norm(log_map(T1) + log_map(T2), alpha, beta, gamma))
    