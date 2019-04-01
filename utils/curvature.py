import numpy as np
from numpy import linalg as LA

class Curvature:
    def __init__(self):
        self.second_fund_form = np.zeros((3, 3))
        self.prin_curvatures = np.zeros(2)
        self.prin_dirs = np.zeros((3, 2))
        self.norm_dir = np.zeros(3)

def get_curvatures(mesh):
    curvatures = [Curvature() for p in range(mesh.vertices.shape[0])]
    get_second_fund_form(mesh, curvatures)
    get_prin_curvatures(mesh, curvatures)
    return curvatures
        
def get_second_fund_form(mesh, curvatures):
    '''
    Compute a per-vertex ingegrated fundamental form of the triangle mesh.
    Reference: Anisotropic polygonal remeshing. P. Alliez, D. Cohen-Steiner, O. Devillers, B. Levy, and M. Desbrum, 2003.
    Input:
    - mesh: a triangular mesh
    Output:
    - curvatures[p].second_fund_form (p index of mesh vertex): second fundamental form of each vertex, of shape (3, 3)
    '''
    nv = mesh.vertices.shape[0] # number of vertices
    nt = mesh.faces.shape[0] # number of triangles
    for p in range(nv):
        curvatures[p].second_fund_form = np.eye(3)
        curvatures[p].second_fund_form[2,2] = 0
    
def get_prin_curvatures(mesh, curvatures):
    '''
    Compute the principle curvatures, principle directions and normal direction at each vertex.
    Input:
    - mesh: a triangular mesh
    - curvatures[p].second_fund_form (p index of mesh vertex): second fundamental form of each vertex, of shape (3, 3)
    Output:
    - curvatures[p].prin_curvatures: principle curvatures at each vertex, of shape (2,)
    - curvatures[p].prin_dirs: principle directions at each vertex, of shape (3, 2)
    - curvatures[p].norm_dirs: normal direction at each vertex, of shape (3,)
    '''
    nv = mesh.vertices.shape[0] # number of vertices
    nt = mesh.faces.shape[0] # number of triangles
    for p in range(nv):
        eigvals, eigvects = LA.eigh(curvatures[p].second_fund_form)
        zero_idx = np.argmin(np.absolute(eigvals))
        curvatures[p].norm_dir = eigvects[:, zero_idx]
        nonzeros = np.arange(3)[np.arange(3) != zero_idx]
        curvatures[p].prin_curvatures = eigvals[nonzeros]
        curvatures[p].prin_dirs = eigvects[:, nonzeros]
