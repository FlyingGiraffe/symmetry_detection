import numpy as np
from numpy import linalg as LA
import trimesh

class Curvature:
    def __init__(self):
        self.second_fund_form = np.zeros((3, 3)) # second fundamental form
        self.prin_curvatures = np.zeros(2) # principle curvatures
        self.prin_dirs = np.zeros((3, 2)) # principle directions
        self.norm_dir = np.zeros(3) # normal direction
        self.bary_area = 0 # barycentric area

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
    hinges = trimesh.graph.face_adjacency(mesh.faces, mesh, return_edges=True) # [adjacent faces; shared edge]
    for h in range(hinges[0].shape[0]):
        e = mesh.vertices[hinges[1][h,0]] - mesh.vertices[hinges[1][h,1]]
        e_bar = e / LA.norm(e)
        n1 = mesh.face_normals[hinges[0][h,0]]; n2 = mesh.face_normals[hinges[0][h,1]]
        beta = 2 * np.pi - mesh.face_adjacency_angles[h]
        if mesh.face_adjacency_convex[h] == False:
            beta = -beta
        curvatures[hinges[1][h,0]].second_fund_form += LA.norm(e) * np.dot(e_bar, e_bar.transpose()) / 2
        curvatures[hinges[1][h,1]].second_fund_form += LA.norm(e) * np.dot(e_bar, e_bar.transpose()) / 2
    for f in range(nt):
        curvatures[mesh.faces[f,0]].bary_area += mesh.area_faces[f] / 3
        curvatures[mesh.faces[f,1]].bary_area += mesh.area_faces[f] / 3
        curvatures[mesh.faces[f,2]].bary_area += mesh.area_faces[f] / 3
    for p in range(nv):
        curvatures[p].second_fund_form /= curvatures[p].bary_area
    
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
