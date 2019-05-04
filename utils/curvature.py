import numpy as np
from numpy import linalg as LA
import trimesh

class Curvature:
    def __init__(self):
        self.second_fund_form = np.zeros((3, 3)) # second fundamental form
        self.prin_curvatures = np.zeros(2) # principle curvatures
        self.prin_dirs = np.zeros((3, 2)) # principle directions
        self.bary_area = 0 # barycentric area
        self.valid = True

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
    # For convenience, collect matrices of vertices (nv x 3) and triangles (nt x 3)
    X = mesh.vertices # each row is the position of a vertex
    T = mesh.faces # rows are (i,j,k) indices of triangle vertices
    nv = X.shape[0] # number of vertices
    nt = T.shape[0] # number of triangles
    vnorm = mesh.vertex_normals
    
    eps = 1e-6
    
    for t in range(nt):
        vertex1 = X[T[t,0],:]; vertex2 = X[T[t,1],:]; vertex3 = X[T[t,2],:];
        edge12 = vertex2-vertex1;
        edge23 = vertex3-vertex2;
        edge31 = vertex1-vertex3;
        # vertex 1
        t12 = np.dot(np.eye(3) - vnorm[T[t,0],:,None] * vnorm[T[t,0],:], edge12.T); t12 /= LA.norm(t12) + eps;
        t13 = np.dot(np.eye(3) - vnorm[T[t,0],:,None] * vnorm[T[t,0],:], -edge31.T); t13 /= LA.norm(t13) + eps;
        k12 = 2 * np.dot(vnorm[T[t,0],:], edge12) / sum(edge12 * edge12);
        k13 = 2 * np.dot(vnorm[T[t,0],:], -edge31) / sum(edge31 * edge31);
        # vertex 2
        t21 = np.dot(np.eye(3) - vnorm[T[t,1],:,None] * vnorm[T[t,1],:], -edge12.T); t21 /= LA.norm(t21) + eps;
        t23 = np.dot(np.eye(3) - vnorm[T[t,1],:,None] * vnorm[T[t,1],:], edge23.T); t23 /= LA.norm(t23) + eps;
        k21 = 2 * np.dot(vnorm[T[t,1],:], -edge12) / sum(edge12 * edge12);
        k23 = 2 * np.dot(vnorm[T[t,1],:], edge23) / sum(edge23 * edge23);
        # vertex 3
        t31 = np.dot(np.eye(3) - vnorm[T[t,2],:,None] * vnorm[T[t,2],:], edge31.T); t31 /= LA.norm(t31) + eps;
        t32 = np.dot(np.eye(3) - vnorm[T[t,2],:,None] * vnorm[T[t,2],:], -edge23.T); t32 /= LA.norm(t32) + eps;
        k31 = 2 * np.dot(vnorm[T[t,2],:], edge31) / sum(edge31 * edge31);
        k32 = 2 * np.dot(vnorm[T[t,2],:], -edge23) / sum(edge23 * edge23);
        # area of triangle t = weight on edge uv in this triangle
        A = mesh.area_faces[t]
        curvatures[mesh.faces[t,0]].second_fund_form += A * k12 * (t12[:,None] * t12) + A * k13 * (t13[:,None] * t13);
        curvatures[mesh.faces[t,1]].second_fund_form += A * k21 * (t21[:,None] * t21) + A * k23 * (t23[:,None] * t23);
        curvatures[mesh.faces[t,2]].second_fund_form += A * k31 * (t31[:,None] * t31) + A * k32 * (t32[:,None] * t32);
        # the weight w should be normalized to sum to 1 at last
        curvatures[mesh.faces[t,0]].bary_area += A * 2;
        curvatures[mesh.faces[t,1]].bary_area += A * 2;
        curvatures[mesh.faces[t,2]].bary_area += A * 2;
    
    for v in range(nv):
        curvatures[v].second_fund_form /= curvatures[v].bary_area
    
    '''
    nv = mesh.vertices.shape[0] # number of vertices
    nt = mesh.faces.shape[0] # number of triangles
    hinges = trimesh.graph.face_adjacency(mesh.faces, mesh, return_edges=True) # [adjacent faces; shared edge]
    for h in range(hinges[0].shape[0]):
        e = mesh.vertices[hinges[1][h,0]] - mesh.vertices[hinges[1][h,1]]
        e_bar = e / LA.norm(e)
        n1 = mesh.face_normals[hinges[0][h,0]]; n2 = mesh.face_normals[hinges[0][h,1]]
        beta = 2 * np.pi - mesh.face_adjacency_angles[h]
        #if mesh.face_adjacency_convex[h] == False:
            #beta = -beta
        #curvatures[hinges[1][h,0]].second_fund_form += beta * LA.norm(e) * (e_bar * e_bar[:,np.newaxis]) / 2
        #curvatures[hinges[1][h,1]].second_fund_form += beta * LA.norm(e) * (e_bar * e_bar[:,np.newaxis]) / 2
        curvatures[hinges[1][h,0]].second_fund_form += beta * (e_bar * e_bar[:,np.newaxis]) / 2
        curvatures[hinges[1][h,1]].second_fund_form += beta * (e_bar * e_bar[:,np.newaxis]) / 2
    for f in range(nt):
        curvatures[mesh.faces[f,0]].bary_area += mesh.area_faces[f] / 3
        curvatures[mesh.faces[f,1]].bary_area += mesh.area_faces[f] / 3
        curvatures[mesh.faces[f,2]].bary_area += mesh.area_faces[f] / 3
    #for p in range(nv):
        #curvatures[p].second_fund_form /= curvatures[p].bary_area/mesh.area
   '''
    
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
    eps = 1e-6
    nv = mesh.vertices.shape[0] # number of vertices
    nt = mesh.faces.shape[0] # number of triangles
    for p in range(nv):
        eigvals, eigvects = LA.eigh(curvatures[p].second_fund_form)
        zero_idx = np.argmin(np.absolute(eigvals))
        nonzeros = np.arange(3)[np.arange(3) != zero_idx]
        lam = np.sort(eigvals[nonzeros])
        curvatures[p].prin_curvatures[0] = 3 * lam[0] - lam[1]
        curvatures[p].prin_curvatures[1] = 3 * lam[1] - lam[0]
        curvatures[p].prin_dirs = eigvects[:, nonzeros]
        
        min_abs = min(abs(curvatures[p].prin_curvatures[0]), abs(curvatures[p].prin_curvatures[1]))
        max_abs = max(abs(curvatures[p].prin_curvatures[0]), abs(curvatures[p].prin_curvatures[1]))
        if abs(min_abs / (max_abs + eps)) > 0.9 or max_abs < eps:
            curvatures[p].valid = False
        
def signature(mesh, curvatures, p):
    '''
    Compute the signature of a point in the signature space.
    Input:
    - mesh: a triangular mesh
    - curvatures: an array of Curvature object storing per-vertex second fundamental form, of shape (nv,) 
    - p: index of the vertex on the mesh, integer
    Output:
    - sig: signature of mesh.vertices[p], real number
    '''
    eps = 1e-6
    return curvatures[p].prin_curvatures[0] / (curvatures[p].prin_curvatures[1] + eps)