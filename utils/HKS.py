import numpy as np
from numpy import linalg as LA
from scipy import sparse
from scipy.sparse.linalg import eigs
from scipy.sparse.linalg import eigsh
import trimesh

def get_HKS(mesh, num_eigs=25, num_samples=100):
    L,A = cot_laplacian(mesh)
    eigenvalues, eigenvectors = eigs(L/A, k=num_eigs, sigma=0, which='LM', maxiter=1000)
    
    square_integral = (np.abs(eigenvectors)**2).T * np.diag(A)
    eigenvectors = eigenvectors / (square_integral.T)
    
    tmin = 4 * np.log(10) / eigenvalues[-1]
    tmax = 4 * np.log(10) / eigenvalues[1]
    ts = np.logspace(np.log10(tmin), np.log10(tmax), num_samples)
    
    heatSignature = np.dot(np.abs(eigenvectors)**2, np.exp(-np.abs(eigenvalues)[:,None] * ts)).astype(float)
    heatSignature /= np.sum(A * heatSignature, axis=0)
    
    return heatSignature

def cot_laplacian(mesh):
    '''
    Compute the cotangent weight Laplacian.
    Input:
    - mesh:  a triangular mesh
    Output:
    - W: the symmetric cot Laplacian
    - A: the area weights 
    
    '''
    eps = 1e-6
    X = mesh.vertices
    T = mesh.faces
    nv = X.shape[0]
    nf = T.shape[0]

    # Find orig edge lengths and angles
    L1 = LA.norm(X[T[:,1],:] - X[T[:,2],:], axis=1)
    L2 = LA.norm(X[T[:,0],:] - X[T[:,2],:], axis=1)
    L3 = LA.norm(X[T[:,0],:] - X[T[:,1],:], axis=1)
    EL = np.concatenate((L1[:,None], L2[:,None], L3[:,None]), axis=1)
    A1 = (L2**2 + L3**2 - L1**2) / (2 * L2 * L3)
    A2 = (L1**2 + L3**2 - L2**2) / (2 * L1 * L3)
    A3 = (L1**2 + L2**2 - L3**2) / (2 * L1 * L2)
    A = np.concatenate((A1[:,None], A2[:,None], A3[:,None]), axis=1)
    A = np.arccos(A);

    # The Cot Laplacian
    I = np.concatenate((T[:,0], T[:,1], T[:,2]), axis=0)
    J = np.concatenate((T[:,1], T[:,2], T[:,0]), axis=0)
    S = 0.5 * (np.tan(np.concatenate((A[:,2], A[:,0], A[:,1]), axis=0)) + eps)
    In = np.concatenate((I, J, I, J), axis=0)
    Jn = np.concatenate((J, I, I, J), axis=0)
    Sn = np.concatenate((-S, -S, S, S), axis=0)
    W = sparse.coo_matrix((Sn,(In,Jn)),shape=(nv,nv))

    # Triangle areas
    N = np.cross(X[T[:,0],:] - X[T[:,1],:], X[T[:,0],:] - X[T[:,2],:])
    Ar = 0.5 * LA.norm(N, axis=1)

    # Vertex areas = sum triangles nearby
    I = np.concatenate((T[:,0], T[:,1], T[:,2]), axis=0)
    J = np.zeros(I.shape)
    S = np.concatenate((Ar, Ar, Ar), axis=0) / 3
    A = sparse.coo_matrix((S,(I,J)),shape=(nv,1)).toarray()
    
    return (W, A)
