from sklearn.manifold import MDS

import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)

import numpy as np

def visualize_dist_2D(dissim, savefig=False, filename=None):
    '''
    Visualize high dimensional data in 2D according to the dissimilarity matrix.
    Input:
    - dissim: pairwise dissimilarities between any two given data, of shape (N,N).
    Output:
    - visualization of the data in 2D
    '''
    # embed the data into 2D
    embedding = MDS(n_components=2, n_jobs=20, dissimilarity='precomputed')
    X = embedding.fit_transform(dissim)
    
    # plot
    trace = go.Scatter(x = X[:,0], y = X[:,1], mode = 'markers')
    data = go.Data([trace])
    # Plot and embed in ipython notebook!
    iplot(data, filename='basic-scatter')
    
def visualize_dist_3D(dissim, savefig=False, filename=None):
    '''
    Visualize high dimensional data in 2D according to the dissimilarity matrix.
    Input:
    - dissim: pairwise dissimilarities between any two given data, of shape (N,N).
    Output:
    - visualization of the data in 2D
    '''
    # embed the data into 2D
    embedding = MDS(n_components=3, n_jobs=20, dissimilarity='precomputed')
    X = embedding.fit_transform(dissim)
    
    # plot
    trace = go.Scatter3d(x=X[:,0], y=X[:,1], z=X[:,2],
                    mode='markers',
                    marker=dict(size=3,
                        color='rgba(217, 217, 217, 0.14)',
                        #line=dict(color='rgba(217, 217, 217, 0.14)', width=0.5),
                        opacity=0.8))
    data = go.Data([trace])
    layout = go.Layout(margin=dict(l=0, r=0, b=0, t=0))
    fig = go.Figure(data=data, layout=layout)
    iplot(fig, filename='simple-3d-scatter')

def visualize_point_2D(X, axis=[0,1], labeled=None, savefig=False, filename=None):
    '''
    Visualize high dimensional data in 2D according by projection.
    Input:
    - X: a list of data points, of shape (N,D).
    Output:
    - visualization of the data in 2D
    '''
    trace = go.Scatter(x = X[:,axis[0]], y = X[:,axis[1]],
                 mode = 'markers',
                 marker=dict(size=2))
    layout = go.Layout(width=600, height=600)
    if labeled is None:
        data = go.Data([trace])
    else:
        trace1 = go.Scatter(x = labeled[:,axis[0]], y = labeled[:,axis[1]],
                 mode = 'markers',
                 marker=dict(size=8))
        data = go.Data([trace, trace1])
    fig = go.Figure(data=data, layout=layout)
    # Plot and embed in ipython notebook!
    iplot(fig, filename='basic-scatter')

    
def visualize_point_3D(X, axis=[0,1,2], labeled=None, savefig=False, filename=None):
    '''
    Visualize high dimensional data in 3D according by projection.
    Input:
    - X: a list of data points, of shape (N,D).
    Output:
    - visualization of the data in 3D
    '''
    trace = go.Scatter3d(x=X[:,axis[0]], y=X[:,axis[1]], z=X[:,axis[2]],
                    mode='markers',
                    marker=dict(size=2,
                        line=dict(color='rgba(217, 217, 217, 0.14)', width=0.5),
                        opacity=0.8))
    if labeled is None:
        data = go.Data([trace])
    else:
        trace1 = go.Scatter3d(x=labeled[:,axis[0]], y=labeled[:,axis[1]], z=labeled[:,axis[2]],
                    mode='markers',
                    marker=dict(size=5,
                        line=dict(color='rgba(217, 217, 217, 0.14)', width=0.5)))
        data = go.Data([trace, trace1])
    layout = go.Layout(margin=dict(l=0, r=0, b=0, t=0))
    fig = go.Figure(data=data, layout=layout)
    iplot(fig, filename='simple-3d-scatter')
    