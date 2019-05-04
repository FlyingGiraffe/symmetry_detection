import numpy as np
import trimesh
import plotly
import plotly.plotly as py
from plotly.graph_objs import *
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import matplotlib.cm as cm
from functools import reduce

def map_z2color(zval, colormap, vmin, vmax):
    t=(zval-vmin)/float((vmax-vmin)); R, G, B, alpha=colormap(t)
    return 'rgb('+'{:d}'.format(int(R*255+0.5))+','+'{:d}'.format(int(G*255+0.5))+','+'{:d}'.format(int(B*255+0.5))+')'   

def tri_indices(simplices):
    return ([triplet[c] for triplet in simplices] for c in range(3))

def plotly_trisurf(x, y, z, colors, simplices, colormap=cm.RdBu, plot_edges=False):
    
    points3D=np.vstack((x,y,z)).T
    tri_vertices=map(lambda index: points3D[index], simplices)
    
    ncolors = colors
    if not (colors==0).all():
        ncolors = (colors-np.min(colors))/(np.max(colors)-np.min(colors))
    
    I,J,K=tri_indices(simplices)
    triangles=Mesh3d(x=x,y=y,z=z,
                     intensity=ncolors,
                     colorscale='Viridis',
                     i=I,j=J,k=K,name='',
                     showscale=True,
                     colorbar=ColorBar(tickmode='array', 
                                       tickvals=[np.min(z), np.max(z)], 
                                       ticktext=['{:.3f}'.format(np.min(colors)), 
                                                 '{:.3f}'.format(np.max(colors))]))
    
    if plot_edges is False: # the triangle sides are not plotted
        return Data([triangles])
    else:
        lists_coord=[[[T[k%3][c] for k in range(4)]+[ None] for T in tri_vertices] for c in range(3)]
        Xe, Ye, Ze=[reduce(lambda x,y: x+y, lists_coord[k], lists_coord[0]) for k in range(3)]
        lines=Scatter3d(x=Xe,y=Ye,z=Ze,mode='lines',line=Line(color='rgb(50,50,50)', width=1.5))
        return Data([triangles, lines])

def textured_mesh(mesh, per_vertex_signal=None, samples=None, filename='visualize mesh'):
    if per_vertex_signal is None:
        per_vertex_signal = np.zeros(mesh.vertices.shape[0])
    x = mesh.vertices.transpose()[0]; y = mesh.vertices.transpose()[1]; z = mesh.vertices.transpose()[2];
    ran = mesh.bounds[1,:] - mesh.bounds[0,:]
    data1 = plotly_trisurf(x, y, z, per_vertex_signal, mesh.faces, colormap=cm.RdBu,plot_edges=True)
    if samples is not None:
        X = mesh.vertices[samples]
        ns = mesh.vertex_normals[samples]
        X += ns * 0.001
        data2 = Scatter3d(x=X[:,0], y=X[:,1], z=X[:,2],
                    mode='markers',
                    marker=dict(size=3,
                        line=dict(color='rgba(217, 217, 217, 0.14)', width=0.5),
                        opacity=1))
        data1.append(data2)
    axis = dict(showbackground=True,backgroundcolor="rgb(230, 230,230)",gridcolor="rgb(255, 255, 255)",zerolinecolor="rgb(255, 255, 255)")
    layout = Layout(width=800, height=800,scene=Scene(xaxis=XAxis(axis),yaxis=YAxis(axis),zaxis=ZAxis(axis),
                      aspectratio=dict(x=1,y=1,z=1)),
                   xaxis=dict(range=[mesh.bounds[0,0], mesh.bounds[1,0]]),
                   yaxis=dict(range=[mesh.bounds[0,1], mesh.bounds[1,1]]))
    fig1 = Figure(data=data1, layout=layout)
    iplot(fig1, filename=filename)