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

def textured_mesh(mesh, per_vertex_signal=None, filename='visualize mesh'):
    if per_vertex_signal == None:
        per_vertex_signal = np.zeros(mesh.vertices.shape[0])
    x = mesh.vertices.transpose()[0]; y = mesh.vertices.transpose()[1]; z = mesh.vertices.transpose()[2];
    xrange = max(x)-min(x); yrange = max(y)-min(y); zrange = max(z)-min(z)
    data1 = plotly_trisurf(x, y, z, per_vertex_signal, mesh.faces, colormap=cm.RdBu,plot_edges=True)
    axis = dict(showbackground=True,backgroundcolor="rgb(230, 230,230)",gridcolor="rgb(255, 255, 255)",zerolinecolor="rgb(255, 255, 255)")
    layout = Layout(width=800, height=800,scene=Scene(xaxis=XAxis(axis),yaxis=YAxis(axis),zaxis=ZAxis(axis),
                      aspectratio=dict(x=xrange,y=yrange,z=zrange)))
    fig1 = Figure(data=data1, layout=layout)
    iplot(fig1, filename=filename)