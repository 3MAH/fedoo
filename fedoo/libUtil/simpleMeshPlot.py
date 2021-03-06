# -*- coding: utf-8 -*-
"""
Created on Thu May  7 15:22:31 2020

@author: Etienne
"""
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

import numpy as np
from fedoo.libMesh import Mesh
from fedoo.libAssembly import Assembly
from fedoo.libConstitutiveLaw import ConstitutiveLaw



def meshPlot2d(mesh, disp=None, data=None, data_min=None,data_max=None, scale_factor = 1, plot_edge = True, nb_level = 6):
    """
    Simple function for ploting 2D mesh with node data and node displacement
    
    Parameters
    ----------
    mesh : TYPE
        DESCRIPTION.
    disp : TYPE
        DESCRIPTION.
    data : TYPE
        DESCRIPTION.
    scale_factor : TYPE, optional
        DESCRIPTION. The default is 1.
    plot_edge : TYPE, optional
        DESCRIPTION. The default is False.
    nb_level : TYPE, optional
        DESCRIPTION. The default is 6.

    Returns
    -------
    None.

    """
    # Create triangulation.
    color = plt.cm.hsv

    if isinstance(mesh, str): mesh = Mesh.GetAll()[mesh]

    crd = mesh.GetNodeCoordinates()
    elm = mesh.GetElementTable()
    type_el = mesh.GetElementShape()

    if disp is not None: 
    #Get the displacement vector on nodes for export to vtk
        U = np.reshape(disp,(2,-1)).T    
        N = mesh.GetNumberOfNodes()
        U = np.c_[U,np.zeros(N)]    
        x = crd[:,0] + U[:,0]*scale_factor
        y = crd[:,1] + U[:,1]*scale_factor
    else: 
        x = crd[:,0]
        y = crd[:,1] 
    
    if type_el in ['tri3', 'tri6']:
        triang = mtri.Triangulation(x, y, elm[:,0:3])
    else:
        triang = mtri.Triangulation(x, y, np.vstack((elm[:,0:3],elm[:, [0,2,3]])))
    
    fig = plt.figure()
    ax = fig.gca()
    
    if data is not None:
        if data_min is None and data_max is None: 
            plt.tricontourf(triang, data, nb_level, cmap=color)
        else:
            if data_min is None: data_min = data.min()
            if data_max is None: data_max = data.max()
            plt.tricontourf(triang, data, vmax=data_max, levels = np.linspace(data_min,data_max,nb_level),extend = 'both', cmap=color)        
        
    if plot_edge == True:
        if type_el in ['tri3', 'tri6']:
            plt.triplot(triang, lw=0.5, color='k') 
        else:#quad
            crd_scaled = np.c_[x,y] #crd[:,0:2] + U[:,0:2]*50           
            Nnde_elm = len(elm[0]) 
            patches = []
            
            for i in range(len(elm)):        
                polygon = Polygon(crd_scaled[elm[i,0:4],0:2], True)
                patches.append(polygon)                      
            
            p = PatchCollection(patches, edgecolors='k', fc='None', lw =0.5, alpha=1)            
            ax.add_collection(p)            
            ax.set_xlim(crd_scaled[:,0].min(), crd_scaled[:,0].max())
            ax.set_ylim(crd_scaled[:,1].min(), crd_scaled[:,1].max())           
    
                
    ax.set_aspect('equal')
    # ax.set_title('Stress')
    # ax = fig.gca()
    
    # fig, ax = plt.subplots()
    if data is not None:
        plt.colorbar(orientation = "horizontal")
        # plt.clim(data_min, data_max)
    plt.xlabel('x')
    plt.ylabel('y')
    # plt.ion()
    
    
def fieldPlot2d(mesh, MatID, disp, dataID=None, component=0, data_min=None,data_max=None, scale_factor = 1, plot_edge = True, nb_level = 6, type_plot = "real"):

    if isinstance(mesh, str): mesh = Mesh.GetAll()[mesh]

    #type_plot is "real" or "smooth"
    if dataID is None: # no data, just plot mesh
        meshPlot2d(mesh, disp, None, None, None, scale_factor, plot_edge, nb_level)
        return    
    elif dataID.lower() == 'disp': #if data is disp, no difference between "real" and "smooth" plot
        try:
            meshPlot2d(mesh, disp, disp.reshape(2,-1)[component], data_min,data_max, scale_factor, plot_edge, nb_level)
        except: 
            raise NameError('DataID: '+ str(dataID)+ ' and component: '+ str(component)+" doesn't exist")    
        
        plt.gca().set_title(dataID+'_'+str(component))
        return
    
    if isinstance(mesh,str): mesh = Mesh.GetAll()[mesh]
    crd = mesh.GetNodeCoordinates()
    elm = mesh.GetElementTable()
    type_el = mesh.GetElementShape()
    
    if type_plot.lower() == "smooth":
        mesh2 = Mesh(crd, elm, type_el, ID='visu')
        crd2=crd ; elm2=elm
        U = disp
    elif type_plot.lower() == "real":
        crd2 = crd[elm.ravel()]
        elm2 = np.arange(elm.shape[0]*elm.shape[1]).reshape(-1,elm.shape[1])
        mesh2 = Mesh(crd2, elm2, type_el, ID='visu')         
        U = ((disp.reshape(2,-1).T[elm.ravel()]).T).ravel()
    
    #reload the assembly with the new mesh
    Assembly(None, 'visu', type_el, ID="visu", MeshChange = True) 
    Assembly.GetAll()['visu'].PreComputeElementaryOperators(mesh2, type_el)

    #compute tensorstrain and tensorstress
    TensorStrain = Assembly.GetAll()['visu'].GetStrainTensor(U, "Nodal", nlgeom = False)       
    TensorStress = ConstitutiveLaw.GetAll()[MatID].GetStress(TensorStrain)
    
    try:
        if dataID.lower() == 'stress':
            if isinstance(component,str) and component.lower()=='vm':
                data=TensorStress.vonMises()
            else: 
                data=TensorStress[component]            
        elif dataID.lower() == 'strain':
            data = TensorStrain[component]
    except: 
        raise NameError('DataID: '+ str(dataID)+ ' and component: '+ str(component)+" doesn't exist")
    
    meshPlot2d('visu', U, data, data_min, data_max, scale_factor, plot_edge, nb_level)
    
    # # Create triangulation.
    # color = plt.cm.hsv

    # U = np.reshape(U,(2,-1)).T
    # N = mesh2.GetNumberOfNodes()
    # U = np.c_[U,np.zeros(N)]
        
    # x = crd2[:,0] + U[:,0]*scale_factor
    # y = crd2[:,1] + U[:,1]*scale_factor
    # triang = mtri.Triangulation(x, y, elm2[:,0:3])
     


    # #plot the new mesh
    # # Set up the figure
    # fig = plt.figure()
    
    # # color = plt.cm.get_cmap('hsv',256)
    # color = plt.cm.hsv
    # nb_level = 10
    # plt.tricontourf(triang, data, nb_level, cmap=color)
    # if plot_edge == True:
    #     plt.triplot(triang, lw=0.2, color='k')
    # ax = fig.gca()
    # ax.set_aspect('equal')
    # ax.set_title('Triangular grid')
    # # ax = fig.gca()
    
    # # fig, ax = plt.subplots()
    # plt.colorbar(orientation = 'horizontal')
    # plt.xlabel('x')
    # plt.ylabel('y')
    plt.gca().set_title(dataID+'_'+str(component))






    
   





    
   

