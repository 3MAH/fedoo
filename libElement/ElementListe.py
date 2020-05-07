import numpy as np
from numpy import linalg

#lin2, lin3, lin2C1, lin2Bubble, lin3Bubble


def GetNodePositionInElementCoordinates(element, nNd_elm=None):
    #return xi_nd ie the position of nodes in the element local coordinate
    if element in ['lin2', 'lin3', 'lin2C1', 'lin2Bubble', 'lin3Bubble','cohesive2D']:
        if nNd_elm == 2: return np.c_[[0., 1.]]
        elif nNd_elm == 3: return np.c_[[0., 1., 0.5]]
    elif element in ['tri3','tri6','tri3Bubble']:
        if nNd_elm == 3: return np.c_[[0. , 1., 0.],\
                                      [0. , 0., 1.]]            
        elif nNd_elm == 6: return np.c_[[0. , 1., 0., 0.5, 0.5, 0. ],\
                                        [0. , 0., 1., 0. , 0.5, 0.5]]
    elif element in ['quad4','quad8','quad9','cohesive2D']:
        if nNd_elm == 4: return np.c_[[-1. , 1., 1., -1.],\
                                      [-1. , -1., 1., 1.]]
        elif nNd_elm == 8: return np.c_[[-1. , 1. , 1., -1., 0., 1., 0.,-1.],\
                                        [-1. , -1., 1., 1. ,-1., 0., 1., 0.]]
        elif nNd_elm == 9: return np.c_[[-1. , 1. , 1., -1., 0., 1., 0.,-1., 0.],\
                                        [-1. , -1., 1., 1. ,-1., 0., 1., 0., 0.]]
    elif element in ['tet4','tet10']: 
        if nNd_elm == 4: return np.c_[[0. , 0. , 0. , 1.],\
                                      [1. , 0. , 0. , 0.],\
                                      [0. , 1. , 0. , 0.]]
        elif nNd_elm == 10: 
            return np.c_[[0. , 0. , 0. , 1. , 0. , 0. , 0. , 0.5 , 0.5 , 0.5],\
                         [1. , 0. , 0. , 0. , 0.5 , 0. , 0.5 , 0.5 , 0. , 0.],\
                         [0. , 1. , 0. , 0. , 0.5 , 0.5 , 0. , 0. , 0.5 , 0.]]        
    elif element in ['hex8','hex20']:
        if nNd_elm == 8:
            return np.c_[[-1. ,  1. , 1. , -1. , -1.,  1. , 1. ,-1.],\
                               [-1. , -1. , 1. ,  1. , -1., -1. , 1. , 1.],\
                               [-1. , -1. , -1., -1. ,  1.,  1. , 1. , 1.]]
        elif nNd_elm == 20:
            return np.c_[[-1. ,  1. , 1. , -1. , -1.,  1. , 1. ,-1. , 0. ,  1. , 0. , -1. , -1.,  1. , 1. ,-1. , 0.,  1. , 0. ,-1.],\
                               [-1. , -1. , 1. ,  1. , -1., -1. , 1. , 1. , -1. , 0. , 1. ,  0. , -1., -1. , 1. , 1. , -1.,  0. , 1. ,0.],\
                               [-1. , -1. , -1., -1. ,  1.,  1. , 1. , 1. , -1. , -1. , -1., -1. ,  0.,  0. , 0. , 0. , 1.,  1. , 1. ,1.]]            
    elif element in ['cohesive1D']:
        return np.c_[[0., 0.]] #The values are arbitrary, only the size is important

def GetDefaultNbPG(element, mesh=None, raiseError=True):
    if element in ['cohesive1D']:
        return 1  
    elif element in ['lin2', 'lin2Bubble','cohesive2D']:
        return 2
    elif element in ['lin3', 'lin3Bubble','tri3','tri3Bubble']:
        return 3
    elif element in ['lin2C1', 'lin4', 'beam', 'quad4','tri6','cohesive3D','tet4']:
        return 4
    elif element in ['hex8']:
        return 8
    elif element in ['quad8','quad9']:
        return 9
    elif element in ['tet10']: 
        return 15
    elif element in ['hex20']:
        return 27
    elif element in ['parameter', 'node']:
        return 0
    
    if mesh is None and raiseError == True: 
        raise NameError('Element unknown: no default number of integration points')
        
    if mesh is not None: 
        return GetDefaultNbPG(mesh.GetElementShape())
