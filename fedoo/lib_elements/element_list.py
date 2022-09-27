import numpy as np
from numpy import linalg
from fedoo.lib_elements.beam import *
from fedoo.lib_elements.cohesive import *
from fedoo.lib_elements.hexahedron import *
from fedoo.lib_elements.line import *
from fedoo.lib_elements.plate import *
from fedoo.lib_elements.tetrahedron import *
from fedoo.lib_elements.triangle import *
from fedoo.lib_elements.quadrangle import *
from fedoo.lib_elements.finite_difference_1d import *


def get_element(element_str):
    list_element = {'lin2':Lin2, 'lin3':Lin3, 'lin2bubble':Lin2Bubble, 'lin3bubble':Lin3Bubble,
            'cohesive1d':Cohesive1D, 'cohesive2d':Cohesive2D, 'cohesive3d':Cohesive3D,
            'tri3':Tri3,'tri6':Tri6,'tri3Bubble':Tri3Bubble,
            'quad4':Quad4,'quad8':Quad8,'quad9':Quad9,
            'tet4':Tet4,'tet10':Tet10,'hex8':Hex8,'hex20':Hex20,                        
            'beam':Beam, 'beamfcq':BeamFCQ, 'bernoullibeam':BernoulliBeam,
            'parameter':Parameter, 'node':Node,
            'pquad4':pquad4, 'ptri3':ptri3, 'pquad8':pquad8, 'ptri6':ptri6, 'pquad9':pquad9,
            'bernoullibeam_rot': BernoulliBeam_rot, 'bernoullibeam_disp': BernoulliBeam_disp,
            'beamfcq_lin2':BeamFCQ_lin2, 'beamfcq_rot':BeamFCQ_rot,'beamfcq_disp':BeamFCQ_disp,
            'beam_dispy':Beam_dispY,'beam_dispz':Beam_dispZ,
            'beam_rotz':Beam_rotZ, 'beam_roty':Beam_rotY
    }
    return list_element[element_str.lower()]

def GetNodePositionInElementCoordinates(element, nNd_elm=None):
    #return xi_nd ie the position of nodes in the element local coordinate
    if element in ['lin2', 'lin3', 'lin2Bubble', 'lin3Bubble','cohesive2D']:
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

def get_DefaultNbPG(element, mesh=None, raiseError=True):
    if element in ['cohesive1D']:
        return 1  
    elif element in ['lin2', 'lin2Bubble','cohesive2D']:
        return 2
    elif element in ['lin3', 'lin3Bubble','tri3','tri3Bubble']:
        return 3
    elif element in ['beam', 'beamFCQ', 'bernoulliBeam', 'lin4', 'quad4','tri6','cohesive3D','tet4']:
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
        return get_DefaultNbPG(mesh.elm_type)

