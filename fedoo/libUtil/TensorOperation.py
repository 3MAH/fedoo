import numpy as np

def PutInPrincipalBase(TensorVoigt):
    """
    S must be a list of 6 terms within the Voigt notation.

    - returns the three terms of the diagonal tensor
    """ 

    assert isinstance(TensorVoigt, list), "S must be a list"
    assert len(TensorVoigt)==6,           "length of S must be 6"

    TensorMatrix = FromVoigtTomatrix(TensorVoigt) # convert the Voigt tensor in a matrix tensor

    temp = np.array([np.linalg.eig(e)[1] for e in TensorMatrix])

    PrincipalDirection = []
    
    PrincipalDirection.append( np.c_[temp[:,:,0], np.zeros(len(temp))] ) # PrincipaleDirection1
    PrincipalDirection.append( np.c_[temp[:,:,1], np.zeros(len(temp))] ) # PrincipaleDirection2
    PrincipalDirection.append( np.c_[temp[:,:,2], np.zeros(len(temp))] ) # PrincipaleDirection3

    return PrincipalDirection

def FromVoigtTomatrix(S):
    """
    - returns a list of 3x3
    """
    
    return [ [[xx,xy,xz],[xy,yy,yz],[xz,yz,zz]] for (xx,yy,zz,yz,xz,xy) in zip(S[0],S[1],S[2],S[3],S[4],S[5]) ]
