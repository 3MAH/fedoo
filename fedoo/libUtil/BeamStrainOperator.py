from fedoo.libUtil.Operator  import OpDiff
from fedoo.libUtil.ModelingSpace import ModelingSpace


from fedoo.libUtil.Operator  import OpDiff
def GetBeamStrainOperator():
    n = ModelingSpace.GetDimension()

    epsX = OpDiff('DispX',  'X', 1) # dérivée en repère locale
    xsiZ = OpDiff('RotZ',  'X', 1) # flexion autour de Z
    gammaY = OpDiff('DispY', 'X', 1) - OpDiff('RotZ') #shear/Y
    
    if n == "2Dplane":
        eps = [epsX, gammaY, 0, 0, 0, xsiZ]

    elif n == "2Dstress":
        assert 0, "no 2Dstress for a beam kinematic"

    elif n == "3D":
        xsiX = OpDiff('RotX', 'X', 1) # torsion autour de X
        xsiY = OpDiff('RotY',  'X', 1) # flexion autour de Y
        gammaZ = OpDiff('DispZ', 'X', 1) + OpDiff('RotY') #shear/Z
    
        eps = [epsX, gammaY, gammaZ, xsiX, xsiY, xsiZ]
        
    eps_vir = [e.virtual() if e != 0 else 0 for e in eps ]
        
    return eps, eps_vir 

def GetBernoulliBeamStrainOperator():
    n = ModelingSpace.GetDimension()

    epsX = OpDiff('DispX',  'X', 1) # dérivée en repère locale
    xsiZ = OpDiff('RotZ',  'X', 1) # flexion autour de Z

    if n == "2Dplane":
        eps = [epsX, 0, 0, 0, 0, xsiZ]

    elif n == "2Dstress":
        assert 0, "no 2Dstress for a beam kinematic, use '2Dplane' instead"

    elif n == "3D":
        xsiX = OpDiff('RotX', 'X', 1) # torsion autour de X
        xsiY = OpDiff('RotY',  'X', 1) # flexion autour de Y
        eps = [epsX, 0, 0, xsiX, xsiY, xsiZ]
        
    eps_vir = [e.virtual() if e != 0 else 0 for e in eps ]
        
    return eps, eps_vir 


