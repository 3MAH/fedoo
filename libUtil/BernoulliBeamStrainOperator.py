from fedoo.libUtil.Operator  import OpDiff
from fedoo.libUtil.Dimension import ProblemDimension

def GetBernoulliBeamStrainOperator():
    n = ProblemDimension.Get()

    epsX = OpDiff('DispX',  'X', 1) # dérivée en repère locale
    xsiZ = OpDiff('DispY',  'X', 2) # flexion autour de Z

    if n == "2Dplane":
        eps = [epsX, 0, 0, 0, 0, xsiZ]

    elif n == "2Dstress":
        assert 0, "no 2Dstress for a beam kinematic"

    elif n == "3D":
        xsiX = OpDiff('ThetaX', 'X', 1) # torsion autour de X
        xsiY = -OpDiff('DispZ',  'X', 2) # flexion autour de Y
        eps = [epsX, 0, 0, xsiX, xsiY, xsiZ]
        
    eps_vir = [e.virtual() if e != 0 else 0 for e in eps ]
        
    return eps, eps_vir # todo, put it in a tuple

if __name__=="__main__":
    Dimension("3D")
    A,B = GetBernoulliBeamStrainOperator()

