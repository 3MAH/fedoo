from fedoo.libUtil.Operator  import *
from fedoo.libUtil.Dimension import *

def GetDispOperator():
    n = ProblemDimension.Get()

    U=[]
    U.append(OpDiff('DispX'))
    U.append(OpDiff('DispY'))

    if n == "3D":
        U.append(OpDiff('DispZ'))

    U_vir = [e.virtual() if e != 0 else 0 for e in U ]
        
    return U, U_vir # todo, put it in a tuple

if __name__=="__main__":
    Dimension("3D")
    A,B = GetDispOperator()
