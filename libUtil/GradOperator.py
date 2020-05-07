from fedoo.libUtil.Operator  import OpDiff
from fedoo.libUtil.Dimension import ProblemDimension


def GetGradOperator():
    if ProblemDimension.Get() == "3D":        
        GradOperator = [[OpDiff(IDvar, IDcoord,1) for IDcoord in ['X','Y','Z']] for IDvar in ['DispX','DispY','DispZ']]    
    else:
        GradOperator = [[OpDiff(IDvar, IDcoord,1) for IDcoord in ['X','Y']] + [0] for IDvar in ['DispX','DispY']]       
        GradOperator += [[0,0,0]]
            
    return GradOperator

if __name__=="__main__":
    ProblemDimension("3D")
    A,B = GetGradOperator()

