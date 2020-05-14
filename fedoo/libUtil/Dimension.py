from fedoo.libUtil.Coordinate import *

class ProblemDimension:
    __dimension = "" # attribut statique de la classe

    def __init__(self,value):
        assert isinstance(value,str) , "The dimension value must be a string"

        assert value=="3D" or value=="2Dplane" or value=="2Dstress", "Dimension must be '3D', '2Dplane' or '2Dstress'"

        ProblemDimension.__dimension = value

        Coordinate('X') 
        Coordinate('Y')
                
        if value == "3D":
            ProblemDimension.__DoF = 3
            Coordinate('Z')
        if value == "2Dplane" or value == "2Dstress":
            ProblemDimension.__DoF = 2
            
        print("\nDimension of the problem is now set on "+ value)

    @staticmethod
    def Get():
        assert ProblemDimension.__dimension != "" , "You must define a dimension for your problem"
        return ProblemDimension.__dimension
    
    @staticmethod
    def GetDoF():
        assert ProblemDimension.__dimension != "" , "You must define a dimension for your problem"
        return ProblemDimension.__DoF
        
if __name__=="__main__":
    ProblemDimension("3D")
    print(ProblemDimension.Get())
    ProblemDimension("2Dplane")
    print(ProblemDimension.Get())
    ProblemDimension(3)
    ProblemDimension("ee")
