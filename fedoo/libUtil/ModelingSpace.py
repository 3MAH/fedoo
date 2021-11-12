class ModelingSpace:
    #__ModelingSpace = {"Main"}      
    __activeSpace = None
    __dic = {} #dic containing all the modeling spaces

    def __init__(self, dimension, ID="Main"):
        assert ID not in ModelingSpace.__dic, str(ID) + " already exist. Delete it first."
        assert isinstance(dimension,str) , "The dimension value must be a string"
        assert dimension=="3D" or dimension=="2Dplane" or dimension=="2Dstress", "Dimension must be '3D', '2Dplane' or '2Dstress'"        
        
        #Static attributs 
        ModelingSpace.__activeSpace = self
        ModelingSpace.__dic[ID] = self
        
        self.__ID = ID
        
        #coordinates
        self._coordinate = {} # dic containing all the coordinate related to the modeling space
        self._crd_rank = 0
        
        #variables
        self._variable = {} # attribut statique de la classe
        self._variableDerivative = {} #define the derivative of the Variables
        self._vector = {}
        self._var_rank = 0
        
        #dimension
        if dimension == "2D": dimension = "2Dplane"                
        self._dimension = dimension
        
        ModelingSpace.Coordinate('X') 
        ModelingSpace.Coordinate('Y')
        
        if dimension == "3D":
            self._DoF = 3
            ModelingSpace.Coordinate('Z')
        if dimension == "2Dplane" or dimension == "2Dstress":
            self._DoF = 2            
        


    def GetID(self):
        return self.__ID
    
    def MakeActive(self):
        ModelingSpace.__activeSpace = self
    
    @staticmethod
    def SetActive(ID):
        ModelingSpace.__activeSpace = ModelingSpace.__dic[ID]
    
    @staticmethod    
    def GetActive():
        return ModelingSpace.__activeSpace
    
    @staticmethod
    def GetAll():
        return ModelingSpace.__dic
        
    @staticmethod
    def GetDimension():
        assert ModelingSpace.__activeSpace is not None, "You must define a dimension for your problem"
        return ModelingSpace.__activeSpace._dimension
    
    @staticmethod
    def GetDoF():
        assert ModelingSpace.__activeSpace is not None, "You must define a dimension for your problem"
        return ModelingSpace.__activeSpace._DoF


    #Methods related to Coordinates
    @staticmethod
    def Coordinate(name): 
        """
        Create a new coordinate
        """
        assert isinstance(name,str) , "The coordinte name must be a string"

        self = ModelingSpace.__activeSpace
        if name not in self._coordinate.keys():
            self._coordinate[name] = self._crd_rank
            self._crd_rank +=1
    
    @staticmethod
    def GetCoordinateRank(name): 
        self = ModelingSpace.__activeSpace
        if name not in self._coordinate.keys():
            assert 0, "the coordinate name does not exist" 
        return self._coordinate[name]

    @staticmethod    
    def GetNumberOfCoordinate():
        return ModelingSpace.__activeSpace._crd_rank

    @staticmethod    
    def ListCoordinate():
        return ModelingSpace.__activeSpace._coordinate.keys()          

    #Methods related to Varibale
    @staticmethod
    def Variable(name):
        assert isinstance(name,str) , "The variable must be a string"
        assert name[:2] != '__', "Names of variable should not begin by '__'"
        
        self = ModelingSpace.__activeSpace
        if name not in self._variable.keys():
            self._variable[name] = self._var_rank
            self._var_rank +=1

    @staticmethod
    def GetVariableRank(name):
        """
        Return the rank (int) of a variable associated to a given name (str)
        """
        self = ModelingSpace.__activeSpace
        if name not in self._variable.keys():
            assert 0, "the variable " +str(name)+ " does not exist" 
        return self._variable[name]

    @staticmethod
    def GetVariableName(rank):
        """
        Return the name of the variable associated to a given rank
        """
        self = ModelingSpace.__activeSpace
        return list(self._variable.keys())[list(self._variable.values()).index(rank)]

    @staticmethod
    def Vector(name, listOfVariables):
        """
        Define a vector name from a list Of Variables. 3 variables are required in 3D and 2 variables in 2D.
        In listOfVariales, the first variable is assumed to be associated to the coordinate 'X', the second to 'Y', and the third to 'Z'
        """     
        ModelingSpace.__activeSpace._vector[name] = [ModelingSpace.GetVariableRank(var) for var in listOfVariables]

    @staticmethod
    def GetVector(name):
        return ModelingSpace.__activeSpace._vector[name]            

    @staticmethod
    def GetNumberOfVariable():
        return ModelingSpace.__activeSpace._var_rank

    @staticmethod
    def ListVariable():
        return ModelingSpace.__activeSpace._variable.keys()
  
    @staticmethod
    def ListVector():
        return ModelingSpace.__activeSpace._vector.keys()

# if __name__=="__main__":
#     ProblemDimension("3D")
#     print(ProblemDimension.Get())
#     ProblemDimension("2Dplane")
#     print(ProblemDimension.Get())
#     ProblemDimension(3)
#     ProblemDimension("ee")


def ProblemDimension(value, modelingSpaceID = "Main"):
    """
    Create a new modeling space with the specified problem dimension
    value must be '3D', '2Dplane' or '2Dstress'
    """
    ModelingSpace(value, modelingSpaceID)
    
def Coordinate(name):
    #     """
    #     Create a new coordinate in the active Modeling Space
    #     """
    ModelingSpace.Coordinate(name)    

def Variable(name):
    #     """
    #     Create a new variable in the active Modeling Space
    #     """
    ModelingSpace.Variable(name)    
        
def Vector(name, listOfVariables):
    """
    Define a vector name from a list Of Variables. 3 variables are required in 3D and 2 variables in 2D.
    In listOfVariales, the first variable is assumed to be associated to the coordinate 'X', the second to 'Y', and the third to 'Z'
    """      
    ModelingSpace.Vector(name, listOfVariables)    

def GetNumberOfDimensions():
    return ModelingSpace.GetDoF()

def GetDimension():
    return ModelingSpace.GetDimension()
        
# class Coordinate:
#     __coordinate = {} # attribut statique de la classe
#     __rank = 0

#     def __init__(self,name):
#         assert isinstance(name,str) , "The coordinte name must be a string"

#         if name not in Coordinate.__coordinate.keys():
#             Coordinate.__coordinate[name] = Coordinate.__rank
#             Coordinate.__rank +=1

#     @staticmethod
#     def GetRank(name):
#         if name not in Coordinate.__coordinate.keys():
#             assert 0, "the coordinate name does not exist" 
#         return Coordinate.__coordinate[name]

#     @staticmethod
#     def GetNumberOfCoordinate():
#         return Coordinate.__rank

#     @staticmethod
#     def ListAll():
#         return Coordinate.__coordinate.keys()    

# class Variable:
#     __variable = {} # attribut statique de la classe
#     __variableDerivative = {} #define the derivative of the Variables
#     __vector = {}

#     __rank = 0

#     def __init__(self,name):
#         assert isinstance(name,str) , "The variable must be a string"
#         assert name[:2] != '__', "Names of variable should ne begin by '__'"

#         if name not in Variable.__variable.keys():
#             Variable.__variable[name] = Variable.__rank
#             Variable.__rank +=1

#     @staticmethod
#     def GetRank(name):
#         """
#         Return the rank (int) of a variable associated to a given name (str)
#         """
#         if name not in Variable.__variable.keys():
#             assert 0, "the variable " +str(name)+ " does not exist" 
#         return Variable.__variable[name]

#     @staticmethod
#     def GetName(rank):
#         """
#         Return the name of the variable associated to a given rank
#         """
#         return list(Variable.__variable.keys())[list(Variable.__variable.values()).index(rank)]

#     # @staticmethod
#     # def SetDerivative(name , name_derivative, crd = 'X', sign = 1):
#     #     """
#     #     Define a variable name_derivative as the derivative of the variable name with respect with the coordinate defined in crd (not used for beam).
#     #     This method is used in the context of class C1 elements where a variable related to the derivative has to be defined 
#     #     (for example, angular variables in the bending Bernoulli beam model)
#     #     name is a str who is an existing variable.
#     #     name_derivative is a str. If the variable name_derivative doesn't exist, it is created.        
#     #     """
#     #     #crd is not used for beam, but will be required for plate elements
#     #     if name not in Variable.__variable.keys():
#     #         assert 0, "the variable does not exist" 
#     #     if name_derivative not in Variable.__variable.keys():
#     #         Variable(name_derivative)
#     #     Variable.__variableDerivative[Variable.GetRank(name)] = [Variable.GetRank(name_derivative), sign]

#     @staticmethod
#     def SetVector( name, listOfVariables):
#         """
#         Define a vector name from a list Of Variables. 3 variables are required in 3D and 2 variables in 2D.
#         In listOfVariales, the first variable is assumed to be associated to the coordinate 'X', the second to 'Y', and the third to 'Z'
#         """        
#         Variable.__vector[name] = {'listOfVariables': [Variable.GetRank(var) for var in listOfVariables]} 

#     @staticmethod
#     def GetVector( name ):
#         return Variable.__vector[name]['listOfVariables']

#     # @staticmethod
#     # def GetDerivative(var):
#     #     if isinstance(var, str):
#     #         var = Variable.GetRank(var)
#     #     if var in Variable.__variableDerivative:
#     #         return Variable.__variableDerivative[var]
#     #     else: return None                

#     @staticmethod
#     def GetNumberOfVariable():
#         return Variable.__rank

#     @staticmethod
#     def List():
#         return Variable.__variable.keys()

#     @staticmethod    
#     def ListVector():
#         return Variable.__vector.keys()

# if __name__=="__main__":
#     ProblemDimension("3D")
#     print(ProblemDimension.Get())
#     ProblemDimension("2Dplane")
#     print(ProblemDimension.Get())
#     ProblemDimension(3)
#     ProblemDimension("ee")
