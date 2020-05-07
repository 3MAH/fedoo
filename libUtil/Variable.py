class Variable:
    __variable = {} # attribut statique de la classe
    __variableDerivative = {} #define the derivative of the Variables
    __vector = {}

    __rank = 0

    def __init__(self,name):
        assert isinstance(name,str) , "The variable must be a string"

        if name not in Variable.__variable.keys():
            Variable.__variable[name] = Variable.__rank
            Variable.__rank +=1

    @staticmethod
    def GetRank(name):
        """
        Return the rank (int) of a variable associated to a given name (str)
        """
        if name not in Variable.__variable.keys():
            assert 0, "the variable does not exist" 
        return Variable.__variable[name]

    @staticmethod
    def GetName(rank):
        """
        Return the name of the variable associated to a given rank
        """
        return list(Variable.__variable.keys())[list(Variable.__variable.values()).index(rank)]

    @staticmethod
    def SetDerivative(name , name_derivative, crd = 'X', sign = 1):
        """
        Define a variable name_derivative as the derivative of the variable name with respect with the coordinate defined in crd (not used for beam).
        This method is used in the context of class C1 elements where a variable related to the derivative has to be defined 
        (for example, angular variables in the bending Bernoulli beam model)
        name is a str who is an existing variable.
        name_derivative is a str. If the variable name_derivative doesn't exist, it is created.        
        """
        #crd is not used for beam, but will be required for plate elements
        if name not in Variable.__variable.keys():
            assert 0, "the variable does not exist" 
        if name_derivative not in Variable.__variable.keys():
            Variable(name_derivative)
        Variable.__variableDerivative[Variable.GetRank(name)] = [Variable.GetRank(name_derivative), sign]

    @staticmethod
    def SetVector( name, listOfVariables, CoordinateSystem = 'local' ):
        """
        Define a vector name from a list Of Variables. 3 variables are required in 3D and 2 variables in 2D.
        In listOfVariales, the first variable is assumed to be associated to the coordinate 'X', the second to 'Y', and the third to 'Z'
        If the coordinate system is 'local' or 'global'. If it is set to 'global', a change of basis is applied during the assembly if the mesh is associated to the physical space (i.e mesh.coordinateId incudes 'X', 'Y', and/or 'Z').
        """        
        Variable.__vector[name] = {'listOfVariables': [Variable.GetRank(var) for var in listOfVariables], 'CoordinateSystem': CoordinateSystem} 

    @staticmethod
    def GetVector( name ):
        return Variable.__vector[name]['listOfVariables']

    @staticmethod    
    def GetVectorCoordinateSystem( name ):
        return Variable.__vector[name]['CoordinateSystem']   

    @staticmethod
    def GetDerivative(var):
        if isinstance(var, str):
            var = Variable.GetRank(var)
        if var in Variable.__variableDerivative:
            return Variable.__variableDerivative[var]
        else: return None                

    @staticmethod
    def GetNumberOfVariable():
        return Variable.__rank

    @staticmethod
    def List():
        return Variable.__variable.keys()

    @staticmethod    
    def ListVector():
        return Variable.__vector.keys()

if __name__=="__main__":
    import sys
    sys.path.append("/home/jeje/fedoo/swampy.1.4")
    from Lumpy import *
    from TurtleWorld import *

    lumpy = Lumpy()
    lumpy.make_reference()
    
    #world = TurtleWorld()
    #bob = Turtle(world)
    v = Variable('VariableName')
    lumpy.object_diagram()
    lumpy.class_diagram()
