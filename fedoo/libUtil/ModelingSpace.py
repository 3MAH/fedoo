from fedoo.libUtil.Operator  import OpDiff


class ModelingSpace:
    #__ModelingSpace = {"Main"}      
    __active_space = None
    __dic = {} #dic containing all the modeling spaces

    def __init__(self, dimension, ID="Main"):
        # assert ID not in ModelingSpace.__dic, str(ID) + " already exist. Delete it first."
        assert isinstance(dimension,str) , "The dimension value must be a string"
        assert dimension=="3D" or dimension=="2Dplane" or dimension=="2Dstress", "Dimension must be '3D', '2Dplane' or '2Dstress'"        
        
        #Static attributs 
        ModelingSpace.__active_space = self
        ModelingSpace.__dic[ID] = self
        
        self.__ID = ID
        
        #coordinates
        self._coordinate = {} # dic containing all the coordinate related to the modeling space
        self._ncrd = 0
        
        #variables
        self._variable = {} # attribut statique de la classe
        self._nvar = 0
        self._vector = {}
        
        #dimension
        if dimension == "2D": dimension = "2Dplane"                
        self._dimension = dimension
        
        self.new_coordinate('X') 
        self.new_coordinate('Y')
        
        if dimension == "3D":
            self.__ndim = 3
            self.new_coordinate('Z')
        if dimension == "2Dplane" or dimension == "2Dstress":
            self.__ndim = 2                    

    def GetID(self):
        return self.__ID
    
    def MakeActive(self):
        ModelingSpace.__active_space = self
    
    @staticmethod
    def SetActive(ID):
        ModelingSpace.__active_space = ModelingSpace.__dic[ID]
    
    @staticmethod    
    def GetActive():
        assert ModelingSpace.__active_space is not None, "You must define a dimension for your problem"
        return ModelingSpace.__active_space
    
    @staticmethod
    def GetAll():
        return ModelingSpace.__dic
        
    def GetDimension(self):
        return self._dimension
    
    @property
    def ndim(self):
        return self.__ndim


    #Methods related to Coordinates
    def new_coordinate(self,name): 
        """
        Create a new coordinate
        """
        assert isinstance(name,str) , "The coordinte name must be a string"

        if name not in self._coordinate.keys():
            self._coordinate[name] = self._ncrd
            self._ncrd +=1
    
    def coordinate_rank(self,name): 
        if name not in self._coordinate.keys():
            assert 0, "the coordinate name " + str(name) + " does not exist" 
        return self._coordinate[name]
    
    def coordinate_name(self,rank):
        """
        Return the name of the variable associated to a given rank
        """
        return list(self._coordinate.keys())[list(self._variable.values()).index(rank)]

    def list_coordinate(self):
        return self._coordinate.keys()  

    @property
    def ncrd(self):
        return self._ncrd      

    #Methods related to Varibale
    def new_variable(self, name):
        assert isinstance(name,str) , "The variable must be a string"
        assert name[:2] != '__', "Names of variable should not begin by '__'"
        
        self = ModelingSpace.__active_space
        if name not in self._variable.keys():
            self._variable[name] = self._nvar
            self._nvar +=1

    def variable_rank(self,name):
        """
        Return the rank (int) of a variable associated to a given name (str)
        """
        if name not in self._variable.keys():
            assert 0, "the variable " +str(name)+ " does not exist" 
        return self._variable[name]

    def variable_name(self,rank):
        """
        Return the name of the variable associated to a given rank
        """
        return list(self._variable.keys())[list(self._variable.values()).index(rank)]

    def new_vector(self, name, list_variables):
        """
        Define a vector name from a list Of Variables. 3 variables are required in 3D and 2 variables in 2D.
        In listOfVariales, the first variable is assumed to be associated to the coordinate 'X', the second to 'Y', and the third to 'Z'
        """     
        self._vector[name] = [self.variable_rank(var) for var in list_variables]

    def get_vector(self,name):
        return self._vector[name]            

    @property
    def nvar(self):
        return self._nvar

    def list_variable(self):
        return self._variable.keys()
  
    def list_vector(self):
        return self._vector.keys()
    
    def opdiff(self, u, x=0, ordre=0, decentrement=0, vir=0):
        if isinstance(u,str):
            u = self.variable_rank(u)
        if isinstance(x,str):
            x = self.coordinate_rank(x)
        return OpDiff(u, x, ordre, decentrement, vir)
    
    #build usefull list of operators 
    def op_grad_u(self):
       if self.ndim == 3:        
           return [[self.opdiff(IDvar, IDcoord,1) for IDcoord in ['X','Y','Z']] for IDvar in ['DispX','DispY','DispZ']]
       else:
           return [[self.opdiff(IDvar, IDcoord,1) for IDcoord in ['X','Y']] + [0] for IDvar in ['DispX','DispY']] + [[0,0,0]]
       
                   
    def op_strain(self, InitialGradDisp = None):
        # InitialGradDisp = StrainOperator.__InitialGradDisp

        if (InitialGradDisp is None) or (InitialGradDisp is 0):
            du_dx = self.opdiff('DispX', 'X', 1)
            dv_dy = self.opdiff('DispY', 'Y', 1)
            du_dy = self.opdiff('DispX', 'Y', 1)
            dv_dx = self.opdiff('DispY', 'X', 1)
        
            if self.ndim == 2:
                eps = [du_dx, dv_dy, 0, du_dy+dv_dx, 0, 0]
        
            else: #assume ndim == 3
                dw_dz = self.opdiff('DispZ', 'Z', 1)
                du_dz = self.opdiff('DispX', 'Z', 1)
                dv_dz = self.opdiff('DispY', 'Z', 1)
                dw_dx = self.opdiff('DispZ', 'X', 1)
                dw_dy = self.opdiff('DispZ', 'Y', 1)
                eps = [du_dx, dv_dy, dw_dz, du_dy+dv_dx, du_dz+dw_dx, dv_dz+dw_dy]
          
        else:
            GradOperator = self.op_grad_u()
            if self.ndim == 2:
                eps = [GradOperator[i][i] + sum([GradOperator[k][i]*InitialGradDisp[k][i] for k in range(2)]) for i in range(2)] 
                eps += [0]
                eps += [GradOperator[0][1] + GradOperator[1][0] + sum([GradOperator[k][0]*InitialGradDisp[k][1] + GradOperator[k][1]*InitialGradDisp[k][0] for k in range(2)])]  
                eps += [0, 0]
            
            else:
                eps = [GradOperator[i][i] + sum([GradOperator[k][i]*InitialGradDisp[k][i] for k in range(3)]) for i in range(3)] 
                eps += [GradOperator[0][1] + GradOperator[1][0] + sum([GradOperator[k][0]*InitialGradDisp[k][1] + GradOperator[k][1]*InitialGradDisp[k][0] for k in range(3)])]          
                eps += [GradOperator[0][2] + GradOperator[2][0] + sum([GradOperator[k][0]*InitialGradDisp[k][2] + GradOperator[k][2]*InitialGradDisp[k][0] for k in range(3)])]
                eps += [GradOperator[1][2] + GradOperator[2][1] + sum([GradOperator[k][1]*InitialGradDisp[k][2] + GradOperator[k][2]*InitialGradDisp[k][1] for k in range(3)])]
            
        return eps
    
    def op_beam_strain(self):
        epsX = self.opdiff('DispX',  'X', 1) # dérivée en repère locale
        xsiZ = self.opdiff('RotZ',  'X', 1) # flexion autour de Z
        gammaY = self.opdiff('DispY', 'X', 1) - self.opdiff('RotZ') #shear/Y
        
        if self.__ndim == 2:
            eps = [epsX, gammaY, 0, 0, 0, xsiZ]

        else: #assume ndim == 3
            xsiX = self.opdiff('RotX', 'X', 1) # torsion autour de X
            xsiY = self.opdiff('RotY',  'X', 1) # flexion autour de Y
            gammaZ = self.opdiff('DispZ', 'X', 1) + self.opdiff('RotY') #shear/Z
        
            eps = [epsX, gammaY, gammaZ, xsiX, xsiY, xsiZ]
            
        # eps_vir = [e.virtual if e != 0 else 0 for e in eps ]
            
        return eps
    
    def op_disp(self):
        if self.__ndim == 2:
            return [self.opdiff('DispX'), self.opdiff('DispY')]
        else:
            return [self.opdiff('DispX'), self.opdiff('DispY'), self.opdiff('DispZ')]
            
        

    # def op_beam_strain_bernoulli(self):
    #     epsX = self.opdiff('DispX',  'X', 1) # dérivée en repère locale
    #     xsiZ = self.opdiff('RotZ',  'X', 1) # flexion autour de Z

    #     if self.__ndim == 2:
    #         eps = [epsX, 0, 0, 0, 0, xsiZ]

    #     else: #assume ndim == 3
    #         xsiX = self.opdiff('RotX', 'X', 1) # torsion autour de X
    #         xsiY = self.opdiff('RotY',  'X', 1) # flexion autour de Y
    #         eps = [epsX, 0, 0, xsiX, xsiY, xsiZ]
            
    #     # eps_vir = [e.virtual if e != 0 else 0 for e in eps ]
            
    #     return eps 



    
def ProblemDimension(value, modelingSpaceID = "Main"):
    """
    Create a new modeling space with the specified problem dimension
    value must be '3D', '2Dplane' or '2Dstress'
    """
    ModelingSpace(value, modelingSpaceID)
    
# def new_coordinate(name):
#     #     """
#     #     Create a new coordinate in the active Modeling Space
#     #     """
#     ModelingSpace.GetActive().new_coordinate(name)    

# def new_variable(name):
#     #     """
#     #     Create a new variable in the active Modeling Space
#     #     """
#     ModelingSpace.GetActive().new_variable(name)    
        
# def new_vector(name, listOfVariables):
#     """
#     Define a vector name from a list Of Variables. 3 variables are required in 3D and 2 variables in 2D.
#     In listOfVariales, the first variable is assumed to be associated to the coordinate 'X', the second to 'Y', and the third to 'Z'
#     """      
#     ModelingSpace.GetActive().new_vector(name, listOfVariables)    

# def GetNumberOfDimensions():
#     return ModelingSpace.GetDoF()

# def GetDimension():
#     return ModelingSpace.GetDimension()
        

# if __name__=="__main__":
#     ProblemDimension("3D")
#     print(ProblemDimension.Get())
#     ProblemDimension("2Dplane")
#     print(ProblemDimension.Get())
#     ProblemDimension(3)
#     ProblemDimension("ee")
