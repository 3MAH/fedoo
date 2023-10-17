from fedoo.core.diffop  import DiffOp


class ModelingSpace:
    #__ModelingSpace = {"Main"}      
    _active = None
    __dic = {} #dic containing all the modeling spaces

    def __init__(self, dimension, name ="Main"):
        """
        Space in which are defined the coordinates and variables (ie functions over coordinates).        
        A modeling space is required to be able to define weak equations.
                    
        Parameters
        ----------
        dimension: str in ['2D', '3D', '2Dplane' or '2Dstress']
            Type of modeling space. 
            * '2D' or '2Dplane': general 2D problems, with default coordinates 'X' and 'Y' and plane strain assumption
            * '3D' for 3D problems with default coordinates 'X', 'Y' and 'Z'
            * '2Dstress': same as '2D' but using plane stress assumption for mechanical problems
        name: str, default = 'Main'
            The name of the modeling space
            
        Example
        --------
        Create a new empty 2D modeling space with plane stress assumption
        >>> import fedoo as fd             
        >>> fd.ModelingSpace('2Dstress')   
        >>> print(fd.ModelingSpace['Main'].ndim)
        """
        
        
        # assert name not in ModelingSpace.__dic, str(name) + " already exist. Delete it first."
        assert isinstance(dimension,str) , "The dimension value must be a string: '2D', '3D', '2Dplane' or '2Dstress'."
        if dimension == "2D": dimension = "2Dplane"
        assert dimension=="3D" or dimension=="2Dplane" or dimension=="2Dstress", "Dimension must be '2D', '3D', 2Dplane' or '2Dstress'"        
        
        #Static attributs 
        ModelingSpace._active = self
        ModelingSpace.__dic[name] = self
        
        self.__name = name
        
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
       

    def __class_getitem__(cls, item):
        return cls.__dic[item]
      
   
    def get_dimension(self):
        """
        Return an str that define the dimension of the ModelingSpace:
        * '2Dplane': general 2D problems, with default coordinates 'X' and 'Y' and plane strain assumption.
        * '3D' for 3D problems with default coordinates 'X', 'Y' and 'Z'.
        * '2Dstress': same as '2D' but using plane stress assumption for mechanical problems.
        """
        return self._dimension

    
    def make_active(self):
        """Define the modeling space as the active ModelingSpace."""
        ModelingSpace._active = self

    
    @staticmethod
    def set_active(name):
        """
        Static method.
        Define the active ModelingSpace from its name.
        """
        if isinstance(name, ModelingSpace): ModelingSpace._active = name
        elif name in ModelingSpace.__dic: ModelingSpace._active = ModelingSpace.__dic[name]
        else: raise NameError("{} is not a valid ModelingSpace".format(name))

    
    @staticmethod    
    def get_active():
        """Return the active ModelingSpace."""
        assert ModelingSpace._active is not None, "You must define a ModelingSpace for your problem."
        return ModelingSpace._active

    
    @staticmethod
    def get_all():
        """Return a dict containing all the defined ModelingSpace"""
        return ModelingSpace.__dic

            
    @property
    def ndim(self):
        """Return the number of dimensions."""
        return self.__ndim


    @property
    def name(self):
        """Return the name of the ModelingSpace."""
        return self.__name


    #===================================================
    #Methods related to Coordinates
    #===================================================
    def new_coordinate(self,name): 
        """Create a new coordinate with the given name."""        
        assert isinstance(name,str) , "The coordinte name must be a string"

        if name not in self._coordinate.keys():
            self._coordinate[name] = self._ncrd
            self._ncrd +=1

    
    def coordinate_rank(self,name): 
        """Return the rank (int id) of the coordinate name"""
        if name not in self._coordinate.keys():
            assert 0, "the coordinate name " + str(name) + " does not exist" 
        return self._coordinate[name]

    
    def coordinate_name(self,rank):
        """
        Return the rank (int) of a coordinate associated to a given name (str)
        """
        return list(self._coordinate.keys())[list(self._variable.values()).index(rank)]


    def list_coordinates(self):
        """return a list containing all the coordinates name"""
        return self._coordinate.keys()  


    @property
    def ncrd(self):
        """Return the number of coordinates defined in the ModelingSpace"""
        return self._ncrd      


    #===================================================
    #Methods related to Varibale
    #===================================================
    def new_variable(self, name):
        """Create a new variable with the given name. """
        assert isinstance(name,str) , "The variable must be a string"
        assert name[:2] != '__', "Names of variable should not begin by '__'"
        
        if name not in self._variable.keys():
            self._variable[name] = self._nvar
            self._nvar +=1
        
        # return DiffOp(self._variable[name])


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
        """Return the vector (list of ndim variable ranks) associated with the given name."""
        return self._vector[name]            


    @property
    def nvar(self):
        """Return the number of variables defined in the ModelingSpace"""
        return self._nvar


    def list_variables(self):
        """return a list containing all the variables name"""
        return self._variable.keys()

    
    def list_vectors(self):
        """return a list containing all the vectors name"""
        return self._vector.keys()
    
    
    def derivative(self, u, x=0, ordre=1, decentrement=0, vir=0):
        """Return a simple DiffOp containing only a derivative."""
        if isinstance(u,str):
            u = self.variable_rank(u)
        if isinstance(x,str):
            x = self.coordinate_rank(x)
        return DiffOp(u, x, ordre, decentrement, vir)
    
    
    def variable(self, u):
        """Return a simple DiffOp containing only the given variable."""
        if isinstance(u,str): u = self.variable_rank(u)
        return DiffOp(u)
        
        
    
    #===================================================
    #build usefull list of operators 
    #===================================================
    def op_grad_u(self):        
       if self.ndim == 3:        
           return [[self.derivative(namevar, namecoord) for namecoord in ['X','Y','Z']] for namevar in ['DispX','DispY','DispZ']]
       else:
           return [[self.derivative(namevar, namecoord) for namecoord in ['X','Y']] + [0] for namevar in ['DispX','DispY']] + [[0,0,0]]
       
    def op_div_u(self):
        if self.ndim == 3:
            return self.derivative('DispX', 'X') + self.derivative('DispY', 'Y') + self.derivative('DispZ', 'Z')
        else:
            return self.derivative('DispX', 'X') + self.derivative('DispY', 'Y')
    
    def op_strain(self, InitialGradDisp = None):
        # InitialGradDisp = StrainOperator.__InitialGradDisp

        if (InitialGradDisp is None) or (InitialGradDisp is 0):
            du_dx = self.derivative('DispX', 'X')
            dv_dy = self.derivative('DispY', 'Y')
            du_dy = self.derivative('DispX', 'Y')
            dv_dx = self.derivative('DispY', 'X')
        
            if self.ndim == 2:
                eps = [du_dx, dv_dy, 0, du_dy+dv_dx, 0, 0]
        
            else: #assume ndim == 3
                dw_dz = self.derivative('DispZ', 'Z')
                du_dz = self.derivative('DispX', 'Z')
                dv_dz = self.derivative('DispY', 'Z')
                dw_dx = self.derivative('DispZ', 'X')
                dw_dy = self.derivative('DispZ', 'Y')
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
        epsX = self.derivative('DispX',  'X') # dérivée en repère locale
        xsiZ = self.derivative('RotZ',  'X') # flexion autour de Z
        gammaY = self.derivative('DispY', 'X') - self.variable('RotZ') #shear/Y
        
        if self.__ndim == 2:
            eps = [epsX, gammaY, 0, 0, 0, xsiZ]

        else: #assume ndim == 3
            xsiX = self.derivative('RotX', 'X') # torsion autour de X
            xsiY = self.derivative('RotY',  'X') # flexion autour de Y
            gammaZ = self.derivative('DispZ', 'X') + self.variable('RotY') #shear/Z
        
            eps = [epsX, gammaY, gammaZ, xsiX, xsiY, xsiZ]
                        
        return eps
    
    
    def op_disp(self):
        if self.__ndim == 2:
            return [self.variable('DispX'), self.variable('DispY')]
        else:
            return [self.variable('DispX'), self.variable('DispY'), self.variable('DispZ')]
            
        

    # def op_beam_strain_bernoulli(self):
    #     epsX = self.derivative('DispX',  'X') # dérivée en repère locale
    #     xsiZ = self.derivative('RotZ',  'X') # flexion autour de Z

    #     if self.__ndim == 2:
    #         eps = [epsX, 0, 0, 0, 0, xsiZ]

    #     else: #assume ndim == 3
    #         xsiX = self.derivative('RotX', 'X') # torsion autour de X
    #         xsiY = self.derivative('RotY',  'X') # flexion autour de Y
    #         eps = [epsX, 0, 0, xsiX, xsiY, xsiZ]
            
    #     # eps_vir = [e.virtual if e != 0 else 0 for e in eps ]
            
    #     return eps 

        
if __name__=="__main__":
    space = ModelingSpace("3D", name = "my space")
    print(ModelingSpace.get_active().name)
    another_space = ModelingSpace("2Dplane")
    print(ModelingSpace.get_all())
    print(ModelingSpace.get_active().ndim)
    print(ModelingSpace['my space'].ndim)

