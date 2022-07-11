"""Base classes for principles objects.
Should not be used, excepted to create inherited classes.
"""
from copy import deepcopy
from fedoo.core.modelingspace import ModelingSpace


#=============================================================
# Base class for Mesh object
#=============================================================
class MeshBase:
    """Base class for Mesh object."""

    __dic = {}

    def __init__(self, name = ""):
        assert isinstance(name, str) , "name must be a string" 
        self.__name = name
        
        if name != "":
            MeshBase.__dic[self.__name] = self

    def __class_getitem__(cls, item):
        return cls.__dic[item]

    @property
    def name(self):
        return self.__name

    @staticmethod
    def get_all():
        return MeshBase.__dic
    

#=============================================================
# Base class for Assembly object
#=============================================================
class AssemblyBase:
    """Base class for Assembly object."""

    __dic = {}

    def __init__(self, name = "", space=None):
        assert isinstance(name, str) , "An name must be a string" 
        self.__name = name

        self.global_matrix = None
        self.global_vector = None
        self.mesh = None 
        
        if name != "": AssemblyBase.__dic[self.__name] = self
        self.__space = space

    def __class_getitem__(cls, item):
        return cls.__dic[item]

    def get_global_matrix(self):
        if self.global_matrix is None: self.assemble_global_mat()        
        return self.global_matrix

    def get_global_vector(self):
        if self.global_vector is None: self.assemble_global_mat()        
        return self.global_vector
        
    def assemble_global_mat(self):
        #needs to be defined in inherited classes
        pass

    def delete_global_mat(self):
        """
        Delete Global Matrix and Global Vector related to the assembly. 
        This method allow to force a new assembly
        """
        self.global_matrix = None
        self.global_vector = None
    
    
    @staticmethod
    def get_all():
        return AssemblyBase.__dic
    
    # @staticmethod
    # def Launch(name):
    #     """
    #     Assemble the global matrix and global vector of the assembly name
    #     name is a str
    #     """
    #     AssemblyBase.get_all()[name].assemble_global_mat()    
    
    @property
    def space(self):
        return self.__space
   
    @property
    def name(self):
        return self.__name


#=============================================================
# Base class for constitutive laws (cf constitutive law lib)
#=============================================================
class ConstitutiveLaw:
    """Base class for constitutive laws (cf constitutive law lib)."""

    __dic = {}

    def __init__(self, name = ""):
        assert isinstance(name, str) , "An name must be a string" 
        self.__name = name
        self.__localFrame = None
        self._dimension = None #str or None to specify a space and associated model (for instance "2Dstress" for plane stress)

        ConstitutiveLaw.__dic[self.__name] = self        


    def __class_getitem__(cls, item):
        return cls.__dic[item]


    def SetLocalFrame(self, localFrame):
        self.__localFrame = localFrame


    def GetLocalFrame(self):
        return self.__localFrame 

    
    def reset(self): 
        #function called to restart a problem (reset all internal variables)
        pass

    
    def set_start(self):  
        #function called when the time is increased. Not used for elastic laws
        pass

    
    def to_start(self):
        #function called if the time step is reinitialized. Not used for elastic laws
        pass


    def initialize(self, assembly, pb, t0 = 0., nlgeom=False):
        #function called to initialize the constutive law 
        pass

    
    def update(self,assembly, pb, dtime):
        #function called to update the state of constitutive law 
        pass

    
    def copy(self, new_id = ""):
        """
        Return a raw copy of the constitutive law without keeping current internal variables.

        Parameters
        ----------
        new_id : TYPE, optional
            The name of the created constitutive law. The default is "".

        Returns
        -------
        The copy of the constitutive law
        """
        new_cl = deepcopy(self)        
        new_cl._ConstitutiveLaw__name = new_id
        self.__dic[new_id] = new_cl
        new_cl.reset()
        return new_cl

    
    @staticmethod
    def get_all():
        return ConstitutiveLaw.__dic
    
    
    @property
    def name(self):
        return self.__name


#=============================================================
# Base class for weakforms (cf weakforms lib)
#=============================================================
class WeakForm:
    """Base class for weakforms (cf weakforms lib)."""

    __dic = {}

    def __init__(self, name = "", space=None):
        assert isinstance(name, str) , "An name must be a string" 
        self.__name = name
        if space is None: 
            space = ModelingSpace.get_active()
        elif isinstance(space, str):
            space = ModelingSpace[space]
        self.__space = space
        self.assembly_options = {}
        #possible options : 
        # * 'assume_sym' - self.assembly_options['assume_sym'] = True  to accelerate assembly if the weak form may be considered as symmetric
        # * 'n_elm_gp' - set the default n_elm_gp
        # * 'mat_lumping' - matrix lumping if set to True
        
        if name != "":WeakForm.__dic[self.__name] = self
        
    
    def __class_getitem__(cls, item):
        return cls.__dic[item]

    
    def GetConstitutiveLaw(self):
        #no constitutive law by default
        pass
    
    
    def GetDifferentialOperator(self, mesh=None, localFrame = None):
        pass
            
    
    def initialize(self, assembly, pb, t0=0.):
        #function called at the very begining of the resolution
        pass


    def set_start(self, assembly, pb, dt):
        #function called at the begining of a new time increment
        #For now, used only to inform the weak form the the time step for the next increment.
        pass
    

    def update(self, assembly, pb, dtime):
        #function called when the problem is updated (NR loop or time increment)
        #- New initial Stress
        #- New initial Displacement
        #- Possible modification of the mesh
        #- Change in constitutive law (internal variable)
        pass
    
    
    def to_start(self):
        #function called if the time step is reinitialized. Used to reset variables to the begining of the step
        pass
    

    def reset(self):
        #function called if all the problem history is reseted.
        pass     
      
    
    def copy(self):
        #function to copy a weakform at the initial state
        raise NotImplementedError()
      
        
    @staticmethod
    def nvar(self):
        return self.__space.nvar


    @staticmethod
    def get_all():
        return WeakForm.__dic


    @property
    def space(self):
        return self.__space
    
    
    @property
    def name(self):
        return self.__name
    



