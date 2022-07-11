"""Base classes for principles objects.
Should not be used, excepted to create inherited classes.
"""
from copy import deepcopy

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
# simple class to update several constitutive laws at once
#=============================================================
class ListConstitutiveLaw(ConstitutiveLaw):
    """Simple class to update several constitutive laws at once."""
    
    def __init__(self, list_constitutivelaw, name =""):    
        ConstitutiveLaw.__init__(self,name)   
        
        self.__list_constitutivelaw = set(list_constitutivelaw) #remove duplicated cl
    
    
    def initialize(self, assembly, pb, t0=0., nlgeom=False):
        for cl in self.__list_constitutivelaw:
            cl.initialize(assembly, pb, t0)

    
    def update(self, assembly, pb, dtime):        
        for cl in self.__list_constitutivelaw:
            cl.update(assembly, pb, dtime)
    
    
    def set_start(self):  
        for cl in self.__list_constitutivelaw:
            cl.set_start()
    
    
    def to_start(self):
        for cl in self.__list_constitutivelaw:
            cl.to_start()


    def reset(self):
        for cl in self.__list_constitutivelaw:
            cl.reset()
    
    
    def copy(self):
        #function to copy a weakform at the initial state
        raise NotImplementedError()


