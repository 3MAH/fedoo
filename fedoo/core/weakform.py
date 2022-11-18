"""This module contains the core objects to create some weakforms"""

from fedoo.core.modelingspace import ModelingSpace
from fedoo.core.base import ConstitutiveLaw
#=======================================================================
# Class to define specific assembly instruction at the weakform level
#=======================================================================
class _AssemblyOptions:
    def __init__(self):
        self.elm_options = {None:{}} #None for default value
    
    def __setitem__(self, item, value):
        #possible options : 
        # * 'assume_sym' - self.assembly_options['assume_sym'] = True  to accelerate assembly if the weak form may be considered as symmetric
        # * 'n_elm_gp' - set the default n_elm_gp
        # * 'mat_lumping' - matrix lumping if set to True
        
        if isinstance(item, tuple):
            assert len(item)  == 2, "item not understoond"
            self.set(item[0], value, item[1])
        else:
            self.set(item, value) #set default value for all element
    
    def __getitem__(self, item):
        if isinstance(item, tuple):                        
            assert len(item)  == 2, "item not understoond"
            return self.get(item[0], item[1])
        else: 
            return self.get(item)
    
    def __repr__(self):
        
        list_str = ["default:"]
        list_str.append(str(self.elm_options[None]))
        
        for elm in self.elm_options:
            if elm is not None:                
                list_str.append(elm + ":")        
                list_str.append(str(self.elm_options[elm]))
        
        return "\n".join(list_str)
    
               
    def get(self, option, elm_type = None, default = None):     
        assert isinstance(option, str) and (isinstance(elm_type, str) or elm_type is None), "option and elm_type should be str"

        if elm_type in self.elm_options and option in self.elm_options[elm_type]:
            return self.elm_options[elm_type][option]
        else: 
            return self.elm_options[None].get(option, default)
        

    def set(self, option, value, elm_type = None): 
        assert isinstance(option, str) and (isinstance(elm_type, str) or elm_type is None), "option and elm_type should be str"
            
        if elm_type not in self.elm_options:
            self.elm_options[elm_type] = {option:value}
        else:     
            self.elm_options[elm_type][option] = value


#=======================================================================
# Base class for all weakform objects
#=======================================================================       
class WeakFormBase:
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
        self.assembly_options = _AssemblyOptions()
        
        if name != "":WeakFormBase.__dic[self.__name] = self
        
    
    def __class_getitem__(cls, item):
        return cls.__dic[item]
    
    def __add__(self, wf):
        if isinstance(wf, WeakFormBase):
            return WeakFormSum([self, wf])
        else: 
            return NotImplemented


    @staticmethod
    def sum(wf1, wf2):
        if isinstance(wf1, WeakFormBase):
            return wf1 + wf2
        else:
            return NotImplemented
        
        
    def GetConstitutiveLaw(self):
        #no constitutive law by default
        pass
    
    
    def get_weak_equation(self, mesh=None):
        return NotImplemented
            
    
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
        """Return the number of variables used in the modeling space associated to the WeakForm."""
        return self.__space.nvar


    @staticmethod
    def get_all():
        """Return the list of all weak forms."""        
        return WeakFormBase.__dic


    @property
    def space(self):
        """Return the ModelingSpace associated to the WeakForm if defined."""        
        return self.__space
    
    
    @property
    def name(self):
        """Return the name of the WeakForm."""
        return self.__name
    
#=======================================================================
# Class WeakForm to set weakform from a weak equation (DiffOp object)
#=======================================================================   
class WeakForm(WeakFormBase):
    
    def __init__(self, weak_equation, name = "", space=None):    
        WeakFormBase.__init__(self, name, space)
        self.weak_equation = weak_equation
    
    def get_weak_equation(self, mesh=None):
        return self.weak_equation
    
    
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
        

#=============================================================
# Class that build a sum of WeakForm
#=============================================================
class WeakFormSum(WeakFormBase):
    
    def __init__(self, list_weakform, name =""):    
        assert len(set([a.space for a in list_weakform])) == 1, \
            "Sum of assembly are possible only if all assembly are associated to the same modeling space"
        WeakFormBase.__init__(self, name, space = list_weakform[0].space)        
        
        if any([wf.assembly_options!={} for wf in list_weakform]):
            self.assembly_options = None
            # if assembly_options is None, the weakForm have to be splited into several sub-weakform before 
            # being used in an Assembly. This is automatically done when using Assembly.create function
            # The restulting Assembly will be an AssemblySum object
            
        self.__constitutivelaw = ListConstitutiveLaw([a.GetConstitutiveLaw() for a in list_weakform if a.GetConstitutiveLaw() is not None])
        self.__list_weakform = list_weakform
        
    def GetConstitutiveLaw(self):
        #return a list of constitutivelaw
        return self.__constitutivelaw    
    
    def get_weak_equation(self, mesh=None):
        Diff = 0
        self._list_mat_lumping = []
        
        if mesh is None: elm_type = None
        else: elm_type = mesh.elm_type
        
        for wf in self.__list_weakform: 
            Diff_wf = wf.get_weak_equation(mesh)
            mat_lumping = wf.assembly_options.get('mat_lumping', elm_type, False) #True of False
            if Diff_wf != 0:
                self._list_mat_lumping.extend([mat_lumping for i in range(len(Diff_wf.op))]) #generate a list of mat_lumping value for each elementary op
            Diff += Diff_wf            
        return Diff
    
    def initialize(self, assembly, pb, t0=0.):
        for wf in self.__list_weakform:
            wf.initialize(assembly, pb, t0)

    def set_start(self, assembly, pb, dt):
        for wf in self.__list_weakform:
            wf.set_start(assembly, pb, dt)
    
    def update(self, assembly, pb, dtime):        
        for wf in self.__list_weakform:
            wf.update(assembly, pb, dtime)
    
    def to_start(self):
        #function called if the time step is reinitialized. Used to reset variables to the begining of the step
        for wf in self.__list_weakform:
            wf.to_start()

    def reset(self):
        #function called if all the problem history is reseted.
        for wf in self.__list_weakform:
            wf.reset()
    
    def copy(self):
        #function to copy a weakform at the initial state
        raise NotImplementedError()

    @property
    def list_weakform(self):
        return self.__list_weakform