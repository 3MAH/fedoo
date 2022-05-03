# base class
from fedoo.libUtil.ModelingSpace import ModelingSpace
from fedoo.libConstitutiveLaw.ConstitutiveLaw import ListConstitutiveLaw

class WeakForm:

    __dic = {}

    def __init__(self, ClID = "", space=None):
        assert isinstance(ClID, str) , "An ID must be a string" 
        self.__ID = ClID
        if space is None: 
            space = ModelingSpace.GetActive()
        elif isinstance(space, str):
            space = ModelingSpace.GetAll()[space]
        self.__space = space
        self.assembly_options = {}
        #possible options : 
        # * 'assume_sym' - self.assembly_options['assume_sym'] = True  to accelerate assembly if the weak form may be considered as symmetric
        # * 'nb_pg' - set the default nb_pg
        # * 'mat_lumping' - matrix lumping if set to True
        
        if ClID != "":WeakForm.__dic[self.__ID] = self
        

    def GetID(self):
        return self.__ID

    def GetNumberOfVariables(self):
        return self.__space.nvar
        # return self.GetDifferentialOperator().nvar()

    def GetConstitutiveLaw(self):
        #no constitutive law by default
        pass
            
    def Initialize(self, assembly, pb, initialTime=0.):
        #function called at the very begining of the resolution
        pass

    def InitTimeIncrement(self, assembly, pb, dtime):
        #function called at the begining of a new time increment
        #For now, used only to inform the weak form the the time step for the next increment.
        pass

    def Update(self, assembly, pb, dtime):
        #function called when the problem is updated (NR loop or time increment)
        #- New initial Stress
        #- New initial Displacement
        #- Possible modification of the mesh
        #- Change in constitutive law (internal variable)
        pass
    
    def NewTimeIncrement(self):  
        #function called at the end of a time increment. Used to update variables to the new time.
        pass
    
    def ResetTimeIncrement(self):
        #function called if the time step is reinitialized. Used to reset variables to the begining of the step
        pass

    def Reset(self):
        #function called if all the problem history is reseted.
        pass           
    
    def copy(self):
        #function to copy a weakform at the initial state
        raise NotImplementedError()

    @property
    def space(self):
        return self.__space
        
    @staticmethod
    def GetAll():
        return WeakForm.__dic
    
class WeakFormSum(WeakForm):
    
    def __init__(self, list_weakform, ID=""):    
        assert len(set([a.space for a in list_weakform])) == 1, \
            "Sum of assembly are possible only if all assembly are associated to the same modeling space"
        WeakForm.__init__(self, ID, space = list_weakform[0].space)        
        
        self.__constitutivelaw = ListConstitutiveLaw([a.GetConstitutiveLaw() for a in list_weakform])
        self.__list_weakform = list_weakform
        
        
        
    def GetConstitutiveLaw(self):
        #return a list of constitutivelaw
        return self.__constitutivelaw    
    
    def Initialize(self, assembly, pb, initialTime=0.):
        for wf in self.__list_weakform:
            wf.Initialize(assembly, pb, initialTime)

    def InitTimeIncrement(self, assembly, pb, dtime):
        for wf in self.__list_weakform:
            wf.InitTimeIncrement(assembly, pb, dtime)
    
    def Update(self, assembly, pb, dtime):        
        for wf in self.__list_weakform:
            wf.Update(assembly, pb, dtime)
    
    def NewTimeIncrement(self):  
        for wf in self.__list_weakform:
            wf.NewTimeIncrement()
    
    def ResetTimeIncrement(self):
        #function called if the time step is reinitialized. Used to reset variables to the begining of the step
        for wf in self.__list_weakform:
            wf.ResetTimeIncrement()

    def Reset(self):
        #function called if all the problem history is reseted.
        for wf in self.__list_weakform:
            wf.Reset()
    
    def copy(self):
        #function to copy a weakform at the initial state
        raise NotImplementedError()

    @property
    def list_weakform(self):
        return self.__list_weakform
        


def GetAll():
    return WeakForm.GetAll()


