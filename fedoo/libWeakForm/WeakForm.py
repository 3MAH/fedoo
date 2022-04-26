# base class
from fedoo.libUtil.ModelingSpace import ModelingSpace


class WeakForm:

    __dic = {}

    def __init__(self, ClID = "", space=None):
        assert isinstance(ClID, str) , "An ID must be a string" 
        self.__ID = ClID
        self.assumeSymmetric = False #use to accelerate assembly if the weak form may be considered as symmetric
        if space is None: 
            space = ModelingSpace.GetActive()
        elif isinstance(space, str):
            space = ModelingSpace.GetAll()[space]
        self.__space = space
        
        WeakForm.__dic[self.__ID] = self

    def GetID(self):
        return self.__ID

    def GetNumberOfVariables(self):
        return self.GetDifferentialOperator().nvar()

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
    
    


def GetAll():
    return WeakForm.GetAll()


