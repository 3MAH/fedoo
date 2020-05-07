#baseclass

class ConstitutiveLaw:

    __dic = {}

    def __init__(self, ClID = ""):
        assert isinstance(ClID, str) , "An ID must be a string" 
        self.__ID = ClID
        self.__localFrame = None

        ConstitutiveLaw.__dic[self.__ID] = self

    def GetID(self):
        return self.__ID

    def SetLocalFrame(self, localFrame):
        self.__localFrame = localFrame

    def GetLocalFrame(self):
        return self.__localFrame 
    
    def Reset(self): 
        #function called to restart a problem (reset all internal variables)
        pass
    
    def NewTimeIncrement(self):  
        #function called when the time is increased. Not used for elastic laws
        pass
    
    def ResetTimeIncrement(self):
        #function called if the time step is reinitialized. Not used for elastic laws
        pass

    @staticmethod
    def GetAll():
        return ConstitutiveLaw.__dic


def GetAll():
    return ConstitutiveLaw.GetAll()


