#baseclass

class MeshBase:

    __dic = {}

    def __init__(self, meshID = ""):
        assert isinstance(meshID, str) , "An ID must be a string" 
        self.__ID = meshID

        MeshBase.__dic[self.__ID] = self

    def GetID(self):
        return self.__ID

    @staticmethod
    def GetAll():
        return MeshBase.__dic

def GetAll():
    return MeshBase.GetAll()
