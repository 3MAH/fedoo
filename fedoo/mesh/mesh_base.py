#baseclass

class MeshBase:

    __dic = {}

    def __init__(self, name = ""):
        assert isinstance(name, str) , "name must be a string" 
        self.__name = name
        
        if name != "":
            MeshBase.__dic[self.__name] = self

    @property
    def name(self):
        return self.__name

    @staticmethod
    def get_all():
        return MeshBase.__dic

def get_all():
    return MeshBase.get_all()
