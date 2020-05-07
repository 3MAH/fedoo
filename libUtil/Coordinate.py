class Coordinate:
    __coordinate = {} # attribut statique de la classe

    __rank = 0

    def __init__(self,name):
        assert isinstance(name,str) , "The coordinte name must be a string"

        if name not in Coordinate.__coordinate.keys():
            Coordinate.__coordinate[name] = Coordinate.__rank
            Coordinate.__rank +=1

    @staticmethod
    def GetRank(name):
        if name not in Coordinate.__coordinate.keys():
            assert 0, "the coordinate name does not exist" 
        return Coordinate.__coordinate[name]

    @staticmethod
    def GetNumberOfCoordinate():
        return Coordinate.__rank

    @staticmethod
    def ListAll():
        return Coordinate.__coordinate.keys()