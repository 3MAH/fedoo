#baseclass

class AssemblyBase:

    __dic = {}

    def __init__(self, ID = ""):
        assert isinstance(ID, str) , "An ID must be a string" 
        self.__ID = ID

        self.__GlobalMatrix = None
        self.__GlobalVector = None
        self.__Mesh = None 
        
        AssemblyBase.__dic[self.__ID] = self

    def GetID(self):
        return self.__ID

    def GetMatrix(self):
        if self.__GlobalMatrix is None: self.ComputeGlobalMatrix()        
        return self.__GlobalMatrix

    def GetVector(self):
        if self.__GlobalVector is None: self.ComputeGlobalMatrix()        
        return self.__GlobalVector

    def SetVector(self, V):
        self.__GlobalVector = V 

    def SetMatrix(self, M):
        self.__GlobalMatrix = M
    
    def AddMatrix(self, M):
        self.__GlobalMatrix += M
        
    def computeGlobalMatrix(self):
        #needs to be defined in inherited classes
        pass

    def deleteGlobalMatrix(self):
        """
        Delete Global Matrix and Global Vector related to the assembly. 
        This method allow to force a new assembly
        """
        self.__GlobalMatrix = None
        self.__GlobalVector = None

    @staticmethod
    def GetAll():
        return AssemblyBase.__dic
    
    @staticmethod
    def Launch(ID):
        """
        Assemble the global matrix and global vector of the assembly ID
        ID is a str
        """
        AssemblyBase.GetAll()[ID].ComputeGlobalMatrix()


class AssemblySum(AssemblyBase):
   
    def __init__(self, ListAssembly, ID="", **kargs):        
        for i,assembly in enumerate(ListAssembly):
            if isinstance(assembly, str): ListAssembly[i] = AssemblyBase.GetAll()[assembly]                                
            
        self.__ListAssembly = ListAssembly
                
#        assert assembly1.GetMesh().GetNumberOfNodes() == assembly2.GetMesh().GetNumberOfNodes(), \
#                    "Sum of assembly are possible only if the two meshes have the same number of Nodes"
                    
        self.__Mesh = ListAssembly[0].GetMesh()

        if ID == "":
            ID = '_'.join([assembly.GetID() for assembly in ListAssembly])    
            
        self.__reload = kargs.pop('reload', 'all')                      
        
        AssemblyBase.__init__(self, ID)                       

    def SetMesh(self, mesh):
        self.__Mesh = mesh

    def GetMesh(self):
        return self.__Mesh

    def ComputeGlobalMatrix(self,compute='all'):
        if self.__reload == 'all': 
            for assembly in self.__ListAssembly:
                assembly.ComputeGlobalMatrix(compute)
        else:
            for numAssembly in self.__reload:
                self.__ListAssembly[numAssembly].ComputeGlobalMatrix(compute)
            
        if not(compute == 'vector'):         
            self.SetMatrix(sum([assembly.GetMatrix() for assembly in self.__ListAssembly]))
        if not(compute == 'matrix'):
            self.SetVector(sum([assembly.GetVector() for assembly in self.__ListAssembly]))
    
    def Update(self, pb, time=None, compute = 'all'):
        """
        Update the associated weak form and assemble the global matrix
        Parameters: 
            - pb: a Problem object containing the Dof values
            - time: the current time        
        """
        if self.__reload == 'all' or compute == 'vector': #if compute == 'vector' the reload arg is ignored
            for assembly in self.__ListAssembly:
                assembly.Update(pb,time,compute)           
        else:
            for numAssembly in self.__reload:
                self.__ListAssembly[numAssembly].Update(pb,time,compute)
                    
        if not(compute == 'vector'):         
            self.SetMatrix( sum([assembly.GetMatrix() for assembly in self.__ListAssembly]) )
        if not(compute == 'matrix'):
            self.SetVector( sum([assembly.GetVector() for assembly in self.__ListAssembly]) )

    def ResetTimeIncrement(self):
        """
        Reset the current time increment (internal variable in the constitutive equation)
        Doesn't assemble the new global matrix. Use the Update method for that purpose.
        """
        for assembly in self.__ListAssembly:
            assembly.ResetTimeIncrement()        

    def NewTimeIncrement(self):
        """
        Apply the modification to the constitutive equation required at each change of time increment. 
        Generally used to increase non reversible internal variable
        Doesn't assemble the new global matrix. Use the Update method for that purpose.
        """
        for assembly in self.__ListAssembly:
            assembly.NewTimeIncrement()        

    def Reset(self):
        """
        Reset the assembly to it's initial state.
        Internal variable in the constitutive equation are reinitialized 
        And stored global matrix and vector are deleted
        """
        for assembly in self.__ListAssembly:
            assembly.Reset() 
        self.deleteGlobalMatrix()




def Sum(*listAssembly, ID="", **kargs):
    """
    Return a new assembly which is a sum of N assembly. 
    Assembly.Sum(assembly1, assembly2, ..., assemblyN, ID="", reload = [1,4] )
    
    The N first arguments are the assembly to be summed.
    ID is the name of the created assembly:
    reload: a list of indices for subassembly that are recomputed at each time the summed assembly
    is Launched. Default is 'all' (equivalent to all indices).     
    """
    return AssemblySum(list(listAssembly), ID, **kargs)
            
def GetAll():
    return AssemblyBase.GetAll()

def Launch(ID):
    AssemblyBase.GetAll()[ID].ComputeGlobalMatrix()    
