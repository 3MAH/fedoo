#baseclass

class AssemblyBase:

    __dic = {}

    def __init__(self, ID = "", space=None):
        assert isinstance(ID, str) , "An ID must be a string" 
        self.__ID = ID

        self.__GlobalMatrix = None
        self.__GlobalVector = None
        self.__Mesh = None 
        
        if ID != "": AssemblyBase.__dic[self.__ID] = self
        self.__space = space

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
    
    @property
    def space(self):
        return self.__space

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
    """
    Build a sum of Assembly objects
    All the Assembly objects should be associated to:
    * meshes based on the same list of nodes.
    * the same modeling space (ie the same space property)
        
    Parameters
    ----------
    list_assembly: list of Assembly 
        list of Assembly objects to sum
    ID: str
        ID of the Assembly             
    assembly_output: Assembly (optional keyword arg)
        Assembly object used to extract output values (using Problem.GetResults or Problem.SaveResults)
    """
    def __init__(self, list_assembly, ID="", **kargs):        
        for i,assembly in enumerate(list_assembly):
            if isinstance(assembly, str): list_assembly[i] = AssemblyBase.GetAll()[assembly]                                
            
        assert len(set([a.space for a in list_assembly])) == 1, \
            "Sum of assembly are possible only if all assembly are associated to the same modeling space"
        assert len(set([a.GetMesh().n_nodes for a in list_assembly])) == 1,\
            "Sum of assembly are possible only if the two meshes have the same number of Nodes"

        self.__list_assembly = list_assembly
        self.__assembly_output = kargs.get('assembly_output', None)
                        
        self.__Mesh = list_assembly[0].GetMesh()

        if ID == "":
            ID = '_'.join([assembly.GetID() for assembly in list_assembly])    
            
        self.__reload = kargs.pop('reload', 'all')                      
        
        AssemblyBase.__init__(self, ID)                       

    def SetMesh(self, mesh):
        self.__Mesh = mesh

    def GetMesh(self):
        return self.__Mesh

    def ComputeGlobalMatrix(self,compute='all'):
        if self.__reload == 'all': 
            for assembly in self.__list_assembly:
                assembly.ComputeGlobalMatrix(compute)
        else:
            for numAssembly in self.__reload:
                self.__list_assembly[numAssembly].ComputeGlobalMatrix(compute)
            
        if not(compute == 'vector'):         
            self.SetMatrix(sum([assembly.GetMatrix() for assembly in self.__list_assembly]))
        if not(compute == 'matrix'):
            self.SetVector(sum([assembly.GetVector() for assembly in self.__list_assembly]))
    
    def Update(self, pb, dtime=None, compute = 'all'):
        """
        Update the associated weak form and assemble the global matrix
        Parameters: 
            - pb: a Problem object containing the Dof values
            - time: the current time        
        """
        if self.__reload == 'all' or compute in ['vector', 'none']: #if compute == 'vector' or 'none' the reload arg is ignored
            for assembly in self.__list_assembly:
                assembly.Update(pb,dtime,compute)           
        else:
            for numAssembly in self.__reload:
                self.__list_assembly[numAssembly].Update(pb,dtime,compute)
                    
        if not(compute == 'vector'):         
            self.SetMatrix( sum([assembly.GetMatrix() for assembly in self.__list_assembly]) )
        if not(compute == 'matrix'):
            self.SetVector( sum([assembly.GetVector() for assembly in self.__list_assembly]) )

    def InitTimeIncrement(self, pb, dtime=None):
        """
        May be used if required to initialize a new time increment 
        """
        for assembly in self.__list_assembly:
            assembly.InitTimeIncrement(pb, dtime)   

    def Initialize(self, pb, initialTime=0.):
        """
        Reset the current time increment (internal variable in the constitutive equation)
        Doesn't assemble the new global matrix. Use the Update method for that purpose.
        """
        for assembly in self.__list_assembly:
            assembly.Initialize(pb, initialTime=0.)   

    def ResetTimeIncrement(self):
        """
        Reset the current time increment (internal variable in the constitutive equation)
        Doesn't assemble the new global matrix. Use the Update method for that purpose.
        """
        for assembly in self.__list_assembly:
            assembly.ResetTimeIncrement()        

    def NewTimeIncrement(self):
        """
        Apply the modification to the constitutive equation required at each change of time increment. 
        Generally used to increase non reversible internal variable
        Doesn't assemble the new global matrix. Use the Update method for that purpose.
        """
        for assembly in self.__list_assembly:
            assembly.NewTimeIncrement()        

    def Reset(self):
        """
        Reset the assembly to it's initial state.
        Internal variable in the constitutive equation are reinitialized 
        And stored global matrix and vector are deleted
        """
        for assembly in self.__list_assembly:
            assembly.Reset() 
        self.deleteGlobalMatrix()

    @property
    def list_assembly(self):
        return self.__list_assembly
   
    @property
    def assembly_output(self):
        return self.__assembly_output
    



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
