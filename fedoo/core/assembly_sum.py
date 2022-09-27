"""This module contains the AssemblySum class"""

from fedoo.core.base import AssemblyBase, ConstitutiveLaw, WeakForm



#=============================================================
# Class that build a sum of Assembly
#=============================================================

#need to be modified to include a list of constitutivelaw update. 
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
    name: str
        name of the Assembly             
    assembly_output: Assembly (optional keyword arg)
        Assembly object used to extract output values (using Problem.get_results or Problem.save_results)
    """
    def __init__(self, list_assembly, name ="", **kargs):      
        AssemblyBase.__init__(self, name)  
        
        for i,assembly in enumerate(list_assembly):
            if isinstance(assembly, str): list_assembly[i] = AssemblyBase.get_all()[assembly]                                
            
        assert len(set([a.space for a in list_assembly])) == 1, \
            "Sum of assembly are possible only if all assembly are associated to the same modeling space"
        assert len(set([a.mesh.n_nodes for a in list_assembly])) == 1,\
            "Sum of assembly are possible only if the two meshes have the same number of Nodes"

        self.__list_assembly = list_assembly
        self.__assembly_output = kargs.get('assembly_output', None)
                        
        self.mesh = list_assembly[0].mesh

        if name == "":
            name = '_'.join([assembly.name for assembly in list_assembly])    
            
        self.__reload = kargs.pop('reload', 'all')                      


    def assemble_global_mat(self,compute='all'):
        if self.__reload == 'all': 
            for assembly in self.__list_assembly:
                assembly.assemble_global_mat(compute)
        else:
            for numAssembly in self.__reload:
                self.__list_assembly[numAssembly].assemble_global_mat(compute)
            
        if not(compute == 'vector'):         
            self.global_matrix = sum([assembly.get_global_matrix() for assembly in self.__list_assembly])
        if not(compute == 'matrix'):
            self.global_vector = sum([assembly.get_global_vector() for assembly in self.__list_assembly])
    
    def update(self, pb, dtime=None, compute = 'all'):
        """
        Update the associated weak form and assemble the global matrix
        Parameters: 
            - pb: a Problem object containing the Dof values
            - time: the current time        
        """
        if self.__reload == 'all' or compute in ['vector', 'none']: #if compute == 'vector' or 'none' the reload arg is ignored
            for assembly in self.__list_assembly:
                assembly.update(pb,dtime,compute)           
        else:
            for numAssembly in self.__reload:
                self.__list_assembly[numAssembly].update(pb,dtime,compute)
                    
        if not(compute == 'vector'):         
            self.global_matrix =  sum([assembly.get_global_matrix() for assembly in self.__list_assembly])
        if not(compute == 'matrix'):
            self.global_vector =  sum([assembly.get_global_vector() for assembly in self.__list_assembly]) 


    def set_start(self, pb, dt):
        """
        Apply the modification to the constitutive equation required at each new time increment. 
        Generally used to increase non reversible internal variable
        Assemble the new global matrix. 
        """
        for assembly in self.__list_assembly:
            assembly.set_start(pb, dt)   
                

    def initialize(self, pb, t0=0.):
        """
        reset the current time increment (internal variable in the constitutive equation)
        Doesn't assemble the new global matrix. Use the Update method for that purpose.
        """
        for assembly in self.__list_assembly:
            assembly.initialize(pb, t0=0.)   

    def to_start(self):
        """
        Reset the current time increment (internal variable in the constitutive equation)
        Doesn't assemble the new global matrix. Use the Update method for that purpose.
        """
        for assembly in self.__list_assembly:
            assembly.to_start()         

    def reset(self):
        """
        reset the assembly to it's initial state.
        Internal variable in the constitutive equation are reinitialized 
        And stored global matrix and vector are deleted
        """
        for assembly in self.__list_assembly:
            assembly.reset() 
        self.delete_global_mat()

    @property
    def list_assembly(self):
        return self.__list_assembly
   
    @property
    def assembly_output(self):
        return self.__assembly_output
    

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
class WeakFormSum(WeakForm):
    
    def __init__(self, list_weakform, name =""):    
        assert len(set([a.space for a in list_weakform])) == 1, \
            "Sum of assembly are possible only if all assembly are associated to the same modeling space"
        WeakForm.__init__(self, name, space = list_weakform[0].space)        
        
        if any([wf.assembly_options!={} for wf in list_weakform]):
            self.assembly_options = None
            # if assembly_options is None, the weakForm have to be splited into several sub-weakform before 
            # being used in an Assembly. This is automatically done when using Assembly.Create function
            # The restulting Assembly will be an AssemblySum object
            
        self.__constitutivelaw = ListConstitutiveLaw([a.GetConstitutiveLaw() for a in list_weakform])
        self.__list_weakform = list_weakform
        
    def GetConstitutiveLaw(self):
        #return a list of constitutivelaw
        return self.__constitutivelaw    
    
    def get_DifferentialOperator(self, mesh=None, localFrame = None):
        Diff = 0
        self._list_mat_lumping = []
        for wf in self.__list_weakform: 
            Diff_wf = wf.get_DifferentialOperator(mesh, localFrame)
            mat_lumping = wf.assembly_options.get('mat_lumping', False) #True of False
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
        
    

# def Sum(*listAssembly, name ="", **kargs):
#     """
#     Return a new assembly which is a sum of N assembly. 
#     Assembly.Sum(assembly1, assembly2, ..., assemblyN, name ="", reload = [1,4] )
    
#     The N first arguments are the assembly to be summed.
#     name is the name of the created assembly:
#     reload: a list of indices for subassembly that are recomputed at each time the summed assembly
#     is Launched. Default is 'all' (equivalent to all indices).     
#     """
#     return AssemblySum(list(listAssembly), name, **kargs)
