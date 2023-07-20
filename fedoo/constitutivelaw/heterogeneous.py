#derive de ConstitutiveLaw
#compatible with the simcoon strain and stress notation

from fedoo.core.mechanical3d import Mechanical3D
from fedoo.core.assembly import Assembly
from copy import deepcopy
# from fedoo.util.voigt_tensors import StressTensorList, StrainTensorList


import numpy as np


class _SubAssembly(Assembly):
    #Assembly with new definition of sv and sv_start that allow maping the global assembly id to the sub_assembly
    def __init__(self,assembly, elset, copied_fields):        
        self.assembly = assembly
        self.elset = elset        
        self.copied_fields = copied_fields
        super().__init__(assembly.weakform, 
                       assembly.mesh.extract_elements(elset), 
                       assembly.elm_type)

    
    @property
    def sv(self):
        if isinstance(self.elset, str):
            return _SubSV(self.assembly, self.assembly.sv, self.assembly.mesh.element_sets[self.elset], self.copied_fields)
        else: 
            return _SubSV(self.assembly, self.assembly.sv, self.elset, self.copied_fields)
    
    @sv.setter
    def sv(self, value):
        pass #ignored - cl are not supposed to change the sv attribute (only the dict content)        
        
    
    @property
    def sv_start(self):
        if isinstance(self.elset, str):
            return _SubSV(self.assembly, self.assembly.sv_start, self.assembly.mesh.element_sets[self.elset], set())
        else: 
            return _SubSV(self.assembly, self.assembly.sv_start, self.elset, set())

    @sv_start.setter
    def sv_start(self, value):
        pass #ignored - cl are not supposed to change the sv attribute (only the dict content)


class _SubSV():
    #class just here to map the good id for elset in the global state variable
    def __init__(self, assembly, sv, elset, copied_fields):
        self.sv = sv
        self.elset = elset
        self.assembly = assembly
        self.copied_fields = copied_fields
    
    
    def __contains__(self, item):
        return item in self.sv

    
    def __getitem__(self, k):
        #assume sv values are defined on gauss points. 
        #perhaps it may be usefull to allow other definitions
        if self.sv[k] is 0: return 0
        elset = (np.array(self.elset) + np.c_[np.arange(0,self.assembly.n_gauss_points, self.assembly.mesh.n_elements)]).reshape(-1)

        if isinstance(self.sv[k], list):
            try: 
                return self.sv[k].__class__(self.sv[k].asarray()[..., elset])
            except:
                return self.sv[k].__class__(np.array(self.sv[k])[..., elset])                
        else: #shoud be array
            return self.sv[k][..., elset] #gp id should be the last axis
    
    
    def __setitem__(self, k, v):#todefine properly
        #assume sv values are defined on gauss points. 
        #perhaps it may be usefull to allow other definitions        
        
        if k in self.sv: # or k in self.copied_fields: 
            if k not in self.copied_fields:
                self.sv[k] = deepcopy(self.sv[k])
                self.copied_fields.add(k)
            if self.sv[k] is 0: 
                del self.sv[k]
                self.__setitem__(k,v)
            
            elset = (np.array(self.elset) + np.c_[np.arange(0,self.assembly.n_gauss_points, self.assembly.mesh.n_elements)]).reshape(-1)
            
            if isinstance(self.sv[k], list): #assume it is a ListStressTensor or a ListStrainTensor object
                self.sv[k].array[..., elset] = v
            else:
                try:
                    self.sv[k][..., elset] = v 
                except:
                    #try if scalar values are given
                    self.sv[k][..., elset] = v[...,np.newaxis]
        else:             
            if isinstance(v, np.ndarray):
                isarray = True
                arr = v                                
            else: #maybe a List for instance TensorStressList object or a list of list
                isarray = False
                try:
                    arr = v.asarray()
                except:
                    arr = np.array(v)
                
            shape = list(arr.shape)
            #treat the special case where TangentMatrix is a 6x6 matrix (each component are scalar for homogeneous materials)
            if k == 'TangentMatrix' and len(shape) == 2:
                shape.append(self.assembly.n_gauss_points)
            else:                    
                shape[-1] = self.assembly.n_gauss_points
            
            if isarray: self.sv[k] = np.zeros(shape)
            else: self.sv[k] =  v.__class__(np.zeros(shape))
            
            self.copied_fields.add(k)
            self.__setitem__(k,v)
            
            


class Heterogeneous(Mechanical3D):    
   
    def __init__(self, tup_cl, tup_elset , name =""):
       
        Mechanical3D.__init__(self, name) # heritage
        self.list_cl = tup_cl
        self.list_elset = tup_elset

                        
    def initialize(self, assembly, pb):
        # self.list_mesh = [assembly.mesh.extract_elements(elset) for elset in self.list_elset]
        self._copied_fields = set() #set of field that have already been copied
        #assembly.sv field need to be copied and can't be just modified because 
        #it will also modified assembly.sv_start (shallow copy for performance reason)
        #copied_fields is a set that keep in memory the field that have already been copied. 
        self.list_assembly = [_SubAssembly(assembly, elset, self._copied_fields) for elset in self.list_elset]                
        
        for i,cl in enumerate(self.list_cl):
            cl.initialize(self.list_assembly[i], pb)
        

    def update(self, assembly, pb): 
        self._copied_fields.clear() #to force a new copy of each modified fields
        for i,cl in enumerate(self.list_cl):
            cl.update(self.list_assembly[i], pb)
        
        
    def set_start(self, assembly, pb):        
        self._copied_fields.clear() #to force a new copy of each modified fields
        for i,cl in enumerate(self.list_cl):
            cl.set_start(self.list_assembly[i], pb)        
            

    def to_start(self, assembly, pb):
        self._copied_fields.clear() #to force a new copy of each modified fields
        for i,cl in enumerate(self.list_cl):
            cl.to_start(self.list_assembly[i], pb)     
            
            
    # def get_tangent_matrix(self, assembly, dimension=None): #Tangent Matrix in lobal coordinate system (no change of basis) 
    
    #     if dimension is None: dimension = assembly.space.get_dimension()
        
    #     # H = self.local2global_H(self._H)
    #     if dimension == "2Dstress":
    #         return self.get_H_plane_stress(assembly.sv['TangentMatrix'])
    #     else: 
    #          assembly.sv['TangentMatrix']
          

    # def get_elastic_matrix(self, dimension = "3D"):
    #     return self.get_tangent_matrix(None,dimension)
        
    
    # def ComputeStrain(self, assembly, pb, nlgeom, type_output='GaussPoint'):
    #     displacement = pb.get_dof_solution()                
    #     if displacement is 0: 
    #         return 0 #if displacement = 0, Strain = 0
    #     else:
    #         return assembly.get_strain(displacement, type_output)  
    
       
    
    
    
    
     
                        
        