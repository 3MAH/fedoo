# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 15:27:43 2020

@author: Etienne
"""
# from fedoo.core.base   import ProblemBase 
import numpy as np
from fedoo.core.boundary_conditions import BCBase, MPC, ListBC
from fedoo.core.base import ProblemBase
from fedoo.core.mesh import MeshBase


from scipy.spatial.transform import Rotation

# USE_SIMCOON = True

# if USE_SIMCOON: 
#     try:
#         from simcoon import simmit as sim
#         USE_SIMCOON = True
#     except:
#         USE_SIMCOON = False
#         print('WARNING: Simcoon library not found. The simcoon constitutive law is disabled.')       

# if USE_SIMCOON:    
    
    



class RigidTie(BCBase):
    def __init__(self, list_nodes, node_cd, var_cd, name = ""):
        self.list_nodes = list_nodes
        self.node_cd = node_cd
        self.var_cd = var_cd        
        self.bc_type = 'RigidTie'
        BCBase.__init__(self, name)
        self._keep_at_end = True
        
        self._update_during_inc = 1
        
    def __repr__(self):
        list_str = ['Rigid Tie:'.format(self.dim)]
        if self.name != "": list_str.append("name = '{}'".format(self.name))
        
        return "\n".join(list_str)


    
    def initialize(self, problem):
        pass
        # for i,var in enumerate(self.var_cd):
        #     if isinstance(var, str):
        #         self.var_cd[i] = problem.space.variable_rank(var)
                

    def generate(self, problem, t_fact=1, t_fact_old=None):
                
        mesh = problem.mesh                     
        var_cd = self.var_cd
        node_cd = self.node_cd #node_cd[0] -> node defining center of rotation
        list_nodes = self.list_nodes

        # rot_center = node_cd[0]
        res = ListBC()
        
        dof_cd = [problem.space.variable_rank(var_cd[i])*mesh.n_nodes + node_cd[i] for i in range(len(var_cd))]
        
        # dof_ref  = [problem._Xbc[dof] if dof in problem.dof_blocked else problem._X[dof] for dof in dof_cd]
        # dof_ref  = [problem.get_dof_solution()[dof] + problem._Xbc[dof] if dof in problem.dof_blocked else problem.get_dof_solution()[dof] for dof in dof_cd]
        if problem.get_dof_solution() is 0:
            dof_ref  = np.array([problem._Xbc[dof] for dof in dof_cd])
        else:            
            dof_ref  = np.array([problem.get_dof_solution()[dof] + problem._Xbc[dof] for dof in dof_cd])
                
        disp_ref = dof_ref[:3] #reference displacement
        angles = dof_ref[3:] #rotation angle
                
        sin = np.sin(angles)
        cos = np.cos(angles)
        
        # R = Rotation.from_euler("XYZ", angles).as_matrix()
        # #or        
        # R2 = np.array([[cos[1]*cos[2], -cos[1]*sin[2], sin[1]],
        #           [cos[0]*sin[2] + cos[2]*sin[0]*sin[1], cos[0]*cos[2]-sin[0]*sin[1]*sin[2], -cos[1]*sin[0]],
        #           [sin[0]*sin[2] - cos[0]*cos[2]*sin[1], cos[2]*sin[0]+cos[0]*sin[1]*sin[2], cos[0]*cos[1]]] )
        
                    
        
        #approche globale :
        # crd = mesh.nodes + problem.get_disp()
        # Uini = (crd - crd[0]) @ R.T + disp_ref #node disp at the begining of the iteration
        
        #approche incrémentale: 
        
        dR_drx = np.array([[0, 0, 0],
                  [-sin[0]*sin[2] + cos[2]*cos[0]*sin[1], -sin[0]*cos[2]-cos[0]*sin[1]*sin[2], -cos[1]*cos[0]],
                  [cos[0]*sin[2] + sin[0]*cos[2]*sin[1], cos[2]*cos[0]-sin[0]*sin[1]*sin[2], -sin[0]*cos[1]]] )
        
        dR_dry = np.array([[-sin[1]*cos[2], +sin[1]*sin[2], cos[1]],
                  [cos[2]*sin[0]*cos[1], -sin[0]*cos[1]*sin[2], sin[1]*sin[0]],
                  [-cos[0]*cos[2]*cos[1], cos[0]*cos[1]*sin[2], -cos[0]*sin[1]]] )
        
        dR_drz = np.array([[-cos[1]*sin[2], -cos[1]*cos[2], 0],
                  [cos[0]*cos[2] - sin[2]*sin[0]*sin[1], -cos[0]*sin[2]-sin[0]*sin[1]*cos[2], 0],
                  [sin[0]*cos[2] + cos[0]*sin[2]*sin[1], -sin[2]*sin[0]+cos[0]*sin[1]*cos[2], 0]] )
        
        crd = mesh.nodes[list_nodes] - mesh.nodes[node_cd[0]]
        du_drx = crd @ dR_drx.T
        du_dry = crd @ dR_dry.T
        du_drz = crd @ dR_drz.T #shape = (nnodes, nvar) with nvar = 3 in 3d (ux, uy, uz)
        
        #### MPC ####
                
        # dU - dU_ref - du_drx*drx_ref - du_dry*dry_ref - du_drz*drz_ref = 0
        # with shapes: dU, du_drx, ... -> (nnodes, nvar) - dU_ref -> (nvar), drx_ref, ... -> scalar         
        # dU are associated to eliminated dof and should be different than ref dof        
        # or
        # dUx - dUx_ref - du_drx[:,0]*drx_ref - du_dry[:,0]*dry_ref - du_drz[:,0]*drz_ref = 0
        # dUy - dUy_ref - du_drx[1]*drx_ref - du_dry[1]*dry_ref - du_drz[1]*drz_ref = 0
        # dUz - dUz_ref - du_drx[2]*drx_ref - du_dry[2]*dry_ref - du_drz[2]*drz_ref = 0
        res.append(
            MPC([list_nodes, np.full_like(list_nodes,node_cd[0]), np.full_like(list_nodes,node_cd[3]), np.full_like(list_nodes,node_cd[4]), np.full_like(list_nodes,node_cd[5])],
                ['DispX',                     var_cd[0],                     var_cd[3],    var_cd[4],    var_cd[5]], 
                [np.full_like(list_nodes,1.), np.full_like(list_nodes,-1.), -du_drx[:,0], -du_dry[:,0], -du_drz[:,0]] )
            )
        res.append(            
            MPC([list_nodes, np.full_like(list_nodes,node_cd[1]), np.full_like(list_nodes,node_cd[3]), np.full_like(list_nodes,node_cd[4]), np.full_like(list_nodes,node_cd[5])],
                ['DispY',                     var_cd[1],                     var_cd[3],    var_cd[4],    var_cd[5]], 
                [np.full_like(list_nodes,1.), np.full_like(list_nodes,-1.), -du_drx[:,1], -du_dry[:,1], -du_drz[:,1]] )
            )
        res.append(
            MPC([list_nodes, np.full_like(list_nodes,node_cd[2]), np.full_like(list_nodes,node_cd[3]), np.full_like(list_nodes,node_cd[4]), np.full_like(list_nodes,node_cd[5])],
                ['DispZ',                     var_cd[2],                     var_cd[3],    var_cd[4],    var_cd[5]], 
                [np.full_like(list_nodes,1.), np.full_like(list_nodes,-1.), -du_drx[:,2], -du_dry[:,2], -du_drz[:,2]] )
            )
       
        res.initialize(problem)        
        return res.generate(problem, t_fact, t_fact_old)

  
                   
