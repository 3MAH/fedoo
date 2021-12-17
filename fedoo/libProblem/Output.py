# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 17:54:05 2021

@author: Etienne
"""

import numpy as np
from fedoo.libMesh.Mesh import *
from fedoo.libAssembly.AssemblyBase import AssemblyBase
from fedoo.libUtil.ExportData import ExportData



_available_output = ['pkii',   'pk2',   'kirchoff',   'kirchhoff',   'cauchy',
                      'pkii_vm','pk2_vm','krichoff_vm','kirchhoff_vm','cauchy_vm',
                      'pkii_pc', 'pk2_pc', 'kirchoff_pc', 'kirchhoff_pc', 'cauchy_pc', 'stress_pc',
                      'pkii_pdir1', 'pk2_pdir1', 'kirchoff_pdir1', 'kirchhoff_pdir1', 'cauchy_pdir1', 'stress_pdir1',
                      'pkii_pdir2', 'pk2_pdir2', 'kirchoff_pdir2', 'kirchhoff_pdir2', 'cauchy_pdir2', 'stress_pdir2',
                      'pkii_pdir3', 'pk2_pdir3', 'kirchoff_pdir3', 'kirchhoff_pdir3', 'cauchy_pdir3', 'stress_pdir3',
                      'disp', 'rot', 'strain', 'statev', 'stress', 'stress_vm', 'external_force', 'internal_force', 'internal_force_global']
_available_format = ['vtk', 'msh', 'txt', 'npy', 'npz', 'npz_compressed']

_label_dict = {'pkii':'PKII', 'pk2':'PKII', 'kirchoff':'Kirchhoff', 'kirchhoff':'Kirchhoff', 'cauchy':'Cauchy',
              'stress':'Stress', 'strain':'Strain'} #use to get label associated with some outputs

def _GetResults(pb, assemb, output_list, output_type='Node', position = 1, res_format = None):
        
        if isinstance(output_list, str): output_list = [output_list]                

        if output_type.lower() == 'node': output_type = 'Node'
        elif output_type.lower() == 'element': output_type = 'Element'
        elif output_type.lower() == 'gausspoint': output_type = 'GaussPoint'
        else: raise NameError("output_type should be either 'Node', 'Element' or 'GaussPoint'")
                
        for i,res in enumerate(output_list):
            output_list[i] = res = res.lower()
            if res not in _available_output:
                print("WARNING: '", res, "' doens't match to any available output")
                print("Specified output ignored")
                print("List of available output: ", _available_output)
                
        list_filename = []
        list_ExportData = []            
        
        data_sav = {} #dict to keep data in memory that may be used more that one time

        if isinstance(assemb, str): 
            assemb = AssemblyBase.GetAll()[assemb]  
                  
        material = assemb.GetWeakForm().GetConstitutiveLaw()
        
        result = {}
                    
        for res in output_list:
            res = res.lower()
                                
            if res in ['pkii', 'pk2', 'kirchoff', 'kirchhoff', 'cauchy','strain', 'stress']:
                if res in data_sav: 
                    data = data_sav[res] #avoid a new data conversion
                else:
                    if res in ['pkii','pk2']:
                        data = material.GetPKII()
                    elif res in ['stress']:
                        #stress for small displacement
                        data = material.GetStress(position = position)
                    elif res in ['kirchoff','kirchhoff']:
                        data = material.GetKirchhoff()
                    elif res == 'cauchy':
                        data = material.GetCauchy()
                    elif res == 'strain':
                        data = material.GetStrain(position = position)                                                
                    
                    data = data.Convert(assemb, None, output_type)                        
                    
                    #keep data in memory in case it may be used later for vm, pc or pdir stress computation
                    data_sav[res] = data
                
                label_data = _label_dict[res]
                if res_format == 'vtk': data = data.vtkFormat()
                elif res_format in ['msh', 'txt']: data = np.array(data).T
                elif res_format in ['npy', 'npz']: data = np.array(data)
                                        
            elif res == 'disp':
                if output_type == 'Node':
                    if res_format in ['vtk', 'msh', 'txt']:
                        data = pb.GetDisp().T
                    else: 
                        data = pb.GetDisp()
                    label_data = 'Disp'
                else: 
                    raise NameError("Displacement is only a Node data and is incompatible with the output format specified")                    

            elif res == 'rot':
                if output_type == 'Node': 
                    if res_format in ['vtk', 'msh', 'txt']:                            
                        data = pb.GetRot().T
                    else: 
                        data = pb.GetRot()
                    label_data = 'Rot'
                else: 
                    raise NameError("Displacement is only a Node data and is incompatible with the output format specified")                    
            
            elif res == 'external_force':
                if output_type == 'Node':                             
                    data = assemb.GetExternalForces(pb.GetDoFSolution())
                    label_data = 'External_Load'                        
                else: 
                    raise NameError("External_Force is only Node data and is incompatible with the specified output format")    

            elif res in ['pkii_vm', 'pk2_vm', 'kirchoff_vm', 'kirchhoff_vm', 'cauchy_vm', 'stress_vm']:
                if res[:-3] in data_sav: 
                    data=data_sav[res[:-3]]
                else:
                    if res in ['pkii_vm','pk2_vm']:
                        data = material.GetPKII()
                    elif res in ['stress_vm']:
                        data = material.GetStress(position = position)
                    elif res in ['kirchoff_vm','kirchhoff_vm']:
                        data = material.GetKirchhoff()
                    elif res == 'cauchy_vm':
                        data = material.GetCauchy()

                    data = assemb.ConvertData(data, None, output_type)
                    data_sav[res[:-3]] = data
                                            
                label_data = _label_dict[res[:-3]] + '_Mises'
                data = data.vonMises()
                    
            elif res in ['pkii_pc', 'pk2_pc', 'kirchoff_pc', 'kirchhoff_pc', 'cauchy_pc', 'stress_pc', 
                         'pkii_pdir1', 'pk2_pdir1', 'kirchoff_pdir1', 'kirchhoff_pdir1', 'cauchy_pdir1', 'stress_pdir1',
                         'pkii_pdir2', 'pk2_pdir2', 'kirchoff_pdir2', 'kirchhoff_pdir2', 'cauchy_pdir2', 'stress_pdir2',
                         'pkii_pdir3', 'pk2_pdir3', 'kirchoff_pdir3', 'kirchhoff_pdir3', 'cauchy_pdir3', 'stress_pdir3']:
                #stress principal component
                if res[-3:] == '_pc': measure_type = res[:-3]
                else: measure_type = res[:-6]
                
                if  measure_type+'_principal' in data_sav:
                    data = data_sav[measure_type+'_principal']
                    
                elif measure_type in data_sav: 
                    data = data_sav[measure_type]
                    data = data.GetPrincipalStress()
                    data_sav[measure_type+'_principal'] = data
                    
                else:                    
                    if measure_type in ['pkii','pk2']:
                        data = material.GetPKII()
                    elif measure_type in ['stress']:
                        data = material.GetStress(position = position)
                    elif measure_type in ['kirchoff','kirchhoff']:
                        data = material.GetKirchhoff()
                    elif measure_type == 'cauchy':
                        data = material.GetCauchy()
                    
                    data = data.Convert(assemb, None, output_type)            
                    data_sav[measure_type] = data                        
                    data = data.GetPrincipalStress()
                    data_sav[measure_type+'_principal'] = data
                
                if res[-3:] == '_pc': #principal component
                    data = data[0] #principal component
                    label_data = _label_dict[measure_type] + '_Principal'
                elif res[-6:] == '_pdir1': #1st principal direction    
                    data = data[1][0]
                    label_data = _label_dict[measure_type] + '_PrincipalDir1'
                elif res[-6:] == '_pdir2': #2nd principal direction    
                    data = data[1][2]
                    label_data = _label_dict[measure_type] + '_PrincipalDir2'
                elif res[-6:] == '_pdir3': #3rd principal direction    
                    data = data[1][2]
                    label_data = _label_dict[measure_type] + '_PrincipalDir3'
                    
            elif res in ['statev']:
                data = material.GetStatev().T                    
                data = assemb.ConvertData(data, None, output_type)
                label_data = 'State_Variables'
            
            elif res == 'internal_force':
                data = assemb.GetInternalForces(pb.GetDoFSolution(), 'local')
                data = assemb.ConvertData(data, None, output_type)
                label_data = 'Internal_Load_localCoord'   
                
            elif res == 'internal_force_global':
                data = assemb.GetInternalForces(pb.GetDoFSolution(), 'global')
                data = assemb.ConvertData(data, None, output_type)
                label_data = 'Internal_Load_globalCoord' 
            
            result[label_data] = data
        
        return result
        



class _ProblemOutput:
    def __init__(self):
        self.__list_output = [] #a list containint dictionnary with defined output
                
    def AddOutput(self, filename, assemblyID, output_list, output_type='Node', file_format ='vtk', position = 1):
        if output_type.lower() == 'node': output_type = 'Node'
        elif output_type.lower() == 'element': output_type = 'Element'
        elif output_type.lower() == 'gausspoint': output_type = 'GaussPoint'
        else: raise NameError("output_type should be either 'Node', 'Element' or 'GaussPoint'")
        
        file_format = file_format.lower()
        if file_format not in _available_format:
            print("WARNING: '", file_format, "' doens't match to any available file format")
            print("Specified output ignored")
            print("List of available file format: ", _available_format)
        
        for i,res in enumerate(output_list):
            output_list[i] = res = res.lower()
            if res not in _available_output:
                print("WARNING: '", res, "' doens't match to any available output")
                print("Specified output ignored")
                print("List of available output: ", _available_output)
        
        new_output = {'filename': filename, 'assembly': assemblyID, 'type': output_type, 'list': output_list, 'file_format': file_format.lower(), 'position': position}
        self.__list_output.append(new_output)

    def SaveResults(self, pb, comp_output=None):
        
        list_filename = []
        list_ExportData = []            
        
        for output in self.__list_output:
            
            filename = output['filename']
            file_format = output['file_format'].lower()
            output_type = output['type'] #'Node', 'Element' or 'GaussPoint'
            position = output['position']
            
            assemb = AssemblyBase.GetAll()[output['assembly']]               
            # material = assemb.GetWeakForm().GetConstitutiveLaw()
            
            if comp_output is None:
                filename_compl = ""
            else: 
                filename_compl = '_' + str(comp_output)
            
            if file_format in ['vtk', 'msh', 'npz', 'npz_compressed']:                
                filename = filename + filename_compl + '.' + file_format[0:3]
                
                if not(filename in list_filename): 
                    #if file name don't exist in the list we create it
                    list_filename.append(filename)
                    if file_format in ['vtk', 'msh']: OUT = ExportData(assemb.GetMesh().GetID())
                    else: OUT = {} #empty dictionnary cotaining variable                        
                    list_ExportData.append(OUT)                        
                else: 
                    #else, the same file is used   
                    if file_format in ['vtk', 'msh']: 
                        OUT = list_ExportData[list_filename.index(filename)]                                                                             
            
            #compute the results
            res = _GetResults(pb, assemb, output['list'],output_type,position, file_format)
            
            
            # data_sav = {} #dict to keep data in memory that may be used more that one time
            # for res in output['list']:
            #     res = res.lower()
                                    
            #     if res in ['pkii', 'pk2', 'kirchoff', 'kirchhoff', 'cauchy','strain', 'stress']:
            #         if res in data_sav: 
            #             data = data_sav[res] #avoid a new data conversion
            #         else:
            #             if res in ['pkii','pk2']:
            #                 data = material.GetPKII()
            #             elif res in ['stress']:
            #                 #stress for small displacement
            #                 data = material.GetStress(position = position)
            #             elif res in ['kirchoff','kirchhoff']:
            #                 data = material.GetKirchhoff()
            #             elif res == 'cauchy':
            #                 data = material.GetCauchy()
            #             elif res == 'strain':
            #                 data = material.GetStrain(position = position)                                                
                        
            #             data = data.Convert(assemb, None, output_type)                        
                        
            #             #keep data in memory in case it may be used later for vm, pc or pdir stress computation
            #             data_sav[res] = data
                    
            #         label_data = _label_dict[res]
            #         if file_format == 'vtk': data = data.vtkFormat()
            #         elif file_format in ['msh', 'txt', 'npy', 'npz']: data = np.array(data).T
                                            
            #     elif res == 'disp':
            #         if output_type == 'Node':                             
            #             data = pb.GetDisp().T
            #             # data = pb.GetDoFSolution().reshape(6,-1)[0:3].T
            #             label_data = 'Displacement'
            #         else: 
            #             raise NameError("Displacement is only a Node data and is incompatible with the output format specified")                    

            #     elif res == 'rot':
            #         if output_type == 'Node':                             
            #             data = pb.GetRot().T
            #             label_data = 'Rotation'
            #         else: 
            #             raise NameError("Displacement is only a Node data and is incompatible with the output format specified")                    
                
            #     elif res == 'external_force':
            #         if output_type == 'Node':                             
            #             data = assemb.GetExternalForces(pb.GetDoFSolution())
            #             label_data = 'External_Load'                        
            #         else: 
            #             raise NameError("External_Force is only Node data and is incompatible with the specified output format")    

            #     elif res in ['pkii_vm', 'pk2_vm', 'kirchoff_vm', 'kirchhoff_vm', 'cauchy_vm', 'stress_vm']:
            #         if res[:-3] in data_sav: 
            #             data=data_sav[res[:-3]]
            #         else:
            #             if res in ['pkii_vm','pk2_vm']:
            #                 data = material.GetPKII()
            #             elif res in ['stress_vm']:
            #                 data = material.GetStress(position = position)
            #             elif res in ['kirchoff_vm','kirchhoff_vm']:
            #                 data = material.GetKirchhoff()
            #             elif res == 'cauchy_vm':
            #                 data = material.GetCauchy()

            #             data = assemb.ConvertData(data, None, output_type)
            #             data_sav[res[:-3]] = data
                                                
            #         label_data = _label_dict[res[:-3]] + '_Mises'
            #         data = data.vonMises()
                        
            #     elif res in ['pkii_pc', 'pk2_pc', 'kirchoff_pc', 'kirchhoff_pc', 'cauchy_pc', 'stress_pc', 
            #                   'pkii_pdir1', 'pk2_pdir1', 'kirchoff_pdir1', 'kirchhoff_pdir1', 'cauchy_pdir1', 'stress_pdir1',
            #                   'pkii_pdir2', 'pk2_pdir2', 'kirchoff_pdir2', 'kirchhoff_pdir2', 'cauchy_pdir2', 'stress_pdir2',
            #                   'pkii_pdir3', 'pk2_pdir3', 'kirchoff_pdir3', 'kirchhoff_pdir3', 'cauchy_pdir3', 'stress_pdir3']:
            #         #stress principal component
            #         if res[-3:] == '_pc': measure_type = res[:-3]
            #         else: measure_type = res[:-6]
                    
            #         if  measure_type+'_principal' in data_sav:
            #             data = data_sav[measure_type+'_principal']
                        
            #         elif measure_type in data_sav: 
            #             data = data_sav[measure_type]
            #             data = data.GetPrincipalStress()
            #             data_sav[measure_type+'_principal'] = data
                        
            #         else:                    
            #             if measure_type in ['pkii','pk2']:
            #                 data = material.GetPKII()
            #             elif measure_type in ['stress']:
            #                 data = material.GetStress(position = position)
            #             elif measure_type in ['kirchoff','kirchhoff']:
            #                 data = material.GetKirchhoff()
            #             elif measure_type == 'cauchy':
            #                 data = material.GetCauchy()
                        
            #             data = data.Convert(assemb, None, output_type)            
            #             data_sav[measure_type] = data                        
            #             data = data.GetPrincipalStress()
            #             data_sav[measure_type+'_principal'] = data
                    
            #         if res[-3:] == '_pc': #principal component
            #             data = data[0] #principal component
            #             label_data = _label_dict[measure_type] + '_Principal'
            #         elif res[-6:] == '_pdir1': #1st principal direction    
            #             data = data[1][0]
            #             label_data = _label_dict[measure_type] + '_PrincipalDir1'
            #         elif res[-6:] == '_pdir2': #2nd principal direction    
            #             data = data[1][2]
            #             label_data = _label_dict[measure_type] + '_PrincipalDir2'
            #         elif res[-6:] == '_pdir3': #3rd principal direction    
            #             data = data[1][2]
            #             label_data = _label_dict[measure_type] + '_PrincipalDir3'
                        
                    
            #     elif res in ['statev']:
            #         data = material.GetStatev().T                    
            #         data = assemb.ConvertData(data, None, output_type)
            #         label_data = 'State_Variables'
                
            #     elif res == 'internal_force':
            #         data = assemb.GetInternalForces(pb.GetDoFSolution(), 'local')
            #         data = assemb.ConvertData(data, None, output_type)
            #         label_data = 'Internal_Load_localCoord'   
                    
            #     elif res == 'internal_force_global':
            #         data = assemb.GetInternalForces(pb.GetDoFSolution(), 'global')
            #         data = assemb.ConvertData(data, None, output_type)
            #         label_data = 'Internal_Load_globalCoord' 
            
            for label_data, data in res.items():
                if file_format in ['vtk', 'msh']:
                    if output_type == 'Node':
                        OUT.addNodeData(data,label_data)  
                    elif output_type == 'Element':
                        OUT.addElmData(data,label_data)   
                    else: 
                        raise NameError("The specified " + str(output_type) + " can't be exported as vtk data")
                        
                elif file_format == 'txt':
                    #save array in txt file using numpy.savetxt
                    fname = filename + '_' + label_data + '_' + output_type + filename_compl + '.txt'
                    np.savetxt(fname, data)

                elif file_format == 'npy':
                    #save array in npy file (binary file generated by numpy) using numpy.save
                    fname = filename + '_' + label_data + '_' + output_type + filename_compl + '.npy'
                    np.save(fname, data)

                elif file_format in ['npz', 'npz_compressed']:
                    #save all arrays for one iteration in a npz file (binary file generated by numpy) using numpy.savez
                    var_name = label_data + '_' + output_type                    
                    OUT[var_name] = data
        
            if file_format in ['vtk', 'msh', 'npz', 'npz_compressed']:    
                for i, OUT in enumerate(list_ExportData):                
                    if file_format == 'vtk': OUT.toVTK(list_filename[i])
                    elif file_format == 'msh': OUT.toMSH(list_filename[i])
                    elif file_format == 'npz': np.savez(list_filename[i], **OUT)
                    elif file_format == 'npz_compressed': np.savez_compressed(list_filename[i], **OUT)
            

