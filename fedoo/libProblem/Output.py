# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 17:54:05 2021
@author: Etienne
"""

import numpy as np
from fedoo.mesh.mesh import *
from fedoo.assembly.AssemblyBase import AssemblyBase
from fedoo.utilities.ExportData import ExportData 
from fedoo.utilities.dataset import MultiFrameDataSet
import os

_available_output = ['PKII',   'PK2',   'Kirchoff',   'Kirchhoff',   'Cauchy',
                     'PKII_vm','PK2_vm','Krichoff_vm','Kirchhoff_vm','Cauchy_vm',
                     'PKII_pc', 'PK2_pc', 'Kirchoff_pc', 'Kirchhoff_pc', 'Cauchy_pc', 'Stress_pc',
                     'PKII_pdir1', 'PK2_pdir1', 'Kirchoff_pdir1', 'Kirchhoff_pdir1', 'Cauchy_pdir1', 'Stress_pdir1',
                     'PKII_pdir2', 'PK2_pdir2', 'Kirchoff_pdir2', 'Kirchhoff_pdir2', 'Cauchy_pdir2', 'Stress_pdir2',
                     'PKII_pdir3', 'PK2_pdir3', 'Kirchoff_pdir3', 'Kirchhoff_pdir3', 'Cauchy_pdir3', 'Stress_pdir3',
                     'Disp', 'Rot', 'Temp', 'Strain', 'Statev', 'Stress', 'Stress_vm', 'Fext', 
                     'Wm', 'Fint', 'Fint_global']

_available_format = ['vtk', 'msh', 'txt', 'npy', 'npz', 'npz_compressed']

_label_dict = {'pkii':'PKII',   'pk2':'PK2',   'kirchoff':'Kirchhoff', 'kirchhoff':'Kirchhoff', 'cauchy':'Cauchy',
                'pkii_vm':'PKII_vm','pk2_vm':'PK2_vm','kirchoff_vm':'Kirchhoff_vm','kirchhoff_vm':'Kirchhoff_vm','cauchy_vm':'Cauchy_vm',
                'pkii_pc':'PKII_pc', 'pk2_pc':'PK2_pc', 'kirchoff_pc':'Kirchhoff_pc', 'kirchhoff_pc':'Kirchhoff_pc', 'cauchy_pc':'Cauchy_pc', 'stress_pc':'Stress_pc',
                'pkii_pdir1':'PKII_pdir1', 'pk2_pdir1':'PK2_pdir1', 'kirchoff_pdir1':'Kirchhoff_pdir1', 'kirchhoff_pdir1':'Kirchhoff_pdir1', 'cauchy_pdir1':'Cauchy_pdir1', 'stress_pdir1':'Stress_pdir1',
                'pkii_pdir2':'PKII_pdir2', 'pk2_pdir2':'PK2_pdir2', 'kirchoff_pdir2':'Kirchhoff_pdir2', 'kirchhoff_pdir2':'Kirchhoff_pdir2', 'cauchy_pdir2':'Cauchy_pdir2', 'stress_pdir2':'Stress_pdir2',
                'pkii_pdir3':'PKII_pdir3', 'pk2_pdir3':'PK2_pdir3', 'kirchoff_pdir3':'Kirchhoff_pdir3', 'kirchhoff_pdir3':'Kirchhoff_pdir3', 'cauchy_pdir3':'Cauchy_pdir1', 'stress_pdir3':'Stress_pdir3',
                'disp':'Disp', 'rot':'Rot', 'temp':'Temp', 'strain':'Strain', 'statev':'Statev', 'stress':'Stress', 'stress_vm':'Stress_vm', 'fext':'Fext', 
                'wm':'Wm', 'fint':'Fint', 'fint_global':'Fint_global' 
}

#dict to get the str used in variable name for each output_type
_output_type_str = {'Node': 'nd', 'Element':'el', 'GaussPoint':'gp'}

#  {'pkii':'PKII', 'pk2':'PKII', 'kirchoff':'Kirchhoff', 'kirchhoff':'Kirchhoff', 'cauchy':'Cauchy',
# 'stress':'Stress', 'strain':'Strain', 'disp':'Disp', 'rot':'Rot'} #use to get label associated with some outputs

def _GetResults(pb, assemb, output_list, output_type='Node', position = 1, res_format = None):
        
        if isinstance(output_list, str): output_list = [output_list]                

        if output_type.lower() == 'node': output_type = 'Node'
        elif output_type.lower() == 'element': output_type = 'Element'
        elif output_type.lower() == 'gausspoint': output_type = 'GaussPoint'
        else: raise NameError("output_type should be either 'Node', 'Element' or 'GaussPoint'")
                
        for i,res in enumerate(output_list):
            output_list[i] = _label_dict[res.lower()] #to allow full lower case str as output
            if res not in _available_output:
                print("WARNING: '", res, "' doens't match to any available output")
                print("Specified output ignored")
                print("List of available output: ", _available_output)
        
        data_sav = {} #dict to keep data in memory that may be used more that one time

        if isinstance(assemb, str): 
            assemb = AssemblyBase.get_all()[assemb]  

        if hasattr(assemb, 'list_assembly'): #AssemblySum object
            if assemb.assembly_output is None:
                raise NameError("AssemblySum objects can't be used to extract outputs")
            else:
                assemb = assemb.assembly_output
                
        material = assemb.weakform.GetConstitutiveLaw()
        
        result = {}
                    
        for res in output_list:                                
            if res in ['PKII', 'PK2', 'Kirchhoff', 'Cauchy','Strain', 'Stress']:
                if res in data_sav: 
                    data = data_sav[res] #avoid a new data conversion
                else:
                    if res in ['PKII','PK2']:
                        data = material.GetPKII()
                    elif res == 'Stress':
                        #stress for small displacement
                        data = material.GetStress(position = position)
                    elif res == 'Kirchhoff':
                        data = material.GetKirchhoff()
                    elif res == 'Cauchy':
                        data = material.GetCauchy()
                    elif res == 'Strain':
                        data = material.GetStrain(position = position)                                                
                    
                    data = data.Convert(assemb, None, output_type)                        
                    
                    #keep data in memory in case it may be used later for vm, pc or pdir stress computation
                    data_sav[res] = data
                
                if res_format == 'vtk': data = data.vtkFormat()
                elif res_format in ['msh', 'txt']: data = np.array(data).T
                elif res_format in ['npy', 'npz']: data = np.array(data)
                                        
            elif res == 'Disp':
                if output_type == 'Node':
                    if res_format in ['vtk', 'msh', 'txt']:
                        data = pb.GetDisp().T
                    else: 
                        data = pb.GetDisp()
                else: 
                    raise NameError("Displacement is only a Node data and is incompatible with the output format specified")                    

            elif res == 'Rot':
                if output_type == 'Node': 
                    if res_format in ['vtk', 'msh', 'txt']:                            
                        data = pb.GetRot().T
                    else: 
                        data = pb.GetRot()
                else: 
                    raise NameError("Displacement is only a Node data and is incompatible with the output format specified")                    
            
            elif res == 'Temp':
                if output_type == 'Node': 
                    data = pb.GetTemp().T
                else: 
                    raise NameError("Temperature is only a Node data and is incompatible with the output format specified")                    
                
            elif res == 'Fext':
                if output_type == 'Node':                             
                    data = assemb.get_ext_forces(pb.GetDoFSolution())
                else: 
                    raise NameError("External_Force is only Node data and is incompatible with the specified output format")    

            elif res in ['PKII_vm', 'PK2_vm', 'Kirchhoff_vm', 'Cauchy_vm', 'Stress_vm']:
                if res[:-3] in data_sav: 
                    data=data_sav[res[:-3]]
                else:
                    if res in ['PKII_vm','PK2_vm']:
                        data = material.GetPKII()
                    elif res == 'Stress_vm':
                        data = material.GetStress(position = position)
                    elif res == 'Kirchhoff_vm':
                        data = material.GetKirchhoff()
                    elif res == 'Cauchy_vm':
                        data = material.GetCauchy()

                    data = assemb.convert_data(data, None, output_type)
                    data_sav[res[:-3]] = data
                                            
                data = data.vonMises()
                    
            elif res in ['PKII_pc', 'PK2_pc', 'Kirchhoff_pc', 'Cauchy_pc', 'Stress_pc', 
                         'PKII_pdir1', 'PK2_pdir1', 'Kirchhoff_pdir1', 'Cauchy_pdir1', 'Stress_pdir1',
                         'PKII_pdir2', 'PK2_pdir2', 'Kirchhoff_pdir2', 'Cauchy_pdir2', 'Stress_pdir2',
                         'PKII_pdir3', 'PK2_pdir3', 'Kirchhoff_pdir3', 'Cauchy_pdir3', 'Stress_pdir3']:
                #stress principal component
                if res[-3:] == '_pc': measure_type = res[:-3]
                else: measure_type = res[:-6]
                
                if  measure_type+'_pc' in data_sav:
                    data = data_sav[measure_type+'_pc']
                    
                elif measure_type in data_sav: 
                    data = data_sav[measure_type]
                    data = data.GetPrincipalStress()
                    data_sav[measure_type+'_pc'] = data
                    
                else:                    
                    if measure_type in ['PKII','PK2']:
                        data = material.GetPKII()
                    elif measure_type == 'Stress':
                        data = material.GetStress(position = position)
                    elif measure_type == 'Kirchhoff':
                        data = material.GetKirchhoff()
                    elif measure_type == 'Cauchy':
                        data = material.GetCauchy()
                    
                    data = data.Convert(assemb, None, output_type)            
                    data_sav[measure_type] = data                        
                    data = data.GetPrincipalStress()
                    data_sav[measure_type+'_pc'] = data
                
                if res[-3:] == '_pc': #principal component
                    data = data[0] #principal component                    
                elif res[-6:] == '_pdir1': #1st principal direction    
                    data = data[1][0]                    
                elif res[-6:] == '_pdir2': #2nd principal direction    
                    data = data[1][1]                    
                elif res[-6:] == '_pdir3': #3rd principal direction    
                    data = data[1][2]                    
                    
            elif res == 'Statev':
                data = material.GetStatev().T                    
                data = assemb.convert_data(data, None, output_type)
            
            elif res in ['Wm']:
                data = material.GetWm().T                    
                data = assemb.convert_data(data, None, output_type)
            
            elif res == 'Fint':
                data = assemb.get_int_forces(pb.GetDoFSolution(), 'local')
                data = assemb.convert_data(data, None, output_type)
                
            elif res == 'Fint_global':
                data = assemb.get_int_forces(pb.GetDoFSolution(), 'global')
                data = assemb.convert_data(data, None, output_type)
            
            result[res] = data
        
        return result
        



class _ProblemOutput:
    def __init__(self):
        self.__list_output = [] #a list containint dictionnary with defined output
        self.data_sets = {}
                
    def AddOutput(self, filename, assemb, output_list, output_type='Node', file_format = 'npz', position = 1, save_mesh = True):
        
        dirname = os.path.dirname(filename)        
        # filename = os.path.basename(filename)
        extension = os.path.splitext(filename)[1]
        if extension == '': 
            #if no extention -> create a new dir using filename as dirname
            dirname = filename+'/'
            filename = dirname+os.path.basename(filename)
            file_format = file_format.lower()
        else: 
            #use extension as file format
            file_format = extension[1:].lower()
            filename = os.path.splitext(filename)[0] #remove extension for the base name
             
        
        if file_format not in _available_format:
            print("WARNING: '", file_format, "' doens't match to any available file format")
            print("Specified output ignored")
            print("List of available file format: ", _available_format)
        
        if output_type.lower() not in ['node', 'element', 'gausspoint']:
            raise NameError("output_type should be either 'Node', 'Element' or 'GaussPoint'")
                
        for i,res in enumerate(output_list):
            output_list[i] = _label_dict[res.lower()] #to allow full lower case str as output
            if res not in _available_output:
                print("WARNING: '", res, "' doens't match to any available output")
                print("Specified output ignored")
                print("List of available output: ", _available_output)
        
        if isinstance(assemb, str): assemb = AssemblyBase.get_all()[assemb]     
        
        if not(os.path.isdir(dirname)): os.mkdir(dirname)
                
        new_output = {'filename': filename, 'assembly': assemb, 'type': output_type, 'list': output_list, 'file_format': file_format.lower(), 'position': position}
        self.__list_output.append(new_output)
        
        if save_mesh:
            assemb.mesh.save(filename)
        
        if file_format in ['npz', 'npz_compressed']:
            if not(filename in self.data_sets):
                res = MultiFrameDataSet(assemb.mesh, [])
                self.data_sets[filename] = res
            else:
                res = self.data_sets[filename]
            return res

    def SaveResults(self, pb, comp_output=None):
        
        list_filename = []
        list_full_filename = []
        list_file_format = []
        list_data = []            
        
        for output in self.__list_output:
            
            filename = output['filename']
            file_format = output['file_format'].lower()
            output_type = output['type'] #'Node', 'Element' or 'GaussPoint'
            position = output['position']                              
            
            assemb = output['assembly']
            # material = assemb.weakform.GetConstitutiveLaw()
            
            if comp_output is None:
                filename_compl = ""
            else: 
                filename_compl = '_' + str(comp_output)
            
            if file_format in ['vtk', 'msh', 'npz', 'npz_compressed']:                
                # filename = filename + filename_compl + '.' + file_format[0:3]
                full_filename = filename + filename_compl + '.' + file_format #filename including iter number and file format
                
                if not(full_filename in list_full_filename): 
                    #if filename don't exist in the list we create it
                    list_filename.append(filename)
                    list_full_filename.append(full_filename)
                    list_file_format.append(file_format)
                    if file_format in ['vtk', 'msh']: OUT = ExportData(assemb.mesh.name)
                    else: OUT = {} #empty dictionnary containing variable                        
                    list_data.append(OUT)                        
                else: 
                    #else, the same file is used   
                    # if file_format in ['vtk', 'msh']: 
                    OUT = list_data[list_full_filename.index(full_filename)]                                                                             
            
            #compute the results
            res = _GetResults(pb, assemb, output['list'],output_type,position, file_format)                        
            
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
                    fname = filename + '_' + label_data + '_' + _output_type_str[output_type] + filename_compl + '.txt'
                    np.savetxt(fname, data)

                elif file_format == 'npy':
                    #save array in npy file (binary file generated by numpy) using numpy.save
                    fname = filename + '_' + label_data + '_' + _output_type_str[output_type] + filename_compl + '.npy'
                    np.save(fname, data)

                elif file_format in ['npz', 'npz_compressed']:
                    #save all arrays for one iteration in a npz file (binary file generated by numpy) using numpy.savez
                    var_name = label_data + '_' + _output_type_str[output_type]                    
                    OUT[var_name] = data
        
        for i, OUT in enumerate(list_data): 
            file_format = list_file_format[i]               
            if file_format == 'vtk': OUT.toVTK(list_full_filename[i])
            elif file_format == 'msh': OUT.toMSH(list_full_filename[i])
            elif file_format == 'npz': 
                np.savez(list_full_filename[i], **OUT)
                self.data_sets[list_filename[i]].list_data.append(list_full_filename[i])
                
            elif file_format == 'npz_compressed': 
                np.savez_compressed(list_filename[i], **OUT)
                i = self.data_sets.data_file.index(list_filename[i])
                self.data_sets[list_filename[i]].list_data.append(list_full_filename[i])                
                
            
           # if output_type.lower() == 'node': output_type = 'nd'
           # elif output_type.lower() == 'element': output_type = 'el'
           # elif output_type.lower() == 'gausspoint': output_type = 'gp'             