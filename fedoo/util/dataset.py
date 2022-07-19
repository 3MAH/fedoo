# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 10:55:37 2022

@author: Etienne
"""
import numpy as np
from os.path import splitext

try: 
    import pyvista as pv
    USE_PYVISTA = True
except:
    USE_PYVISTA = False
    
try:
    import pandas
    USE_PANDA = True
except:
    USE_PANDA = False

class DataSet():
    
    def __init__(self,  mesh = None, data = None, data_type = 'node'):
        
        self.mesh = mesh
        self.node_data = {}
        self.element_data = {}
        self.gausspoint_data = {}
        
        if isinstance(data, dict):
            data_type = data_type.lower()        
            if data_type == 'node':
                self.node_data = data
            elif data_type == 'element':
                self.element_data = data
            elif data_type == 'gausspoint':
                self.gausspoint_data = data
            elif data_type == 'all':
                self.node_data = {k:v for k,v in data.items() if k[-2:] == 'nd'}
                self.element_data = {k:v for k,v in data.items() if k[-2:] == 'el'}
                self.gausspoint_data = {k:v for k,v in data.items() if k[-2:] == 'gp'}
            
        
        self.meshplot = None
        self.meshplot_gp = None #a mesh with discontinuity between each element to plot gauss points field
        
    
    def add_data(self, data_set):
        """
        Update the DataSet object including all the node, element and gausspoint
        data from antoher DataSet object data_set. The associated mesh is not 
        modified. 
        """
        self.node_data.update(data_set.node_data)
        self.element_data.update(data_set.element_data)
        self.gausspoint_data.update(data_set.gausspoint_data)
                            
    
    def plot(self, field = None, **kargs):
        
        if self.mesh is None: 
            raise NameError("Can't generate a plot without an associated mesh. Set the mesh attribute first.")
        
        scalar_type = kargs.pop('scalar_type', None)
        scalars = kargs.pop('scalars', field) #kargs scalars can be used instead of field
        scalars, scalar_type = self.get_scalars(scalars, scalar_type)
    
        component = kargs.pop('component', 0)
        scale = kargs.pop('scale', 1)
        
        show = kargs.pop('show', True)
        show_edges = kargs.pop('show_edges', True)
        sargs=kargs.pop('scalar_bar_args', None) 
        
        if scalar_type == 'gp':
            return NotImplemented
        elif scalars is not None:
            if self.meshplot is None: 
                meshplot = self.meshplot = self.mesh.to_pyvista()
            else: 
                meshplot = self.meshplot            
       
        pl = pv.Plotter()
        # pl = pv.Plotter()
        pl.set_background('White')
        
        if sargs is None: #default value
            sargs = dict(
                interactive=True,
                title_font_size=20,
                label_font_size=16,
                color='Black',
                # n_colors= 10
            )
        
        # cpos = [(-2.69293081283409, 0.4520024822911473, 2.322209100082263),
        #         (0.4698685969042552, 0.46863550630755524, 0.42428354242422084),
        #         (0.5129241539116808, 0.07216479580221505, 0.8553952621921701)]
        # pl.camera_position = cpos
        
        # pl.add_mesh(meshplot.warp_by_vector(factor = 5), scalars = 'Stress', component = 2, clim = [0,10000], show_edges = True, cmap="bwr")

        if 'Disp' in self.node_data:
            self.meshplot.point_data['Disp'] = self.node_data['Disp'].T  
            pl.add_mesh(meshplot.warp_by_vector('Disp', factor = scale), scalars = scalars.T, component = component, show_edges = show_edges, scalar_bar_args=sargs, cmap="jet", **kargs)
        else: 
            pl.add_mesh(meshplot, scalars = scalars.T, component = component, show_edges = show_edges, scalar_bar_args=sargs, cmap="jet", **kargs)
            
        pl.add_axes(color='Black', interactive = True)
        
        # cpos = pl.show(return_cpos = True)        
        if show: 
            return pl.show(return_cpos = True)
        else:
            return pl
        # cpos = pl.show(interactive = False, auto_close=False, return_cpos = True)
        # pl.save_graphic('test.pdf', title='PyVista Export', raster=True, painter=True)
        
    def get_scalars(self, scalars, scalar_type=None):
        if scalar_type is None: 
            if scalars in self.node_data: 
                scalar_type = 'nd'                
            elif scalars in self.element_data: 
                scalar_type = 'el'
            elif scalars in self.gausspoint_data:
                scalar_type = 'gp'
            else: 
                raise NameError("Scalars data not found.")                
        
        if scalar_type == 'nd':             
            data = self.node_data[scalars]            
        elif scalar_type == 'el':
            data = self.element_data[scalars]
        elif scalar_type == 'gp':
            data = self.gausspoint_data[scalars]  
        
        return data, scalar_type
  
    
    def save(self, filename, save_mesh = False):        
        """Save data to a file. 
        File type is inferred from the extension of the filename.                
        
        Parameters
        ----------
        filename : str
            Name of the file including the path.         
        save_mesh : bool (default = False)
            If True, the mesh is also saved in a vtk file using the same filename with a '.vtk' extention.
            For vtk and msh file, the mesh is always included in the file and save_mesh have no effect.
        """
        name, ext = splitext(filename)
        ext = ext.lower()
        if ext == '.vtk': 
            self.to_vtk(filename)
        elif ext == '.msh':
            self.to_msh(filename)
        else:
            if ext == '.npz':
                self.savez(filename,save_mesh)
            elif ext == '.npz_compressed':
                self.savez_compressed(filename, save_mesh)
            elif ext == '.csv':
                self.to_csv(filename, save_mesh)              
            elif ext == '.xlsx':
                self.to_excel(filename, save_mesh)  
            if save_mesh: 
                self.save_mesh(filename)
                
                    
    def save_mesh(self, filename):
        """Save the mesh using a vtk file. The extension of filename is ignored and modified to '.vtk'."""
        name, ext = splitext(filename)
        self.mesh.save(name)
        
   
    
    def load(self, data, load_mesh = False):
        """Load data from a data object. 
        The old data are erased.
                
        Parameters
        ----------
        data : 
        * if type(data) is dict: 
            load data using the load_dict method
        * if type(data) is DataSet:             
            load data from another DataSet object without copy 
        * if type(data) is pyvista.UnstructuredGrid
            load data from a pyvista UnstructuredGrid object without copy 
        * if type(data) is str
            load data from a file. Available extention are 'vtk', 'msh', 'npz' and
            'npz_compressed'            
        load_mesh : bool (default = False)
            If True, the mesh is loaded from the file (if the file contans a mesh). 
            If False, only the data are loaded.
        """
        if isinstance(data, dict):
            self.load_dict(data)
        elif isinstance(data, DataSet):
            self.node_data = data.node_data
            self.element_data = data.element_data
            self.gausspoint_data = data.gausspoint_data
            if load_mesh: self.mesh = data.mesh            
        elif isinstance(data, pv.UnstructuredGrid):
            self.meshplot = data
            self.node_data = data.point_data
            self.element_data = data.cell_data  
            if load_mesh: return NotImplemented
        elif isinstance(data, str):
            #load from a file
            filename = data
            name, ext = splitext(filename)
            ext = ext.lower()
            if ext == '.vtk': 
                #load_mesh ignored because the mesh already in the vtk file
                DataSet.load(self,pv.read(filename))              
            elif ext == '.msh':
                return NotImplemented
            elif ext in ['.npz', '.npz_compressed']:
                if load_mesh: 
                    return NotImplemented 
                data = np.load(filename)
                
                self.load_dict(data)
            elif ext == '.csv':
                return NotImplemented
            elif ext == '.xlsx':
                return NotImplemented     

            else:
               raise NameError("Can't load data -> Data not understood")                 
        else:
            raise NameError("Can't load data -> Data not understood")


    def load_dict(self, data):
        """Load data from a dict generated with the to_dict method.
        The old data are erased."""
        self.node_data = {k[:-3]:v for k,v in data.items() if k[-2:] == 'nd'}
        self.element_data = {k[:-3]:v for k,v in data.items() if k[-2:] == 'el'}
        self.gausspoint_data = {k[:-3]:v for k,v in data.items() if k[-2:] == 'gp'}       
            
                
    def to_pandas(self):
        if USE_PANDA:
            out = {}
            n_data_type = (self.node_data != {}) + (self.element_data != {}) + \
                          (self.gausspoint_data != {})
            if n_data_type > 1: 
                raise NameError("Can't convert to pandas DataSet with with several different data type.")                        
            
            for k, v in self.node_data.items():
                if len(v.shape)==1:
                    out[k] = v
                elif len(v.shape)==2:
                    out.update({k+'_'+str(i):v[i] for i in range(v.shape[0])})                
                else: 
                    return NotImplemented

            for k, v in self.element_data.items():
                if len(v.shape)==1:
                    out[k] = v
                elif len(v.shape)==2:
                    out.update({k+'_'+str(i):v[i] for i in range(v.shape[0])})                
                else: 
                    return NotImplemented
                
            for k, v in self.element_data.items():
                if len(v.shape)==1:
                    out[k] = v
                elif len(v.shape)==2:
                    out.update({k+'_'+str(i):v[i] for i in range(v.shape[0])})                
                else: 
                    return NotImplemented
                
            return pandas.DataFrame.from_dict(out)
        else: 
            raise NameError('Pandas lib is not installed.')        
        
    
    def to_csv(self, filename, save_mesh = False):
        """Write data in a csv file. 
        This method require the installation of pandas library 
        and is available only if 1 type of data (node, element, gausspoint) is defined. 
        
        Parameters
        ----------
        filename : str
            Name of the file including the path.
        save_mesh : bool (default = False)
            If True, the mesh is also saved in a vtk file using the same filename with a '.vtk' extention.
        """
        self.to_pandas().to_csv(filename)    
        if save_mesh: self.save_mesh(filename)      


    def to_excel(self, filename, save_mesh = False):
        """Write data in a xlsx file (excel format). 
        This method require the installation of pandas and openpyxl libraries
        and is available only if 1 type of data (node, element, gausspoint) is defined. 
        
        Parameters
        ----------
        filename : str
            Name of the file including the path.
        save_mesh : bool (default = False)
            If True, the mesh is also saved in a vtk file using the same filename with a '.vtk' extention.
        """
        self.to_pandas().to_excel(filename)          
        if save_mesh: self.save_mesh(filename)      

        
    def to_vtk(self, filename):
        """Write vtk file with the mesh and associated data (gausspoint data not included). 
        
        Parameters
        ----------
        filename : str
            Name of the file including the path.
        """
        from fedoo.util.mesh_writer import write_vtk
        write_vtk(self, filename)

        
    def to_msh(self, filename):
        """Write a msh (gmsh format) file with the mesh and associated data 
        (gausspoint data not included). 
        
        Parameters
        ----------
        filename : str
            Name of the file including the path.
        """
        from fedoo.util.mesh_writer import write_msh
        write_msh(self, filename)
        
    
    def to_dict(self):
        """Return a dict with all the node, element and gausspoint data."""
        out = {k+'_nd':v for k,v in self.node_data.items()}
        out.update({k+'_el':v for k,v in self.element_data.items()})
        out.update({k+'_gp':v for k,v in self.gausspoint_data.items()})
        return out


    def savez(self, filename, save_mesh = False): 
        """Write a npz file using the numpy savez function. 
                
        Parameters
        ----------
        filename : str
            Name of the file including the path.
        save_mesh : bool (default = False)
            If True, the mesh is also saved in a vtk file using the same filename with a '.vtk' extention.
        """
        np.savez(filename, **self.to_dict())
            
        if save_mesh: self.save_mesh(filename)
        
        
    def savez_compressed(self, filename, save_mesh = False):  
        """Write a compressed npz file using the numpy savez_compressed function. 
                
        Parameters
        ----------
        filename : str
            Name of the file including the path.
        save_mesh : bool (default = False)
            If True, the mesh is also saved in a vtk file using the same filename with a '.vtk' extention.
        """
        np.savez_compressed(filename, **self.to_dict())
        
        if save_mesh: self.save_mesh(filename)       
        

class MultiFrameDataSet(DataSet):
    
    def __init__(self, mesh = None, list_data = []):
        
        if not(isinstance(list_data, list)):
            list_data = [list_data]
        self.list_data = list_data                        
        self.loaded_iter = None
        
        DataSet.__init__(self,mesh)                        
    
    
    def load(self,iteration=-1):
        iteration = self.list_data.index(self.list_data[iteration])
        if self.loaded_iter == iteration: 
            return
        if iteration > len(self.list_data): 
            raise NameError("Number of iteration exeed the total number of registered data ({})".format(len(self.list_data)))
        
        DataSet.load(self, self.list_data[iteration])
        self.loaded_iter = iteration

    
    def plot(self, field = None, **kargs):                
        iteration = kargs.pop('iteration', None)
        if iteration is None: 
            if self.loaded_iter is None: 
                self.load(-1) #load last iteration
        else: 
            self.load(iteration)
                
        return DataSet.plot(self, field, **kargs)
        
        
    def write_movie(self, field = None, **kargs):
    
        if self.mesh is None: 
            raise NameError("Can't generate a plot without an associated mesh. Set the mesh attribute first.")
        
        crd = self.mesh.nodes 
        
        self.load(0)
        
        scalar_type = kargs.pop('scalar_type', None)
        scalars = kargs.pop('scalars', field)
        data, scalar_type = self.get_scalars(scalars, scalar_type)

        clim = kargs.pop('clim', None)                
       
        component = kargs.pop('component', 0)
        scale = kargs.pop('scale', 1)
        
        show_edges = kargs.pop('show_edges', True)
        sargs=kargs.pop('scalar_bar_args', None) 
        
        if scalar_type == 'gp':
            return NotImplemented
        elif scalars is not None:
            if self.meshplot is None: 
                meshplot = self.meshplot = self.mesh.to_pyvista()
            else: 
                meshplot = self.meshplot            
       
        window_size = kargs.pop('window_size', [1024, 768])
        pl = pv.Plotter(window_size=window_size)
            
        
        # pl = pv.Plotter()
        pl.set_background('White')
        
        if sargs is None: #default value
            sargs = dict(
                interactive=True,
                title_font_size=20,
                label_font_size=16,
                color='Black',
                # n_colors= 10
            )

        pl.add_axes(color='Black', interactive = True)
        
        framerate = kargs.pop('framerate', 24)
        quality = kargs.pop('quality', 5)
        
        filename = kargs.pop('filename', 'test')
                
        pl.open_movie(filename+'.mp4', framerate=framerate, quality = quality)
        

        # # pl.show(auto_close=False)  # only necessary for an off-screen movie
        # pl.camera.SetFocalPoint(center)
        # pl.camera.position = (-2.090457552750125, 1.7582929402632352, 1.707926514944027)

        for i in range(0,self.n_iter):
            self.load(i)
            data = self.get_scalars(scalars, scalar_type)[0]
            
            if 'Disp' in self.node_data:
                meshplot.points = crd + scale*self.node_data['Disp'].T
                     
            if i == 0:
                pl.add_mesh(meshplot, scalars = data.T, component = component, show_edges = show_edges, scalar_bar_args=sargs, cmap="jet", clim = clim,  **kargs)
            else:
                if clim is None: 
                    if len(data.shape)>1:
                        pl.update_scalar_bar_range([data[component].min(), data[component].max()])
                    else:
                        #scalar field -> component ignored
                        pl.update_scalar_bar_range([data.min(), data.max()])

                pl.update_scalars(data.T)
            
            

            # for res in [res_th, res_me]:
            #     for item in res:
            #         if item[-4:] == 'Node':
            #             if len(res[item]) == len(crd):
            #                 meshplot.point_data[item[:-5]] = res[item]
            #             else:
            #                 meshplot.point_data[item[:-5]] = res[item].T
            #         else:
            #             meshplot.cell_data[item] = res[item].T
                    
            # actor = pl.add_mesh(meshplot, scalars = 'data', show_edges = True, scalar_bar_args=sargs, cmap="bwr", clim = [0,100])
            # meshplot.points = crd + factor*meshplot.point_data['Disp']
            
            # if i == 0: 
            #     pl.add_mesh(meshplot, scalars = field_name, component = component, show_edges = True, scalar_bar_args=sargs, cmap="jet", clim = clim)
    
            # if clim is None: 
            #     pl.update_scalar_bar_range([meshplot.point_data[field_name].min(), meshplot.point_data[field_name].max()])
    
            # pl.camera.Azimuth(2*i/360*np.pi)
            # pl.camera.Azimuth(360/nb_iter)
            # Run through each frame
            # pl.add_text(f"Iteration: {i}", name='time-label', color='Black')
            pl.write_frame() 
            
        pl.close()
        self.meshplot = None
        
    @property
    def n_iter(self):
        return len(self.list_data)
        