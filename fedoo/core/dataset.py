# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 10:55:37 2022

@author: Etienne
"""
import numpy as np
import os
from fedoo.core.mesh import Mesh


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
        
    
    def __getitem__(self,items):
        if isinstance(items, tuple):             
            return self.get_data(*items)
        else:
            return self.get_data(items)
    
            
    def add_data(self, data_set):
        """
        Update the DataSet object including all the node, element and gausspoint
        data from antoher DataSet object data_set. The associated mesh is not 
        modified. 
        """
        self.node_data.update(data_set.node_data)
        self.element_data.update(data_set.element_data)
        self.gausspoint_data.update(data_set.gausspoint_data)
                            
    
    def _build_mesh_gp(self):
        #define a new mesh for the plot to gauss point (duplicate nodes between element)
        crd = self.mesh.nodes
        elm = self.mesh.elements
        nodes_gp = crd[elm.ravel()]
        element_gp = np.arange(elm.shape[0]*elm.shape[1]).reshape(-1,elm.shape[1])
        self.mesh_gp = self.mesh.__class__(nodes_gp, element_gp, self.mesh.elm_type)
        self.meshplot_gp = self.mesh_gp.to_pyvista()    
        
    
    def plot(self, field = None, data_type = None, **kargs):
        
        if self.mesh is None: 
            raise NameError("Can't generate a plot without an associated mesh. Set the mesh attribute first.")
        
        ndim = self.mesh.ndim
        
        scalars = kargs.pop('scalars', field) #kargs scalars can be used instead of field
        scalars, data_type = self.get_data(scalars, data_type, True)
    
        component = kargs.pop('component', 0)
        scale = kargs.pop('scale', 1)
        
        show = kargs.pop('show', True)
        show_edges = kargs.pop('show_edges', True)
        sargs=kargs.pop('scalar_bar_args', None) 
        
        if data_type == 'GaussPoint':
            if self.meshplot_gp is None:
                self._build_mesh_gp()
            meshplot = self.meshplot_gp
            crd = self.mesh_gp.nodes      
                
            scalars = self.mesh_gp.convert_data(scalars, convert_from='GaussPoint', convert_to='Node', n_elm_gp=len(scalars.T)//self.mesh.n_elements)
            if 'Disp' in self.node_data:
                ndim = self.mesh.ndim
                U = ((self.node_data['Disp'].reshape(ndim,-1).T[self.mesh.elements.ravel()]).T).T
                # meshplot.point_data['Disp'] = U  
                                                                    
        elif scalars is not None:
            if self.meshplot is None: 
                meshplot = self.meshplot = self.mesh.to_pyvista()
            else: 
                meshplot = self.meshplot                                    
            crd = self.mesh.nodes            
            
            if 'Disp' in self.node_data:
                U = self.node_data['Disp'].T
                #     meshplot.point_data['Disp'] = self.node_data['Disp'].T          
               
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

        if crd.shape[1] < 3: 
            crd = np.c_[crd, np.zeros((len(crd), 3-crd.shape[1]))]
            
        if 'Disp' in self.node_data:
            if U.shape[1] < 3: 
                U = np.c_[U, np.zeros((len(U), 3-U.shape[1]))]
            meshplot.points = crd + scale*U  
        else: 
            meshplot.points = crd

        #camera position
        meshplot.ComputeBounds()
        center = meshplot.center

        pl.camera.SetFocalPoint(center)
        pl.camera.position = tuple(center+np.array([0,0,2*meshplot.length]))
        pl.camera.up = tuple([0,1,0]) 
        
        if ndim == 3:
            pl.camera.Azimuth(30)
            pl.camera.Elevation(15)

        if component == "norm": 
            component = 0
            scalars = np.linalg.norm(scalars, axis = 0)

        pl.add_mesh(meshplot, scalars = scalars.T, component = component, show_edges = show_edges, scalar_bar_args=sargs, cmap="jet", **kargs)
            
        pl.add_axes(color='Black', interactive = True)
        
        pl.add_text(f"{field}_{component}", name='name', color='Black')

        if show: 
            return pl.show(return_cpos = True)
        else:
            return pl
       
        # cpos = pl.show(interactive = False, auto_close=False, return_cpos = True)
        # pl.save_graphic('test.pdf', title='PyVista Export', raster=True, painter=True)
        
    def get_data(self, field, data_type=None, return_data_type = False):       
        dict_data_type = {'Node':self.node_data, 'Element':self.element_data, 'GaussPoint':self.gausspoint_data}
        if data_type is None: #search if field exist somewhere 
            if field in self.node_data: 
                data_type = 'Node'                
            elif field in self.element_data: 
                data_type = 'Element'
            elif field in self.gausspoint_data:
                data_type = 'GaussPoint'
            else: 
                raise NameError("Field data not found.")
            data = dict_data_type[data_type][field]
        else: 
            if field in dict_data_type[data_type]:
                data = dict_data_type[data_type][field]
            else: #if field is not present whith the given data_type search if it exist elsewhere and convert it
                data, current_data_type = self.get_data(field, return_data_type = True)
                data = self.mesh.convert_data(data, convert_from = current_data_type, convert_to = data_type)
        if return_data_type: 
            return data, data_type
        else:
            return data
    
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
        name, ext = os.path.splitext(filename)
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
        name, ext = os.path.splitext(filename)
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
            self.node_data = {k: v.T for k,v in data.point_data.items()}
            self.element_data = {k: v.T for k,v in data.cell_data.items()} 
            if load_mesh: Mesh.from_pyvista(data)
        elif isinstance(data, str):
            #load from a file
            filename = data
            name, ext = os.path.splitext(filename)
            ext = ext.lower()
            if ext == '.vtk': 
                #load_mesh ignored because the mesh already in the vtk file
                DataSet.load(self,pv.read(filename))              
            elif ext == '.msh':
                return NotImplemented
            elif ext in ['.npz', '.npz_compressed']:
                if load_mesh: 
                    self.mesh = Mesh.read(os.path.splitext(filename)[0]+'.vtk')
                
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
    
    
    def __getitem__(self,items):
        if self.loaded_iter is None:
            self.load()
        return DataSet.__getitem__(self,items)
    
    def save_all(self, filename, file_format='npz'):
        """Save all data from MultiFrameDataSet. 
        If filename has no extension, a filename dir is created which contains 
        the data files for each iteration (npz format by default).
        if filename has an extension, the data files are saved using the given filename and format
        simply adding the iteration number. 
        The mesh is also saved in vtk format in the same directory.
        """
        dirname = os.path.dirname(filename)        
        extension = os.path.splitext(filename)[1]
        if extension == '': 
            dirname = filename+'/'
            filename = dirname+os.path.basename(filename)
            file_format = file_format.lower()
        else: 
            #use extension as file format
            file_format = extension[1:].lower()
            filename = os.path.splitext(filename)[0] #remove extension for the base name
            
        if not(os.path.isdir(dirname)): os.mkdir(dirname)
        for i in range(len(self.list_data)):
            self.load(i)
            self.save(filename + '_' +str(i) + '.' + file_format)
        self.save_mesh(filename + '.vtk')        
        
    
    def load(self,data=-1, load_mesh = False): 
        if isinstance(data, int):         
            #data is the an iteration to load
            iteration = self.list_data.index(self.list_data[data])
            if self.loaded_iter == iteration: 
                return
            if iteration > len(self.list_data): 
                raise NameError("Number of iteration exeed the total number of registered data ({})".format(len(self.list_data)))
            
            DataSet.load(self, self.list_data[iteration])
            self.loaded_iter = iteration
        
        else: 
            DataSet.load(self, data,load_mesh)

    
    def plot(self, field = None, **kargs):                
        iteration = kargs.pop('iteration', None)
        if iteration is None: 
            if self.loaded_iter is None: 
                self.load(-1) #load last iteration
        else: 
            self.load(iteration)
                
        return DataSet.plot(self, field, **kargs)
        
        
    def write_movie(self, filename='test', field = None, data_type = None, **kargs):
        """        
        Generate a video of the MultiFrameDataSet object by loading iteratively every frame.

        Parameters
        ----------
        filename : str
            Name of the videofile to write. 
        field : str
            Name of the field to plot
        data_type : str in {'Node', 'Element' or 'GaussPoint'}, optional
            Type of the data. By default, the data_type is determined automatically by scanning the data arrays. 
        
        Many options are available as keyword args.Some of these options are 
        directly related to pyvista options (for instance in the pyvista.plotter.add_mesh method). 
        Please, refer to the documentation of pyvista for more details.
        
        Available keyword arguments are :         
        * component : int (default = 0)
            The data component to plot in case of vector data   
        * framerate : int (default = 24)
            Number of frames per second
        * quality : int between 1 and 10 (default = 5)
            Define the quality of the writen movie. Higher is better but take more place. 
        * rot_azimuth : scalar (default = 0)
            Angle of azimuth rotation that is made at each new frame. Used to make easy video with moving camera.            
        * rot_elevation :  scalar (default = 0)
            Angle of elevation rotation that is made at each new frame. Used to make easy video with moving camera.            
        * scalar_bar_args' : 
            dict containing the arguments related to scalar bar.             
        * scale : scalar (default = 1)
            The scale used for the nodes displacement, using the 'Disp' vector field
        * show_edges (default = True)
            if True, the mesh edges are shown
        * window_size : list of int (default = [1024, 768])
            Size of the video in pixel            
        """
    
        if self.mesh is None: 
            raise NameError("Can't generate a plot without an associated mesh. Set the mesh attribute first.")
        
        ndim = self.mesh.ndim
          
        scalars = kargs.pop('scalars', field)
        
        framerate = kargs.pop('framerate', 24)
        quality = kargs.pop('quality', 5)
               
        component = kargs.pop('component', 0)
        scale = kargs.pop('scale', 1)

        show_edges = kargs.pop('show_edges', True)
        sargs=kargs.pop('scalar_bar_args', None) 
        
        rot_azimuth = kargs.pop('rot_azimuth', 0)
        rot_elevation = kargs.pop('rot_elevation', 0)
    
        #auto compute boundary        
        Xmin, Xmax, clim = self.get_all_frame_lim(scalars, component, data_type, scale)
        center = (Xmin+Xmax)/2
        length = np.linalg.norm(Xmax-Xmin)
            
        clim = kargs.pop('clim', clim)    
        window_size = kargs.pop('window_size', [1024, 768])

        component_save = component 
    
        self.load(0)
        
        data, data_type = self.get_data(scalars, data_type, True)
        
        if data_type == 'GaussPoint':
            if self.meshplot_gp is None:
                self._build_mesh_gp()
            meshplot = self.meshplot_gp         
            crd = self.mesh_gp.nodes             
        
        elif scalars is not None:
            if self.meshplot is None: 
                meshplot = self.meshplot = self.mesh.to_pyvista()
            else: 
                meshplot = self.meshplot
            crd = self.mesh.nodes 
        else:
            return NotImplemented
       
        pl = pv.Plotter(window_size=window_size)
            
        
        # pl = pv.Plotter()
        pl.set_background('White')
        pl.camera.SetFocalPoint(self.mesh.bounding_box.center)
        # pl.camera.position = (-2.090457552750125, 1.7582929402632352, 1.707926514944027)
        
        if sargs is None: #default value
            sargs = dict(
                interactive=True,
                title_font_size=20,
                label_font_size=16,
                color='Black',
                # n_colors= 10
            )

        pl.add_axes(color='Black', interactive = True)
                        
        pl.open_movie(filename+'.mp4', framerate=framerate, quality = quality)
        

        # pl.show(auto_close=False)  # only necessary for an off-screen movie

        #camera position       
        pl.camera.SetFocalPoint(center)
        pl.camera.position = tuple(center+np.array([0,0,2*length]))
        pl.camera.up = tuple([0,1,0]) 
        
        if ndim == 3:
            pl.camera.Azimuth(30)
            pl.camera.Elevation(15)


        for i in range(0,self.n_iter):
            self.load(i)
            data = self.get_data(scalars, data_type)
            
            if data_type == 'GaussPoint':                
                data = self.mesh_gp.convert_data(data, convert_from='GaussPoint', convert_to='Node', n_elm_gp=len(data.T)//self.mesh.n_elements).T
                if 'Disp' in self.node_data:                    
                    U = ((self.node_data['Disp'].reshape(ndim,-1).T[self.mesh.elements.ravel()]).T).T
                    meshplot.points = crd + scale*U  
            else:            
                if 'Disp' in self.node_data:
                    meshplot.points = crd + scale*self.node_data['Disp'].T

            if component_save == "norm":
                component = 0
                scalars = np.linalg.norm(scalars, axis = 0)                     
                
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
    
            if rot_azimuth: 
                pl.camera.Azimuth(rot_azimuth)
            if rot_elevation:
                pl.camera.Elevation(rot_elevation)

            # Run through each frame
            # pl.add_text(f"Iteration: {i}", name='time-label', color='Black')
            pl.write_frame() 
            
            
            
        pl.close()
        self.meshplot = None
        
        
    def get_all_frame_lim(self, field, component=0, data_type = None, scale = 1):
                
        ndim = self.mesh.ndim
        clim = [np.inf, -np.inf]
        crd = self.mesh.nodes
        
        for i in range(0,self.n_iter):                                   
            self.load(i)
            
            data = self.get_data(field, data_type)[component]
            clim = [np.min([data.min(),clim[0]]), np.max([data.max(), clim[1]])]           

            if 'Disp' in self.node_data:
                new_crd = crd + scale*self.node_data['Disp'].T
                        
                new_Xmin = new_crd.min(axis = 0) ; new_Xmax = new_crd.max(axis=0)
                if i == 0:
                    Xmin = new_Xmin ; Xmax = new_Xmax
                else:
                    Xmin = [np.min([Xmin[i], new_Xmin[i]]) for i in range(ndim)]
                    Xmax = [np.max([Xmax[i], new_Xmax[i]]) for i in range(ndim)]
            
        if 'Disp' not in self.node_data:
            Xmin = self.mesh.bounding_box[0]
            Xmax = self.mesh.bounding_box[1]
            
        return np.array(Xmin), np.array(Xmax), clim

        
    
    @property
    def n_iter(self):
        return len(self.list_data)


def read_data(filename, file_format="npz"):
    
    dirname = os.path.dirname(filename)        
    extension = os.path.splitext(filename)[1]
    if extension == '': 
        dirname = filename+'/'
        filename = dirname+os.path.basename(filename)
        file_format = file_format.lower()
    else: 
        #use extension as file format
        file_format = extension[1:].lower()
        filename = os.path.splitext(filename)[0] #remove extension for the base name
        
    assert (os.path.isdir(dirname)), "File not found"
    
    if os.path.isfile(filename+'.vtk'): 
        mesh = Mesh.read(filename+'.vtk')
    else: 
        mesh = None
    
    if os.path.isfile(filename+'.'+file_format): 
        dataset = DataSet(mesh)
        dataset.load(filename+'.'+file_format)
        return dataset
    if os.path.isfile(filename+'_0.'+file_format): iter0= 0
    elif os.path.isfile(filename+'_0.'+file_format): iter0= 1
    else: raise NameError("File not found") 
        
    dataset = MultiFrameDataSet(mesh)
    i = iter0
    while os.path.isfile(filename+'_'+str(i)+'.'+file_format):
        dataset.list_data.append(filename+'_'+str(i)+'.'+file_format)
        i+=1
        
    return dataset
    
    


