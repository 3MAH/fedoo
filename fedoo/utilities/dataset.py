# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 10:55:37 2022

@author: Etienne
"""
import numpy as np
try: 
    import pyvista as pv
    USE_PYVISTA = True
except:
    USE_PYVISTA = False

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
        if isinstance(self.list_data[iteration], dict):
            data = self.list_data[iteration]
        elif isinstance(self.list_data[iteration], str):
            filename = self.list_data[iteration]
            if filename[-4:].lower() == '.npz' or filename[-15:].lower() == '.npz_compressed':
                data = np.load(filename)
            else: 
                raise NameError("Can't load data -> Data not understood")
        else:
            raise NameError("Can't load data -> Data not understood")
            
        self.node_data = {k[:-3]:v for k,v in data.items() if k[-2:] == 'nd'}
        self.element_data = {k[:-3]:v for k,v in data.items() if k[-2:] == 'el'}
        self.gausspoint_data = {k[:-3]:v for k,v in data.items() if k[-2:] == 'gp'}
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
        