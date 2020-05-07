import scipy as sp
from numpy import linalg

class element:              
    #Lines of xi should contains the coordinates of each point to consider
    def ShapeFunction(self,xi): pass #à définir dans les classes héritées
    #return an array whose lines are the values of the form functions (one line per point defined in xi)
    def ShapeFunctionDerivative(self,xi): pass #à définir dans les classes héritées    
    
    def ComputeDetJacobian(self,vec_x, vec_xi):
        """
        Calcul le Jacobien aux points de gauss dans le cas d'un élément isoparamétrique (c'est à dire que les mêmes fonctions de forme sont utilisées)
        vec_x est un tabeau de dimension 3 où les lignes de vec_x[el] donnent les coordonnées de chacun des noeuds de l'éléments el
        vec_xi est un tableau de dimension 2 dont les lignes donnent les coordonnées dans le repère de référence des points où on souhaite avoir le jacobien (en général pg)
        Calcul le jacobien dans self.JacobianMatrix où self.Jacobien[el,k] est le jacobien de l'élément el au point de gauss k sous la forme [[dx/dxi, dy/dxi, ...], [dx/deta, dy/deta, ...], ...]
        Calcul également le déterminant du jacobien pour le kième point de gauss de l'élément el dans self.detJ[el,k]
        """
        dnn_xi = self.ShapeFunctionDerivative(vec_xi)
        self.JacobianMatrix = sp.moveaxis([sp.dot(dnn,vec_x) for dnn in dnn_xi], 2,0) #shape = (vec_x.shape[0] = Nel, len(vec_xi)=nb_pg, nb_dir_derivative, vec_x.shape[2] = dim)
        
        if self.JacobianMatrix.shape[-2] == self.JacobianMatrix.shape[-1]:
#            self.detJ = [abs(linalg.det(J)) for J in self.JacobianMatrix]
            self.detJ = abs(linalg.det(self.JacobianMatrix)) 
        else: #l'espace réel est dans une dimension plus grande que l'espace de l'élément de référence       
            if sp.shape(self.JacobianMatrix)[-2] == 1: self.detJ = linalg.norm(JacobianMatrix, axis = 3) 
            else: #On doit avoir sp.shape(JacobianMatrix)[-2]=2 (l'elm de ref est défini en 2D) et sp.shape(JacobianMatrix)[-1]=3  (l'espace réel est 3D)
                J = self.JacobianMatrix
                self.detJ = sp.sqrt(abs(J[...,0,1]*J[...,1,2]-J[...,0,2]*J[...,1,1])**2 +\
                                    abs(J[...,0,2]*J[...,1,0]-J[...,1,2]*J[...,0,0])**2 +\
                                    abs(J[...,0,0]*J[...,1,1]-J[...,1,0]*J[...,0,1])**2 )                
    
    def ComputeJacobianMatrix(self, vec_x, vec_xi = None, rep_loc = None): #need validation
        """
        Calcul l'inverse du Jacobien aux points de gauss dans le cas d'un élément isoparamétrique (c'est à dire que les mêmes fonctions de forme sont utilisées)
        vec_xi est un tableau dont les lignes donnent les coordonnées dans le repère de référence où on souhaite avoir le jacobien (en général pg)
        """
        if vec_xi is None: vec_xi = self.xi_pg
        self.ComputeDetJacobian(vec_x, vec_xi)
        
        if rep_loc != None: 
            rep_pg = self.interpolateLocalFrame(rep_loc, vec_xi) #interpolation du repère local aux points de gauss  
            self.JacobianMatrix = sp.matmul(self.JacobianMatrix, sp.swapaxes(rep_pg,2,3) )  #to verify - sp.swapaxes(rep_pg,2,3) is equivalent to a transpose over the axis 2 and 3
#            for k,J in enumerate(self.JacobianMatrix): self.JacobianMatrix[k] = sp.dot(J, rep_pg[k].T)  
                
        if self.JacobianMatrix.shape[-2] == self.JacobianMatrix.shape[-1]:             
            self.inverseJacobian = linalg.inv(self.JacobianMatrix)
#            self.inverseJacobian = [linalg.inv(J) for J in self.JacobianMatrix]
        else: #l'espace réel est dans une dimension plus grande que l'espace de l'élément de référence   
            J = self.JacobianMatrix ; JT = sp.swapaxes(self.JacobianMatrix,2,3)                
            self.inverseJacobian = sp.matmul(JT , linalg.inv(sp.matmul(J,JT)))    #inverseJacobian.shape = (Nel,len(vec_xi)=nb_pg, dim:vec_x.shape[-1], dim:vec_xi.shape[-1])            
#            self.inverseJacobian = [sp.dot(J.T , linalg.inv(sp.dot(J,J.T))) for J in self.JacobianMatrix]                    
        
    def interpolateLocalFrame(self, rep_loc, vec_xi=None): # to do: vectorization #on interpole le repère local aux points de gauss    
        #renvoie la moyenne des repères locaux nodaux (mauvais à améliorer)        
        dim = len(rep_loc[0]) #dimension du repère local
        rep_el = [sp.mean(sp.array([rep[axe] for rep in rep_loc]),0) for axe in range(dim)] 
        rep_el = sp.array([rep_el[axe]/linalg.norm(rep_el[axe]) for axe in range(dim)])
        return [rep_el for xi in vec_xi]
            
    def repLocFromJac(self,rep_loc=None, vec_xi=None):
        return [sp.eye(3) for xi in vec_xi]
        
    def GetLocalFrame(self,vec_x, vec_xi, rep_loc=None):
        self.ComputeDetJacobian(vec_x, vec_xi)
        return self.repLocFromJac(rep_loc, vec_xi)            
    
class element1D(element):
    def __init__(self,nb_pg): #Points de gauss pour les éléments de référence 1D entre 0 et 1
        if nb_pg == 1:
            self.xi_pg = sp.c_[[1./2]]
            self.w_pg = sp.array([1.])
        elif nb_pg == 2: #ordre exacte 2
            self.xi_pg = sp.c_[[1./2 - sp.sqrt(3)/6 , 1./2 + sp.sqrt(3)/6 ]]
            self.w_pg = sp.array([1./2 , 1./2])
        elif nb_pg == 3: #ordre exacte 3
            self.xi_pg = sp.c_[[1./2-sp.sqrt(0.15) , 1./2 , 1./2 + sp.sqrt(0.15)]]
            self.w_pg = sp.array([5./18, 8./18, 5./18])
        elif nb_pg == 4:
            w_1  =   0.5 + 1.0 / (6.0 * sp.sqrt(6.0 / 5.0))
            w_2  =   0.5 - 1.0 / (6.0 * sp.sqrt(6.0 / 5.0))
            a_1  = 0.5*(1 + sp.sqrt((3.0 - 2.0 * sp.sqrt(6.0 / 5.0)) / 7.0))
            b_1  = 0.5*(1 - sp.sqrt((3.0 - 2.0 * sp.sqrt(6.0 / 5.0)) / 7.0))
            a_2  = 0.5*(1 + sp.sqrt((3.0 + 2.0 * sp.sqrt(6.0 / 5.0)) / 7.0))
            b_2  = 0.5*(1 - sp.sqrt((3.0 + 2.0 * sp.sqrt(6.0 / 5.0)) / 7.0))
            self.xi_pg   = sp.c_[[b_2,b_1,a_1,a_2]]
            self.w_pg= sp.array([w_2/2, w_1/2, w_1/2, w_2/2])
        elif nb_pg == 0: #if nb_pg == 0, we take the position of the nodes            
            self.xi_pg = self.xi_nd
        else:
            assert 0, "Number of gauss points "+str(nb_pg)+" unavailable for 1D element"                                                     
                          
        self.ShapeFunctionPG = self.ShapeFunction(self.xi_pg)
        self.ShapeFunctionDerivativePG = self.ShapeFunctionDerivative(self.xi_pg)
        
    def ComputeDetJacobian(self,vec_x, vec_xi):
        dnn_xi = self.ShapeFunctionDerivative(vec_xi)
        self.JacobianMatrix = sp.moveaxis([sp.dot(dnn,vec_x) for dnn in dnn_xi], 2,0) #shape = (vec_x.shape[0] = Nel, len(vec_xi)=nb_pg, nb_dir_derivative, vec_x.shape[2] = dim)
                
#        sp.moveaxis([sp.dot(dnn,vec_x) for dnn in dnn_xi], 2,0)
#        self.JacobianMatrix = [linalg.norm(sp.dot(dnn,vec_x)) for dnn in dnn_xi] #dx_dxi avec x tangeant à l'élément (repère local élémentaire)
        self.detJ = self.JacobianMatrix.reshape(len(vec_x),-1) #In 1D, the jacobian matrix is a scalar   
        
    def ComputeJacobianMatrix(self, vec_x, vec_xi = None, localFrame = None): 
        """
        The localFrame isn't used
        The jacobian is computed along the axis of the 1D element 
        """        
        if vec_xi is None: vec_xi = self.xi_pg
        self.ComputeDetJacobian(vec_x, vec_xi)        
        self.inverseJacobian = 1./self.JacobianMatrix #dxi_dx
#        self.inverseJacobian = [sp.array([1./J]) for J in self.JacobianMatrix] #dxi_dx                                    

    def __GetLocalFrameFromX(self, listX, elementLocalFrame):
        lastAxis = len(listX[0].shape)-1
        if listX[0].shape[lastAxis]==3: #espace 3D
            listY = [] ; listZ =[]        
            if elementLocalFrame == None: 
                for X in listX:                    
                    Z = sp.c_[-X[...,2],0*X[...,0],X[...,0]] #  équivalent à : Z = sp.cross(X,sp.array([0,1,0]))
                    Y = sp.zeros((len(X),3))
                    
                    normZ = linalg.norm(Z,axis=lastAxis)
                    list_nnz_normZ = sp.nonzero(normZ)[0] #list of line of Z where norm Z is not zero
                                        
                    # filling the Z and Y values if normZ is not 0
                    if len(list_nnz_normZ) > 0:
                        Z[list_nnz_normZ] /= linalg.norm(Z[list_nnz_normZ], axis=lastAxis).reshape(-1,1)
                        Y[list_nnz_normZ] = sp.cross(Z[list_nnz_normZ],X[list_nnz_normZ]) # équivalent à : Y = y-X[1]*X ; Y/=norm(Y)
                    
                    list_zero_normZ = sp.nonzero(Y==0)[0] #list of line of Z where norm Z is zero                     
                    
                    # filling the Z and Y values if normZ is 0
                    if len(list_zero_normZ) > 0:
                        Y[list_zero_normZ] = sp.c_[-X[list_zero_normZ,1],X[list_zero_normZ,0],0*X[list_zero_normZ,0]] # équivalent à : Y = sp.cross(sp.array([0,0,1]),X)
                        Y[list_zero_normZ] /= linalg.norm(Y[list_zero_normZ], axis=lastAxis).reshape(-1,1)
                        Z[list_zero_normZ] = sp.cross(X[list_zero_normZ],Y[list_zero_normZ])
                        
                    listY.append(Y) ; listZ.append(Z)
                                 
            else:
                for k,X in enumerate(listX):                    
                    y = elementLocalFrame[...,k][1] ; z = elementLocalFrame[...,k][2]
                    Z = sp.cross(X,y) ; Y = sp.zeros((len(X),3))
                    normZ = linalg.norm(Z,axis=lastAxis)
                    list_nnz_normZ = sp.nonzero(normZ)[0] #list of line of Z where norm Z is not zero
                    
                    if len(list_nnz_normZ) > 0:
                        Z[list_nnz_normZ] /= linalg.norm(Z[list_nnz_normZ], axis=lastAxis).reshape(-1,1)
                        Y[list_nnz_normZ] = sp.cross(Z[list_nnz_normZ],X[list_nnz_normZ]) 
                    
                    list_zero_normZ = sp.nonzero(Y==0)[0] #list of line of Z where norm Z is zero                     
                    
                    if len(list_zero_normZ) > 0:
                        Y[list_zero_normZ] = sp.cross(z[list_zero_normZ],X[list_zero_normZ])
                        Y[list_zero_normZ] /= linalg.norm(Y[list_zero_normZ], axis=lastAxis).reshape(-1,1)                                                                        
                        Z[list_zero_normZ] = sp.cross(X[list_zero_normZ],Y[list_zero_normZ])
                    
                    listY.append(Y) ; listZ.append(Z)
            
            return sp.array([[listX[k],listY[k],listZ[k]] for k in range(len(listX))])
        else: #espace 2D
            listY = [sp.array([-X[:,1],X[:,0]]) for X in listX]
            return sp.array([[listX[k],listY[k]] for k in range(len(listX))])  #shape = (len(listX), dim:listvec, Nel, dim:coordinates)                         
            
    def GetLocalFrame(self,vec_x, vec_xi, localFrame=None): #linear local frame
        if len(vec_x.shape) == 2: vec_x = sp.array([vec_x])
        dnn_xi = self.ShapeFunctionDerivative(vec_xi)        
        listX = [sp.dot(dnn,vec_x)[0] for dnn in dnn_xi]   
        lastAxis = len(listX[0].shape)-1
        listX = [X/linalg.norm(X,axis = lastAxis).reshape(-1,1) for X in listX]
        if localFrame is None: return sp.moveaxis(self.__GetLocalFrameFromX(listX, None) , 2,0)
        else:
            rep_pg = self.interpolateLocalFrame(localFrame, vec_xi) #interpolation du repère local aux points de gauss                  
            return sp.moveaxis(self.__GetLocalFrameFromX(listX, rep_pg) , 2,0)      #shape = (Nel, len(listX), dim:listvec, dim:coordinates)                      
            
class element1DGeom2(element1D): #élément 1D à géométrie affine (interpolée par 2 noeuds)         
    def ComputeDetJacobian(self, vec_x, vec_xi):
        """
        Calcul le Jacobien aux points de gauss
        vec_x est un tabeau dont les lignes vec_x[el] donnent les coordonnées de chaqun des noeuds de l'éléments el
        vec_xi est un tableau dont les lignes donnent les coordonnées dans le repère de référence où on souhaite avoir le jacobien (en général pg)
        """                
        x1 = vec_x[:,0] ; x2 = vec_x[:,1] 
        self.JacobianMatrix = linalg.norm(x2-x1, axis=1) #longueur de l'élément réel car la longueur de élément de référence = 1      #shape = (vec_x.shape[0] = Nel, len(vec_xi)=nb_pg, nb_dir_derivative, vec_x.shape[2] = dim)                  
        self.detJ = self.JacobianMatrix.reshape(-1,1) * sp.ones(len(vec_xi))  #detJ est constant sur l'élément
        #        self.detJ = self.JacobianMatrix.reshape(len(vec_x),-1) #In 1D, the jacobian matrix is a scalar   

    
    def ComputeJacobianMatrix(self, vec_x, vec_xi = None, rep_loc = None):                       
        #rep_loc inutile ici : le repère local élémentaire est utilisé (x : tangeante à l'élément)
        if vec_xi is None: vec_xi = self.xi_pg
        self.ComputeDetJacobian(vec_x, vec_xi)
        self.inverseJacobian = (1./self.JacobianMatrix).reshape(-1,1,1,1) #dxi/dx -> scalar #shape = (vec_x.shape[0] = Nel, len(vec_xi)=nb_pg, nb_dir_derivative, vec_x.shape[2] = dim)
#        self.derivativePG = self.inverseJacobian.reshape(-1,1,1,1) * sp.array(self.ShapeFunctionDerivativePG).reshape(1,len(vec_xi),1,-1)
#        self.inverseJacobian = [sp.array([qq]) for xi in vec_xi] #qq est constant sur l'élément
#        self.derivativePG = sp.array([self.inverseJacobian[k] * self.ShapeFunctionDerivativePG[k] for k in range(len(vec_xi))])

    def GetLocalFrame(self,vec_x, vec_xi, localFrame=None): #linear local frame
        if len(vec_x.shape) == 2: vec_x = sp.array([vec_x])
        listX = [vec_x[...,1,0:3]-vec_x[...,0,0:3]] #only 1 element in the list because frame doesn't change over the element (in general nppg elements in list)
        lastAxis = len(listX[0].shape)-1
        listX = [X/linalg.norm(X,axis = lastAxis).reshape(-1,1) for X in listX]
        
        if localFrame is None: return sp.moveaxis(self._element1D__GetLocalFrameFromX(listX, None), 2,0)
        else:
            rep_pg = self.interpolateLocalFrame(localFrame, vec_xi) #interpolation du repère local aux points de gauss                  
            return sp.moveaxis([self._element1D__GetLocalFrameFromX(listX, rep_pg)[0] for xi in vec_xi], 2,0)  #shape = (Nel, len(listX), dim:listvec, dim:coordinates)

class element2D(element):  

    def ComputeJacobianMatrix(self, vec_x, vec_xi = None, rep_loc = None): 
        if vec_xi is None: vec_xi == self.xi_pg
        self.ComputeDetJacobian(vec_x, vec_xi)
        
        if self.JacobianMatrix.shape[-2] == self.JacobianMatrix.shape[-1]:
            if rep_loc != None:
                rep_pg = self.interpolateLocalFrame(rep_loc, vec_xi) #interpolation du repère local aux points de gauss  -> shape = (len(vec_x),len(vec_xi),dim,dim)
                self.JacobianMatrix = sp.matmul(self.JacobianMatrix, sp.swapaxes(rep_pg,2,3) )  #to verify - sp.swapaxes(rep_pg,2,3) is equivalent to a transpose over the axis 2 and 3
#                rep_pg = self.interpolateLocalFrame(rep_loc, vec_xi) #interpolation du repère local aux points de gauss  
#                for k,J in enumerate(self.JacobianMatrix): self.JacobianMatrix[k] = sp.dot(J, rep_pg[k].T)
        else: #l'espace réel est dans une dimension plus grande que l'espace de l'élément de référence   
            if rep_loc is None:
                J = self.JacobianMatrix ; JT = sp.swapaxes(self.JacobianMatrix,2,3)                
                self.inverseJacobian = sp.matmul(JT , linalg.inv(sp.matmul(J,JT)))    #inverseJacobian.shape = (Nel,len(vec_xi)=nb_pg, dim:vec_x.shape[-1], dim:vec_xi.shape[-1])
#                self.inverseJacobian = [sp.dot(J.T , linalg.inv(sp.dot(J,J.T))) for J in self.JacobianMatrix]   #this line may have a high computational cost
                return                            
            else:
                rep_pg = self.repLocFromJac(rep_loc,vec_xi)[:,0:2,:]
                for k,J in enumerate(self.JacobianMatrix): self.JacobianMatrix[k] = sp.dot(J, rep_pg[k].T)                                         
        self.inverseJacobian = linalg.inv(self.JacobianMatrix) 
        

    def repLocFromJac(self,rep_loc=None, vec_xi=None): #to do: vectorization
        listZ=[sp.cross(J[0],J[1]) for J in self.JacobianMatrix] 
        listX = [] ; listY = []            
        if rep_loc is None:               
            listZ=[Z/linalg.norm(Z) for Z in listZ]  #direction perpendiculaire au plan tangeant   
            x = sp.array([1,0,0]) ; y=sp.array([0,1,0])  
            for Z in listZ:
                X=x-Z[0]*Z ; normX = linalg.norm(X)
                if normX != 0:
                    listX.append(X/normX)
                    listY.append(sp.cross(Z,listX[-1]))
                else: 
                    listY.append(y-Z[1]*Z) ; listY[-1] /= linalg.norm(listY[-1])
                    listX.append(sp.cross(listY[-1],Z))  
        else: 
            rep_pg = self.interpolateLocalFrame(rep_loc, vec_xi) #interpolation du repère local aux points de gauss                  
            for k,Z in enumerate(listZ):
                x = rep_pg[k][0] ; y = rep_pg[k][1]
                X=x-sp.dot(x,Z)*Z ; normX = linalg.norm(X)
                if normX != 0:
                    listX.append(X/normX)
                    listY.append(sp.cross(Z,listX[-1]))
                else: 
                    listY.append(y-sp.dot(y,Z)*Z) ; listY[-1] /= linalg.norm(listY[-1])
                    listX.append(sp.cross(listY[k],Z))

        return sp.array([[listX[k],listY[k],listZ[k]] for k in range(len(listZ))])



