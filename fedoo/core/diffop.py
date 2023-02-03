#===============================================================================
# Derivative and Differential operator
#===============================================================================
#from fedoo.pgd.SeparatedArray import *
# from fedoo.core.modelingspace  import ModelingSpace
import numpy as np

from numbers import Number #classe de base qui permet de tester si un type est numérique

class _Derivative: #derivative operator used in DiffOp
    """    
    Define a derivative operator. 
    _Derivative(u,x,ordre,decentrement)

    Parameters
    ----------
    u (int) : the variable that is derived    
    x (int) : derivative with respect to x
    ordre (int) : the order of the derivative (0 for no derivative)
    decentrement (int) : used only to define a decentrement when using finie diferences method
    """
    
    def __init__(self, u=0, x=0, ordre=0, decentrement=0):
        self.u = u #u est la variable dérivée
        self.x = x #x est la coordonnée par rapport à qui on dérive

                   #x peut etre une liste. Dans ce cas on dérive par rapport à plusieurs coordonnées (ex : si x=[0,1] op = d_dx+d_dy pour la divergence)        
        self.ordre = ordre #ordre de la dérivée (0, 1 ou 2)
        self.decentrement = decentrement #décentrement des dériviées pour différences finies uniquement


class DiffOp: 
    def __init__(self, u, x=0, ordre=0, decentrement=0, vir=0):
       
        self.mesh = None
        
        # mod_space = ModelingSpace.get_active()

        # if isinstance(u,str):
        #     u = mod_space.GetVariableRank(u)
        # if isinstance(x,str):
        #     x = mod_space.GetCoordinateRank(x)
            
        if isinstance(u,int):
            self.coef = [1]
            if vir == 0:
                self.op = [_Derivative(u,x,ordre,decentrement)] ; self.op_vir = [1]
            else:
                self.op_vir = [_Derivative(u,x,ordre,decentrement)] ; self.op = [1]
        elif isinstance(u,list) and isinstance(x,list) and isinstance(ordre,list) : 
            self.op = u
            self.op_vir = x
            self.coef = ordre
        else: raise NameError('Argument error')
            
    def __add__(self, A):
        if isinstance(A, DiffOp): return DiffOp(self.op+A.op, self.op_vir+A.op_vir, self.coef+A.coef)
        elif A == 0: return self
        else: return NotImplemented

    def __sub__(self, A):  
        if isinstance(A, DiffOp): return DiffOp(self.op+A.op, self.op_vir+A.op_vir, self.coef+(-A).coef)
        elif A == 0: return self
        else: return NotImplemented

    def __rsub__(self, A):
        if A == 0: return -self
        else: return NotImplemented
        
    def __neg__(self):    
        return(DiffOp(self.op,self.op_vir, [-cc for cc in self.coef]))
        
    def __mul__(self, A): 
#        if isinstance(A, SeparatedArray) and A.norm() == 0: return 0           
        if isinstance(A, DiffOp): 
            res = DiffOp([],[],[])
            for ii in range(len(A.op)):
                for jj in range(len(self.op)):
                    if (A.op[ii] != 1 and self.op[jj] == 1): #si A contient un opérateur réel et self un virtuel
                            res += DiffOp( [A.op[ii]], [self.op_vir[jj]], [A.coef[ii]*self.coef[jj]] )
                    elif (A.op[ii] == 1 and self.op[jj] != 1): #si c'est l'inverse
                            res += DiffOp( [self.op[ii]], [A.op_vir[jj]], [A.coef[ii]*self.coef[jj]] )
                    else: raise NameError('Impossible operation')
            return res
        else:  #isinstance(A, (Number, SeparatedArray)):        
            if  np.isscalar(A):
                if A == 0: return 0
                if A == 1: return self
            
            return DiffOp(self.op, self.op_vir, [A*cc for cc in self.coef])
            # return DiffOp(self.op, self.op_vir, [A if (np.isscalar(cc) and cc == 1) else A*cc for cc in self.coef]) #should improve time but doesn't work. I don't know why
#        else: 
#            return NotImplemented
                    
    def __radd__(self, A):         
        return self+A

    def __rmul__(self, A):         
        return self*A

    def __div__(self, A):         
        return self*(1/A)
    
    def __str__(self):
        res = ''
        for ii in range(len(self.op)):
            if np.isscalar(self.coef[ii]):
                coef_str = str(self.coef[ii]) + ' '
            else: coef_str = 'f '
            
            if self.op[ii] == 1: op_str = ''
            elif self.op[ii].ordre ==0:
                op_str = 'u'+str(self.op[ii].u)
            else:
                op_str = 'du'+str(self.op[ii].u)+'/dx'+str(self.op[ii].x)
            
            if self.op_vir[ii] == 1: op_vir_str = ''
            elif self.op_vir[ii].ordre ==0:
                op_vir_str = 'v'+str(self.op_vir[ii].u)
            else:
                op_vir_str = 'dv'+str(self.op_vir[ii].u)+'/dx'+str(self.op_vir[ii].x)  
            if ii!=0: res+= ' + '
            res +=  coef_str + op_vir_str + ' ' + op_str
        return res
            
        
    def sort(self):
        nn = 50
        intForSort = []
        for ii in range(len(self.op)):
            if self.op[ii] != 1 and self.op_vir != 1:
                intForSort.append(self.op_vir[ii].ordre + nn* self.op_vir[ii].x + nn**2 * self.op_vir[ii].u + nn**3 * self.op[ii].ordre + nn**4 * self.op[ii].x + nn**5 * self.op[ii].u)
            elif self.op[ii] == 1:
                if self.op_vir == 1: intForSort.append(-1)
                else:
                    intForSort.append(nn**6 + nn**6 *self.op_vir[ii].u + nn**7 * self.op_vir[ii].x + nn**8 * self.op_vir[ii].x) 
            else: #self.op_vir[ii] = 1
                intForSort.append(nn**9 + nn**9 *self.op[ii].u + nn**10 * self.op[ii].x + nn**11 * self.op[ii].x) 
        
        sortedIndices = np.array(intForSort).argsort()
        self.coef = [self.coef[i] for i in sortedIndices]
        self.op = [self.op[i] for i in sortedIndices]
        self.op_vir = [self.op_vir[i] for i in sortedIndices]
        
        return [intForSort[i] for i in sortedIndices], sortedIndices
        

    def nvar(self):
        return max([op.u for op in self.op])+1
        
#    def reduction(self, **kwargs):
#        for gg in self.coef:
#            if isinstance(gg,SeparatedArray):
#                gg.reduction(**kwargs)

    @property
    def virtual(self): #retourne l'opérateur virtuel
        return DiffOp(self.op_vir, self.op, self.coef)    
        
