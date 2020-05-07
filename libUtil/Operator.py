#===============================================================================
# Derivative and Differential operator
#===============================================================================
#from fedoo.libPGD.SeparatedArray import *
from fedoo.libUtil.Variable  import Variable
from fedoo.libUtil.Coordinate  import Coordinate

from numbers import Number #classe de base qui permet de tester si un type est numérique

class OpDerive: #derivative operator used in OpDiff
    """
    Define a derivative operator.
    OpDerive(u,x,ordre,decentrement)

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


class OpDiff: 
    def __init__(self, u, x=0, ordre=0, decentrement=0, vir=0):
       
        self.mesh = None

        if isinstance(u,str):
            u = Variable.GetRank(u)
        if isinstance(x,str):
            x = Coordinate.GetRank(x)
            
        if isinstance(u,int):
            self.coef = [1]
            if vir == 0:
                self.op = [OpDerive(u,x,ordre,decentrement)] ; self.op_vir = [1]
            else:
                self.op_vir = [OpDerive(u,x,ordre,decentrement)] ; self.op = [1]
        elif isinstance(u,list) and isinstance(x,list) and isinstance(ordre,list) : 
            self.op = u
            self.op_vir = x
            self.coef = ordre
        else: raise NameError('Argument error')
            
    def __add__(self, A):
        if isinstance(A, OpDiff): return OpDiff(self.op+A.op, self.op_vir+A.op_vir, self.coef+A.coef)
        elif A == 0: return self
        else: return NotImplemented

    def __sub__(self, A):  
        if isinstance(A, OpDiff): return OpDiff(self.op+A.op, self.op_vir+A.op_vir, self.coef+(-A).coef)
        elif A == 0: return self
        else: return NotImplemented

    def __rsub__(self, A):
        if A == 0: return -self
        else: return NotImplemented
        
    def __neg__(self):    
        return(OpDiff(self.op,self.op_vir, [-cc for cc in self.coef]))
        
    def __mul__(self, A): 
#        if isinstance(A, SeparatedArray) and A.norm() == 0: return 0           
        if isinstance(A, OpDiff): 
            res = OpDiff([],[],[])
            for ii in range(len(A.op)):
                for jj in range(len(self.op)):
                    if (A.op[ii] != 1 and self.op[jj] == 1): #si A contient un opérateur réel et self un virtuel
                            res += OpDiff( [A.op[ii]], [self.op_vir[jj]], [A.coef[ii]*self.coef[jj]] )
                    elif (A.op[ii] == 1 and self.op[jj] != 1): #si c'est l'inverse
                            res += OpDiff( [self.op[ii]], [A.op_vir[jj]], [A.coef[ii]*self.coef[jj]] )
                    else: raise NameError('Impossible operation')
            return res
        else:  #isinstance(A, (Number, SeparatedArray)):        
            if A is 0: return 0
            return OpDiff(self.op, self.op_vir, [A*cc for cc in self.coef])
#        else: 
#            return NotImplemented
                    
    def __radd__(self, A):         
        return self+A

    def __rmul__(self, A):         
        return self*A

    def __div__(self, A):         
        return self*(1/A)

    def nvar(self):
        return max([op.u for op in self.op])+1
        
#    def reduction(self, **kwargs):
#        for gg in self.coef:
#            if isinstance(gg,SeparatedArray):
#                gg.reduction(**kwargs)
          
    def virtual(self): #retourne l'opérateur virtuel
        return OpDiff(self.op_vir, self.op, self.coef)    
        
