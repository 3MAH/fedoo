from fedoo.libUtil.Operator  import OpDiff
from fedoo.libUtil.Dimension import ProblemDimension


class StrainOperator:
    # __InitialGradDisp = None # For initial displacement in case of large displacement    

    # @staticmethod
    # def UpdateInitialGradDisp(InitialGradDisp):
    #     StrainOperator.__InitialGradDisp = InitialGradDisp

    @staticmethod
    def Get(InitialGradDisp = None):
        n = ProblemDimension.Get()
        # InitialGradDisp = StrainOperator.__InitialGradDisp

        if (InitialGradDisp is None) or (InitialGradDisp == 0):
            du_dx = OpDiff('DispX', 'X', 1)
            dv_dy = OpDiff('DispY', 'Y', 1)
            du_dy = OpDiff('DispX', 'Y', 1)
            dv_dx = OpDiff('DispY', 'X', 1)
        
            if n == "2Dplane" or n == "2Dstress":
                eps = [du_dx, dv_dy, 0, 0, 0, du_dy+dv_dx]
        
        #    elif n == "2Dstress":
        #        dw_dx = OpDiff('DispZ', 'X', 1)
        #        dw_dy = OpDiff('DispZ', 'Y', 1)        
        #        eps = [du_dx, dv_dy, 0, dw_dy, dw_dx, du_dy+dv_dx]        
        
            elif n == "3D":
                dw_dz = OpDiff('DispZ', 'Z', 1)
                du_dz = OpDiff('DispX', 'Z', 1)
                dv_dz = OpDiff('DispY', 'Z', 1)
                dw_dx = OpDiff('DispZ', 'X', 1)
                dw_dy = OpDiff('DispZ', 'Y', 1)
                eps = [du_dx, dv_dy, dw_dz, dv_dz+dw_dy, du_dz+dw_dx, du_dy+dv_dx]
          
        else:
            
            if n == "2Dplane" or n == "2Dstress":
                GradOperator = [[OpDiff(IDvar, IDcoord,1) for IDcoord in ['X','Y']] for IDvar in ['DispX','DispY']]
                eps = [GradOperator[i][i] + sum([GradOperator[k][i]*InitialGradDisp[k][i] for k in range(2)]) for i in range(2)] 
                eps += [0, 0, 0]
                eps += [GradOperator[0][1] + GradOperator[1][0] + sum([GradOperator[k][0]*InitialGradDisp[k][1] + GradOperator[k][1]*InitialGradDisp[k][0] for k in range(2)])]  
            
            elif n == "3D":
                GradOperator = [[OpDiff(IDvar, IDcoord,1) for IDcoord in ['X','Y','Z']] for IDvar in ['DispX','DispY','DispZ']]
                eps = [GradOperator[i][i] + sum([GradOperator[k][i]*InitialGradDisp[k][i] for k in range(3)]) for i in range(3)] 
                eps += [GradOperator[1][2] + GradOperator[2][1] + sum([GradOperator[k][1]*InitialGradDisp[k][2] + GradOperator[k][2]*InitialGradDisp[k][1] for k in range(3)])]
                eps += [GradOperator[0][2] + GradOperator[2][0] + sum([GradOperator[k][0]*InitialGradDisp[k][2] + GradOperator[k][2]*InitialGradDisp[k][0] for k in range(3)])]
                eps += [GradOperator[0][1] + GradOperator[1][0] + sum([GradOperator[k][0]*InitialGradDisp[k][1] + GradOperator[k][1]*InitialGradDisp[k][0] for k in range(3)])]          

            
        eps_vir = [e.virtual() if e != 0 else 0 for e in eps ]               
        
        return eps, eps_vir 


def GetStrainOperator(InitialGradDisp = None):
    #return linear the operator to get the strain tensor (to use in weakform)
    #InitialGradDisp is used for initial displacement effect in incremental approach
    return StrainOperator.Get(InitialGradDisp)

if __name__=="__main__":
    ProblemDimension("3D")
    A,B = GetStrainOperator()

