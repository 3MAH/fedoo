import numpy as np

# --------------------------------------
#Reissner-Mindlin plate elements
# --------------------------------------

# simple tri3 plate -> use shear reduced_integration to avoid locking
ptri3 = {'__default':['tri3'],
         '__local_csys':True}      

# simple quad4 plate -> use shear reduced_integration to avoid locking
pquad4 = {'__default':['quad4'],
          '__local_csys':True}

# simple tri6 plate
ptri6 = {'__default':['tri6'],
         '__local_csys':True}

# simple quad8 plate
pquad8 = {'__default':['quad8'],
          '__local_csys':True}

# simple quad9 plate
pquad9 = {'__default':['quad9'],
          '__local_csys':True}

