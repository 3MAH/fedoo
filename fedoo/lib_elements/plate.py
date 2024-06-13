import numpy as np
from fedoo.lib_elements.element_list import CombinedElement

# --------------------------------------
# Reissner-Mindlin plate elements
# --------------------------------------

# simple tri3 plate -> use shear reduced_integration to avoid locking
ptri3 = CombinedElement("ptri3", "tri3", default_n_gp=3, local_csys=True)
# ptri3 = {'__default':['tri3'],
#          '__local_csys':True}

# simple quad4 plate -> use shear reduced_integration to avoid locking
pquad4 = CombinedElement("pquad4", "quad4", default_n_gp=4, local_csys=True)
# pquad4 = {'__default':['quad4'],
#           '__local_csys':True}

# simple tri6 plate
ptri6 = CombinedElement("ptri6", "tri6", default_n_gp=4, local_csys=True)
# ptri6 = {'__default':['tri6'],
#          '__local_csys':True}

# simple quad8 plate
pquad8 = CombinedElement("pquad8", "quad8", default_n_gp=9, local_csys=True)
# pquad8 = {'__default':['quad8'],
#           '__local_csys':True}

# simple quad9 plate
pquad9 = CombinedElement("pquad9", "quad9", default_n_gp=9, local_csys=True)
# pquad9 = {'__default':['quad9'],
#           '__local_csys':True}
