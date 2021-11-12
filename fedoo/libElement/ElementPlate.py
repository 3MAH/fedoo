import numpy as np
from fedoo.libElement.Element import *   

# --------------------------------------
#Reissner-Mindlin plate elements
# --------------------------------------

# simple tri3 plate -> use reduced_integration to avoid locking
ptri3 = {'__default':['tri3'],
         '__TypeOfCoordinateSystem': 'local'}      

# simple quad4 plate -> use reduced_integration to avoid locking
pquad4 = {'__default':['quad4'],
          '__TypeOfCoordinateSystem': 'local'}

# simple tri6 plate
ptri6 = {'__default':['tri6'],
          '__TypeOfCoordinateSystem': 'local'}

# simple quad8 plate
pquad8 = {'__default':['quad8'],
          '__TypeOfCoordinateSystem': 'local'}

# simple quad9 plate
pquad9 = {'__default':['quad9'],
          '__TypeOfCoordinateSystem': 'local'}

