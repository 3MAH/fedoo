"""Elements to avoid volumetric locking in nearly incompressible analysis.

Experimental only. Doesn't seem to work well"""

import numpy as np
from fedoo.lib_elements.element_list import CombinedElement

# selective reduce integration
hex8sri = CombinedElement("hex8sri", "hex8", default_n_gp=8)
hex8sri.set_variable_interpolation("_DispX", "hex8r")
hex8sri.set_variable_interpolation("_DispY", "hex8r")
hex8sri.set_variable_interpolation("_DispZ", "hex8r")

quad4sri = CombinedElement("quad4sri", "quad4", default_n_gp=4)
quad4sri.set_variable_interpolation("_DispX", "quad4r")
quad4sri.set_variable_interpolation("_DispY", "quad4r")
quad4sri.set_variable_interpolation("_DispZ", "quad4r")
