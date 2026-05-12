from fedoo.lib_elements.element_list import CombinedElement

# --------------------------------------
# Reissner-Mindlin plate elements
# --------------------------------------

# tri3 plate element with full integration (subjected to locking)
ptri3 = CombinedElement("ptri3", "tri3", default_n_gp=3, local_csys=True)

# quad4 plate element with full integration (subjected to locking)
pquad4 = CombinedElement("pquad4", "quad4", default_n_gp=4, local_csys=True)

# tri6 plate element. Use 3 integration points by default to avoid locking issues
ptri6 = CombinedElement("ptri6", "tri6", default_n_gp=7, local_csys=True)

# quad8 plate element
pquad8 = CombinedElement("pquad8", "quad8", default_n_gp=9, local_csys=True)

# quad8 plate element reduced integration (may be more efficient)
pquad8ri = CombinedElement("pquad8ri", "quad8", default_n_gp=4, local_csys=True)

# quad9 plate element
pquad9 = CombinedElement("pquad9", "quad9", default_n_gp=9, local_csys=True)

# tri3 plate element with reduced_integration to avoid locking
ptri3sri = CombinedElement("ptri3sri", "tri3", default_n_gp=3, local_csys=True)
ptri3sri.set_variable_interpolation("_DispX", "tri3r")
ptri3sri.set_variable_interpolation("_DispY", "tri3r")
ptri3sri.set_variable_interpolation("_DispZ", "tri3r")
ptri3sri.set_variable_interpolation("_RotX", "tri3r")
ptri3sri.set_variable_interpolation("_RotY", "tri3r")
ptri3sri.set_variable_interpolation("_RotZ", "tri3r")

# quad4 plate element with reduced_integration to avoid locking
pquad4sri = CombinedElement("pquad4sri", "quad4", default_n_gp=4, local_csys=True)
pquad4sri.set_variable_interpolation("_DispX", "quad4r")
pquad4sri.set_variable_interpolation("_DispY", "quad4r")
pquad4sri.set_variable_interpolation("_DispZ", "quad4r")
pquad4sri.set_variable_interpolation("_RotX", "quad4r")
pquad4sri.set_variable_interpolation("_RotY", "quad4r")
pquad4sri.set_variable_interpolation("_RotZ", "quad4r")

# tri6 plate element with reduced_integration (to 3gp) to avoid locking
ptri6sri = CombinedElement("ptri6sri", "tri6", default_n_gp=7, local_csys=True)
ptri6sri.set_variable_interpolation("_DispX", "tri6r")
ptri6sri.set_variable_interpolation("_DispY", "tri6r")
ptri6sri.set_variable_interpolation("_DispZ", "tri6r")
ptri6sri.set_variable_interpolation("_RotX", "tri6r")
ptri6sri.set_variable_interpolation("_RotY", "tri6r")
ptri6sri.set_variable_interpolation("_RotZ", "tri6r")

# quad9 plate element with reduced_integration (to 4gp) to avoid locking
pquad9sri = CombinedElement("pquad9sri", "quad9", default_n_gp=9, local_csys=True)
pquad9sri.set_variable_interpolation("_DispX", "quad9r")
pquad9sri.set_variable_interpolation("_DispY", "quad9r")
pquad9sri.set_variable_interpolation("_DispZ", "quad9r")
pquad9sri.set_variable_interpolation("_RotX", "quad9r")
pquad9sri.set_variable_interpolation("_RotY", "quad9r")
pquad9sri.set_variable_interpolation("_RotZ", "quad9r")