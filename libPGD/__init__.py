import pkgutil

from .AssemblyPGD import AssemblyPGD as Assembly
from .MeshPGD import MeshPGD as Mesh
from .SeparatedArray import SeparatedArray, ConvertArraytoSeparatedArray, SeparatedOnes, SeparatedZeros, MergeSeparatedArray
from .SeparatedOperator import SeparatedOperator 
from .UsualFunctions import inv, sqrt, exp, power, divide


#for loader, module_name, is_pkg in  pkgutil.walk_packages(__path__):
#    module = loader.find_module(module_name).load_module(module_name)
#    exec('from .'+module_name+' import *')

    
