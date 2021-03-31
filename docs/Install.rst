Installation
=================================

Fedoo can be installed on any plateform in which you can install a python 3 
distribution with numpy, scipy and matplotlib.


Installation on Linux (Ubuntu, Debian or Mint)
________________________________________________


1. Install python 3 if not already installed (ANADONDA distribution recommended : https://www.anaconda.com/)

.. code-block:: none

   $ sudo apt-get install python3
      
2. Install numpy, scipy and matplotlib if not already installed.

.. code-block:: none

   $ sudo apt-get update
   $ sudo apt-install python3-pip
   $ pip3 install numpy 
   $ pip3 install scipy
   $ pip3 install matplotlib

3. Download the FEDOO library from https://github.com/3MAH/fedoo

4. Optional (highly recommended): Install simcoon folowing the instructions given here : https://simcoon.readthedocs.io/en/latest/installation.html
  
5. Optional: install the package pardiso4py to increase the performance of direct sparse solver.

.. code-block:: none

   $ conda install -c conda-forge pardiso4py 

6. Optional: if you want to be able to access fedoo from anywhere you can copy 
   the fedoo directory to "<pythondir>/Lib/site-packages" or add the fedoo 
   directory to the PYTHOPATH variable.
   
7. Enjoy !


Installation on Windows
________________________

1. Install python, numpy and scipy. The easiest way is to install 
   the ANADONDA distribution (https://www.anaconda.com/) which include by 
   default numpy, scipy and matplotlib with many other usefull libraries (recommended).
   
2. Download the FEDOO library from https://github.com/3MAH/fedoo
   
3. Optional (highly recommended): Install simcoon. A compiled version of the simcoon python interface for win64 is available in the fedoo github repository, in the util folder.
Just add this library to your path or your project current directory and install the librairies boost and cgal : 

.. code-block:: none

   $ conda install -c anaconda boost
   $ conda install -c conda-forge cgal
 
If you want to recompile the simcoon library yourself (not recommended), try to follow the instructions given  
here : https://simcoon.readthedocs.io/en/latest/installation.html

4. Optional: install the package pardiso4py to increase the performance of direct sparse solver. In an anaconda prompt write : 

.. code-block:: none

   $ conda install -c conda-forge pardiso4py 

5. Optional: if you want to be able to access fedoo from anywhere you can copy 
   the fedoo directory to "<pythondir>/Lib/site-packages" or add the fedoo 
   directory to the PYTHOPATH environment variable.

   
6. Enjoy !