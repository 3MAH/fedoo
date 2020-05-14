Installation
=================================

Fedoo can be installed on any plateform as soon as you can install a python 3 
distribution with numpy and scipy and a subversion (SVN) manager.


Installation on Linux (Ubuntu, Debian or Mint)
________________________________________________


1. Install python 3 if not already installed:

.. code-block:: none

   $ sudo apt-get install python3
      
2. Install numpy and scipy if not already installed

.. code-block:: none

   $ sudo apt-get update
   $ sudo apt-install python3-pip
   $ pip3 install numpy 
   $ pip3 install scipy


3. Download the FEDOO library from https://github.com/3MAH/fedoo
  
4. Optional: if you want to be able to access fedoo from anywhere you can copy 
   the fedoo directory to "<pythondir>/Lib/site-packages" or add the fedoo 
   directory to the PYTHOPATH variable
   


Installation on Windows
________________________

1. Install python, numpy and scipy. The easiest way is to install 
   the ANADONDA distribution (https://www.anaconda.com/) which include by 
   default numpy and and scipy with many other usefull libraries (recommanded).
   
2. Download the FEDOO library from https://github.com/3MAH/fedoo
   
3. Optional: if you want to be able to access fedoo from anywhere you can copy 
   the fedoo directory to "<pythondir>/Lib/site-packages" or add the fedoo 
   directory to the PYTHOPATH environment variable.
   
4. Enjoy