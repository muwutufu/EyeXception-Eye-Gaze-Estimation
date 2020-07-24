"""Copyright (c) 2019 AIT Lab, ETH Zurich

Students and holders of copies of this code, accompanying datasets,
and documentation, are not allowed to copy, distribute or modify
any of the mentioned materials beyond the scope and duration of the
Machine Perception course projects.

That is, no partial/full copy nor modification of this code and
accompanying data should be made publicly or privately available to
current/future students or other parties.
"""

"""Model definitions (one class per file) to define NN architectures."""
from .ModelNV import Model1NetNV
from .ModelInception_v4 import InceptionV4
from .EyeXception import GazeNetPlus
from .NMod import NMOD
from .KerasNet import KNet
from .GazeNetItrack import ITRACK

__all__ = ('Model1NetNV', 'InceptionV4','GazeNetPlus','NMOD','KNet','ITRACK')#ExampleNet
