"""
OPTAMI is a package implementing various optimization algorithms for PyTorch optimizer.
"""

from .cubic_newton import Cubic_Newton
from .bdgm import BDGM
from .superfast import Superfast
from .hyperfast import Hyperfast
from .tfgm import TFG
from .primal_dual import PrimalDualAccelerated
from .fgm import fast_gradient_methodM

del cubic_newton
del bdgm
del superfast
del hyperfast
del tfgm