"""Comparison / baseline methods for RNAGenScape.

These are RNA-adapted reimplementations inspired by the named methods
(DiffAb, IgLM, NOS-C, NOS-D, gg-dWJS, Energy Matching, MPGD). They are not
drop-in ports of the official repos; see each module docstring for intentional
adaptations.

Import style matches the rest of RNAGenScape: put ``src/`` on ``sys.path``, then
``from comparisons import DiffAb`` or ``from comparisons.diffab import DiffAb``.
"""

from comparisons.diffab import DiffAb
from comparisons.em import EM
from comparisons.gg_dwjs import gg_dWJS
from comparisons.iglm import IgLM
from comparisons.mpgd import MPGD
from comparisons.nos_c import NOS_C
from comparisons.nos_d import NOS_D

__all__ = ["DiffAb", "EM", "IgLM", "MPGD", "NOS_C", "NOS_D", "gg_dWJS"]
