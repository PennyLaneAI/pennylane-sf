# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Custom operations
=================

**Module name:** :mod:`pennylane_sf.ops`

.. currentmodule:: pennylane_sf.ops

Contains the :class:`ParamGraphEmbed` operation for encoding parametrized graphs into GBS for
machine learning and optimization applications.

.. autosummary::

   ParamGraphEmbed

Code details
~~~~~~~~~~~~
"""
from pennylane.operation import AllWires, CVOperation


class ParamGraphEmbed(CVOperation):
    r"""ParamGraphEmbed(params, A, n_mean, wires)
    A parametrized embedding of a graph into GBS.

    Any undirected graph can be encoded using its symmetric adjacency matrix. The adjacency
    matrix is first rescaled so that the corresponding GBS device has an initial mean number of
    photons. The adjacency matrix :math:`A` may then be varied using parameters :math:`\bm{w}`
    such that

    .. math::

        A \rightarrow WAW

    with :math:`W` a diagonal matrix set by the parameters :math:`\bm{w}`. The initial choice for
    the parameters can be :math:`\bm{w} = 1` so that :math:`W = \mathbb{I}`.

    .. note::

        This operation is only compatible with the :class:`~.GBS` device.

    **Details:**

    * Number of wires: All
    * Number of parameters: 3

    Args:
        params (array): variable parameters
        A (array): initial adjacency matrix
        n_mean (float): initial mean number of photons
        wires (Sequence[int] or int): the wire(s) the operation acts on
    """

    do_check_domain = False

    num_params = 3
    num_wires = AllWires
    par_domain = "A"

    # grad_method = None
    # grad_recipe = None
