# Copyright 2023 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Module for Pallas, a JAX extension for custom kernels.

This module is an alias of ``jax.pallas``, which is the preferred import path.

See the Pallas documentation at
https://docs.jax.dev/en/latest/pallas/index.html.
"""

from jax.pallas import *  # noqa: F403

# Forward deprecated attributes. The deprecation table is shared with
# jax.pallas, so accelerated deprecations apply to both modules.
import jax.pallas as _module
_deprecations = _module._deprecations  # pylint: disable=protected-access
del _module

import typing as _typing
if not _typing.TYPE_CHECKING:
  from jax._src.deprecations import deprecation_getattr as _deprecation_getattr
  __getattr__ = _deprecation_getattr(__name__, _deprecations)
  del _deprecation_getattr
del _typing
