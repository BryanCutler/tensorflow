# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Arrow Dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.arrow.python.ops import arrow_op_loader  # pylint: disable=unused-import
from tensorflow.contrib.arrow.python.ops import gen_dataset_ops
from tensorflow.python.data.ops.dataset_ops import Dataset
from tensorflow.python.data.util import nest
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape


class ArrowDataset(Dataset):
  """A Arrow Dataset.
  """

  def __init__(self,
               host,
               columns,
               output_types):
    """Create a ArrowDataset.

    Args:
      host: A `tf.string` tensor containing a host address..
    """
    super(ArrowDataset, self).__init__()
    self._host = ops.convert_to_tensor(
        host, dtype=dtypes.string, name="host")
    self._columns = columns
    self._output_types = output_types

  def _as_variant_tensor(self):
    return gen_dataset_ops.arrow_dataset(self._host,
                                         self._columns,
                                         nest.flatten(self.output_types),
                                         nest.flatten(self.output_shapes))

  @property
  def output_classes(self):
    return nest.map_structure(lambda _: ops.Tensor, self._output_types)

  @property
  def output_shapes(self):
    return nest.map_structure(lambda _: tensor_shape.TensorShape([]), self._output_types)

  @property
  def output_types(self):
    return self._output_types
