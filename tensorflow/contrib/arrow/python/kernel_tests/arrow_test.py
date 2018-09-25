# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License.  You may obtain a copy of
# the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
# License for the specific language governing permissions and limitations under
# the License.
# ==============================================================================
"""Tests for ArrowDataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.arrow.python.ops import arrow_dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class ArrowDatasetTest(test.TestCase):

  def setUp(self):
    pass

  def testArrowDataset(self):
    self.assertTrue(True)

if __name__ == "__main__":
  test.main()
