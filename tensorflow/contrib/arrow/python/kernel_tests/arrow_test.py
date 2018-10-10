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

import os
import subprocess
import sys
import tempfile

import pyarrow as pa

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
    names = ["int32", "float32", "fixed array(int32)", "var array(int32)"]

    data = [
       [1, 2, 3, 4],
       [1.1, 2.2, 3.3, 4.4],
       [[1, 1], [2, 2], [3, 3], [4, 4]],
       [[1], [2, 2], [3, 3, 3], [4, 4, 4]],
    ]

    arrays = [
        pa.array(data[0], type=pa.int32()),
        pa.array(data[1], type=pa.float32()),
        pa.array(data[2], type=pa.list_(pa.int32())),
        pa.array(data[3], type=pa.list_(pa.int32())),
    ]

    batch = pa.RecordBatch.from_arrays(arrays, names)

    columns = (0, 1, 2, 3)
    output_types = (dtypes.int32, dtypes.float32, dtypes.int32, dtypes.int32)

    dataset = arrow_dataset_ops.ArrowDataset(
            batch, columns, output_types)

    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    with self.test_session() as sess:
      for row_num in range(len(data[0])):
        value = sess.run(next_element)
        self.assertEqual(value[0], data[0][row_num])
        self.assertAlmostEqual(value[1], data[1][row_num], 2)
        self.assertListEqual(value[2].tolist(), data[2][row_num])
        self.assertListEqual(value[3].tolist(), data[3][row_num])

    df = batch.to_pandas()

    dataset = arrow_dataset_ops.ArrowDataset.from_pandas(
            df, columns, output_types)

    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    with self.test_session() as sess:
      for row_num in range(len(data[0])):
        value = sess.run(next_element)
        self.assertEqual(value[0], data[0][row_num])
        self.assertAlmostEqual(value[1], data[1][row_num], 2)
        self.assertListEqual(value[2].tolist(), data[2][row_num])
        self.assertListEqual(value[3].tolist(), data[3][row_num])



  def testArrowFileDataset(self):
    f = tempfile.NamedTemporaryFile(delete=False)

    names = ["int32", "float32", "fixed array(int32)", "var array(int32)"]

    data = [
       [1, 2, 3, 4],
       [1.1, 2.2, 3.3, 4.4],
       [[1, 1], [2, 2], [3, 3], [4, 4]],
       [[1], [2, 2], [3, 3, 3], [4, 4, 4]],
    ]

    arrays = [
        pa.array(data[0], type=pa.int32()),
        pa.array(data[1], type=pa.float32()),
        pa.array(data[2], type=pa.list_(pa.int32())),
        pa.array(data[3], type=pa.list_(pa.int32())),
    ]

    batch = pa.RecordBatch.from_arrays(arrays, names)
    writer = pa.RecordBatchFileWriter(f, batch.schema)
    writer.write_batch(batch)
    writer.close()
    f.close()

    host = f.name
    columns = (0, 1, 2, 3)
    output_types = (dtypes.int32, dtypes.float32, dtypes.int32, dtypes.int32)

    dataset = arrow_dataset_ops.ArrowFileDataset(
	host, columns, output_types)

    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    with self.test_session() as sess:
      for row_num in range(len(data[0])):
        value = sess.run(next_element)
        self.assertEqual(value[0], data[0][row_num])
        self.assertAlmostEqual(value[1], data[1][row_num], 2)
        self.assertListEqual(value[2].tolist(), data[2][row_num])
        self.assertListEqual(value[3].tolist(), data[3][row_num])

    os.unlink(f.name)

  def testArrowStreamDataset(self):
    names = ["int32", "float32", "fixed array(int32)", "var array(int32)"]

    data = [
       [1, 2, 3, 4],
       [1.1, 2.2, 3.3, 4.4],
       [[1, 1], [2, 2], [3, 3], [4, 4]],
       [[1], [2, 2], [3, 3, 3], [4, 4, 4]],
    ]

    arrays = [
        pa.array(data[0], type=pa.int32()),
        pa.array(data[1], type=pa.float32()),
        pa.array(data[2], type=pa.list_(pa.int32())),
        pa.array(data[3], type=pa.list_(pa.int32())),
    ]

    interpreter_path = sys.executable
    file_path = os.path.abspath(__file__)

    p = subprocess.Popen([interpreter_path, file_path, "run_arrow_stdin"],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)

    batch = pa.RecordBatch.from_arrays(arrays, names)
    writer = pa.RecordBatchStreamWriter(p.stdin, batch.schema)
    writer.write_batch(batch)
    writer.close()

    std_out, std_err = p.communicate()
    status = p.returncode
    '''
    if status != 0:
        raise RuntimeError("Subprocess reading Arrow stream exited with error: " +
                "%d\nSTDERR:\n%s\nSTDOUT:\n%s" % (status, std_err, std_out))
    raise RuntimeError("Success!:\n%s" % std_out)
    '''

def run_arrow_stdin():
    #import tensorflow as tf
    from tensorflow.python.client import session
    host = "STDIN"
    columns = (0, 1, 2, 3)
    output_types = (dtypes.int32, dtypes.float32, dtypes.int32, dtypes.int32)

    dataset = arrow_dataset_ops.ArrowStreamDataset(
	host, columns, output_types)

    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    with session.Session() as sess:
      while True:
        try:
          value = sess.run(next_element)
          print(value)
        except tf.errors.OutOfRangeError:
          break


if __name__ == "__main__":
  import sys
  if len(sys.argv) > 1 and sys.argv[1] == "run_arrow_stdin":
    run_arrow_stdin()
  else:
    test.main()
