# Copyright 2022 The MediaPipe Authors. All Rights Reserved.
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

import os

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from mediapipe.model_maker.python.core.utils import model_util
from mediapipe.model_maker.python.core.utils import quantization
from mediapipe.model_maker.python.core.utils import test_util


class ModelUtilTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='input_only_steps_per_epoch',
          steps_per_epoch=1000,
          batch_size=None,
          train_data=None,
          expected_steps_per_epoch=1000),
      dict(
          testcase_name='input_steps_per_epoch_and_batch_size',
          steps_per_epoch=1000,
          batch_size=32,
          train_data=None,
          expected_steps_per_epoch=1000),
      dict(
          testcase_name='input_steps_per_epoch_batch_size_and_train_data',
          steps_per_epoch=1000,
          batch_size=32,
          train_data=tf.data.Dataset.from_tensor_slices([[0, 1], [1, 1], [0, 0],
                                                         [1, 0]]),
          expected_steps_per_epoch=1000),
      dict(
          testcase_name='input_batch_size_and_train_data',
          steps_per_epoch=None,
          batch_size=2,
          train_data=tf.data.Dataset.from_tensor_slices([[0, 1], [1, 1], [0, 0],
                                                         [1, 0]]),
          expected_steps_per_epoch=2))
  def test_get_steps_per_epoch(self, steps_per_epoch, batch_size, train_data,
                               expected_steps_per_epoch):
    estimated_steps_per_epoch = model_util.get_steps_per_epoch(
        steps_per_epoch=steps_per_epoch,
        batch_size=batch_size,
        train_data=train_data)
    self.assertEqual(estimated_steps_per_epoch, expected_steps_per_epoch)

  def test_get_steps_per_epoch_raise_value_error(self):
    with self.assertRaises(ValueError):
      model_util.get_steps_per_epoch(
          steps_per_epoch=None, batch_size=16, train_data=None)

  def test_warmup(self):
    init_lr = 0.1
    warmup_steps = 1000
    num_decay_steps = 100
    learning_rate_fn = tf.keras.experimental.CosineDecay(
        initial_learning_rate=init_lr, decay_steps=num_decay_steps)
    warmup_object = model_util.WarmUp(
        initial_learning_rate=init_lr,
        decay_schedule_fn=learning_rate_fn,
        warmup_steps=1000,
        name='test')
    self.assertEqual(
        warmup_object.get_config(), {
            'initial_learning_rate': init_lr,
            'decay_schedule_fn': learning_rate_fn,
            'warmup_steps': warmup_steps,
            'name': 'test'
        })

  def test_export_tflite(self):
    input_dim = 4
    model = test_util.build_model(input_shape=[input_dim], num_classes=2)
    tflite_file = os.path.join(self.get_temp_dir(), 'model.tflite')
    model_util.export_tflite(model, tflite_file)
    self._test_tflite(model, tflite_file, input_dim)

  @parameterized.named_parameters(
      dict(
          testcase_name='dynamic_quantize',
          config=quantization.QuantizationConfig.for_dynamic(),
          model_size=1288),
      dict(
          testcase_name='int8_quantize',
          config=quantization.QuantizationConfig.for_int8(
              representative_data=test_util.create_dataset(
                  data_size=10, input_shape=[16], num_classes=3)),
          model_size=1832),
      dict(
          testcase_name='float16_quantize',
          config=quantization.QuantizationConfig.for_float16(),
          model_size=1468))
  def test_export_tflite_quantized(self, config, model_size):
    input_dim = 16
    num_classes = 2
    max_input_value = 5
    model = test_util.build_model([input_dim], num_classes)
    tflite_file = os.path.join(self.get_temp_dir(), 'model_quantized.tflite')

    model_util.export_tflite(model, tflite_file, config)
    self._test_tflite(
        model, tflite_file, input_dim, max_input_value, atol=1e-00)
    self.assertNear(os.path.getsize(tflite_file), model_size, 300)

  def _test_tflite(self,
                   keras_model: tf.keras.Model,
                   tflite_model_file: str,
                   input_dim: int,
                   max_input_value: int = 1000,
                   atol: float = 1e-04):
    np.random.seed(0)
    random_input = np.random.uniform(
        low=0, high=max_input_value, size=(1, input_dim)).astype(np.float32)

    self.assertTrue(
        test_util.is_same_output(
            tflite_model_file, keras_model, random_input, atol=atol))


if __name__ == '__main__':
  tf.test.main()
