# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Code to generate processed features."""
import copy
from typing import List, Mapping, Tuple
from alphafold.model.tf import input_pipeline
from alphafold.model.tf import proteins_dataset
import ml_collections
import numpy as np
import tensorflow.compat.v1 as tf

FeatureDict = Mapping[str, np.ndarray]


def make_data_config(
    config: ml_collections.ConfigDict,
    num_res: int,
    ) -> Tuple[ml_collections.ConfigDict, List[str]]:
  """Makes a data config for the input pipeline."""
  cfg = copy.deepcopy(config.data)

  feature_names = cfg.common.unsupervised_features
  if cfg.common.use_templates:
    feature_names += cfg.common.template_features

  with cfg.unlocked():
    cfg.eval.crop_size = num_res

  return cfg, feature_names


def tf_example_to_features(tf_example: tf.train.Example,
                           config: ml_collections.ConfigDict,
                           random_seed: int = 0) -> FeatureDict:
  """Converts tf_example to numpy feature dictionary."""
  num_res = int(tf_example.features.feature['seq_length'].int64_list.value[0])
  cfg, feature_names = make_data_config(config, num_res=num_res)

  if 'deletion_matrix_int' in set(tf_example.features.feature):
    deletion_matrix_int = (
        tf_example.features.feature['deletion_matrix_int'].int64_list.value)
    feat = tf.train.Feature(float_list=tf.train.FloatList(
        value=map(float, deletion_matrix_int)))
    tf_example.features.feature['deletion_matrix'].CopyFrom(feat)
    del tf_example.features.feature['deletion_matrix_int']

  tf_graph = tf.Graph()
  with tf_graph.as_default(), tf.device('/device:CPU:0'):
    tf.compat.v1.set_random_seed(random_seed)
    tensor_dict = proteins_dataset.create_tensor_dict(
        raw_data=tf_example.SerializeToString(),
        features=feature_names)
    processed_batch = input_pipeline.process_tensors_from_config(
        tensor_dict, cfg)

  tf_graph.finalize()

  with tf.Session(graph=tf_graph) as sess:
    features = sess.run(processed_batch)

  return {k: v for k, v in features.items() if v.dtype != 'O'}


def np_example_to_features(np_example: FeatureDict,
                           config: ml_collections.ConfigDict,
                           random_seed: int = 0) -> FeatureDict:
  """Preprocesses NumPy feature dict using TF pipeline."""
  np_example = dict(np_example)
  num_res = int(np_example['seq_length'][0])
  print("Setting up features...num_res=",num_res)
  print(np_example.keys())
  cfg, feature_names = make_data_config(config, num_res=num_res)
  print("Feature names:",feature_names)

  if 'deletion_matrix_int' in np_example:
    np_example['deletion_matrix'] = (
        np_example.pop('deletion_matrix_int').astype(np.float32))

  tf_graph = tf.Graph()
  with tf_graph.as_default(), tf.device('/device:CPU:0'):
    tf.compat.v1.set_random_seed(random_seed)
    tensor_dict = proteins_dataset.np_to_tensor_dict(
        np_example=np_example, features=feature_names)
    print("Tensor dict keys",tensor_dict.keys())
    print("resample:",cfg.common.resample_msa_in_recycling, "num_ensemble:",cfg.eval.num_ensemble)
    processed_batch = input_pipeline.process_tensors_from_config(
        tensor_dict, cfg)
  print("Processed batch keys",processed_batch.keys())
  tf_graph.finalize()

  with tf.Session(graph=tf_graph) as sess:
    features = sess.run(processed_batch)

  for k, v in features.items():
    if k =="msa_feat":
      n1,n2,n3,n4 = v.shape
      print("SHAPE:",n1,n2,n3,n4)
      f1 = v[0]
      f2 = v[1]
      f3 = v[2]
      f12 = f1 - f2
      f13 = f1 - f3
      f23 = f2 - f3
      a12 = f12[f12 != 0]
      print ("non-zero diffs for msa:",a12)


  return {k: v for k, v in features.items() if v.dtype != 'O'}
