#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
from sklearn.cluster import KMeans


class KMeansRunner(object):
    def __init__(self, n_clusters, **other_cluster_args):
        self.n_clusters = n_clusters
        self.other_cluster_args = other_cluster_args

    def fit(self, x):
        model = KMeans(n_clusters=self.n_clusters, **self.other_cluster_args)
        model.fit(x)

        return model.labels_


def get_cluster_runner(method, n_clusters, **other_cluster_args):
    if method.lower() == "kmeans":
        return KMeansRunner(n_clusters, **other_cluster_args)
    else:
        raise ValueError(f"cluster method={method} is not implemented")
