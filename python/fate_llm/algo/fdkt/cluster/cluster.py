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
from typing import List
from .cluster_method import get_cluster_runner


class SentenceCluster(object):
    def __init__(self, model, cluster_method="kmeans", n_clusters=8, **other_cluster_args):
        self.model = model
        self.cluster_method = cluster_method
        self.n_clusters = n_clusters
        self.other_cluster_args = other_cluster_args

    def get_embeddings(self, sentences: List[str]):
        return self.model.encode(sentences)

    def cluster(self, sentences):
        embeddings = self.get_embeddings(sentences)

        cluster_runner = get_cluster_runner(method=self.cluster_method,
                                            n_clusters=self.n_clusters,
                                            **self.other_cluster_args)

        cluster_rets = cluster_runner.fit(embeddings)

        return cluster_rets
