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

from decimal import getcontext
from transformers import AutoTokenizer
import numpy as np
import json
import tqdm
from typing import List


getcontext().prec = 100


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def save_jsonl(filename, data):
    with open(filename, 'w') as file:
        for item in data:
            json.dump(item, file)
            file.write('\n')


def create_sensitivity_of_embeddings(all_embedding_matrix):
    n_dimensions = all_embedding_matrix.shape[1]
    delta_f_new = np.zeros(n_dimensions)
    for dim in tqdm.trange(n_dimensions):
        dim_data = all_embedding_matrix[:, dim]
        sorted_dim_data = np.sort(dim_data)
        differences = sorted_dim_data[-1] - sorted_dim_data[0]
        delta_f_new[dim] = differences
    return delta_f_new


def create_sorted_embedding_matrix(token_list, similarity_matrix):
    token_2_sorted_distances = dict()
    token_array = np.array(token_list)
    for idx, token in tqdm.tqdm(enumerate(token_list)):
        similarity_array = similarity_matrix[idx]
        sorted_indices = np.argsort(similarity_array)[::-1]
        token_2_sorted_distances[token] = [token_array[sorted_indices].tolist(), similarity_array[sorted_indices].tolist()]
    return token_2_sorted_distances


def cosine_similarity_vectors(A, B):
    dot_product = np.dot(A, B)
    norm_a = np.linalg.norm(A)
    norm_b = np.linalg.norm(B)
    similarity = dot_product / (norm_a * norm_b)
    return similarity


class InferDPTKit(object):

    def __init__(self, token_to_vector_dict, sorted_similarities, delta_f, tokenizer) -> None:
        self.token_to_vector_dict = token_to_vector_dict
        self.sorted_similarities = sorted_similarities
        self.delta_f = delta_f
        self.tokenizer = tokenizer
        assert len(token_to_vector_dict) == len(sorted_similarities)
    

    def save_to_path(self, path):

        # make folder
        import os
        if not os.path.exists(path+'/inferdpt_kit'):
            os.makedirs(path+'/inferdpt_kit')
        
        with open(path+'/inferdpt_kit/token_2_vector.json', 'w', encoding='utf8') as f:
            json.dump(self.token_to_vector_dict, f, ensure_ascii=False, cls=NumpyEncoder)

        with open(path+'/inferdpt_kit/sorted_similarities.json', 'w') as f:
            json.dump(self.sorted_similarities, f, cls=NumpyEncoder)

        with open(path+'/inferdpt_kit/delta_f.json', 'w') as f:
            json.dump(self.delta_f, f, cls=NumpyEncoder)

        self.tokenizer.save_pretrained(path+'/inferdpt_kit/tokenizer/')

    @staticmethod
    def make_inferdpt_kit_param(embedding_matrix: np.ndarray, token_list: List[str]):
        
        def cosine_simi(embedding_matrix1, embedding_matrix2):
            dot_product = np.dot(embedding_matrix1, embedding_matrix2.T)
            norm_matrix1 = np.linalg.norm(embedding_matrix1, axis=1)
            norm_matrix2 = np.linalg.norm(embedding_matrix2, axis=1)
            similarity = dot_product / (np.outer(norm_matrix1, norm_matrix2))

            return similarity
        assert len(embedding_matrix) == len(token_list)
        similarity_matrix = cosine_simi(embedding_matrix, embedding_matrix)
        token_sorted_distance_dict = create_sorted_embedding_matrix(token_list, similarity_matrix)
        delta_f_new = create_sensitivity_of_embeddings(embedding_matrix)

        token_2_embedding = {}
        for token, embedding in zip(token_list, embedding_matrix):
            token_2_embedding[token] = embedding

        return token_2_embedding, token_sorted_distance_dict, delta_f_new

    @staticmethod
    def load_from_path(path):
        
        with open(path+'/inferdpt_kit/token_2_vector.json', 'r', encoding='utf8') as f:
            token_to_vector_dict = json.load(f)
        with open(path+'/inferdpt_kit/sorted_similarities.json', 'r') as f:
            sorted_similarities = json.load(f)
        with open(path+'/inferdpt_kit/delta_f.json', 'r') as f:
            delta_f = np.array(json.load(f))
        tokenizer = AutoTokenizer.from_pretrained(path+'/inferdpt_kit/tokenizer/')
        inferdpt_kit = InferDPTKit(token_to_vector_dict, sorted_similarities, delta_f, tokenizer)
        return inferdpt_kit

    def perturb(self, doc: str, epsilon: float) -> str:
        
        # epsilon > 0
        assert epsilon > 0, "epsilon should be greater than 0"
        tokenizer = self.tokenizer
        tokens = tokenizer.tokenize(doc)
        new_tokens = []
        Delta_u = 1.0  
        exp_factor = epsilon / (2 * Delta_u)
        for origin_token in tokens:
            if origin_token[0] == ' ':
                origin_token = origin_token[1:]
            origin_embed = self.token_to_vector_dict.get(origin_token, None)
            if origin_embed is None:
                new_tokens.append(origin_token)
                continue
            noise_embed = add_laplace_noise_to_vector(origin_embed, epsilon, self.delta_f)
            similarity = cosine_similarity_vectors(origin_embed, noise_embed)
            sorted_distances_for_token = self.sorted_similarities.get(origin_token, None)
            if sorted_distances_for_token is None:
                continue
            token_only = sorted_distances_for_token[0]
            similarity_only = sorted_distances_for_token[1]
            arr = np.flip(similarity_only)
            index = np.searchsorted(arr, similarity)
            index = len(arr) - index
            close_tokens = token_only[:index]
            close_similarities = similarity_only[:index]
            if len(close_tokens) == 0:
                continue
            unnormalized_probabilities = np.exp(exp_factor * np.array(close_similarities))
            total_unnormalized_prob = np.sum(unnormalized_probabilities)
            probabilities = unnormalized_probabilities / total_unnormalized_prob
            selected_token = np.random.choice(close_tokens, p=probabilities)
            new_tokens.append(selected_token)
        token_ids = tokenizer.convert_tokens_to_ids(new_tokens)
        sentence = tokenizer.decode(token_ids)
        return sentence


def cosine_similarity_vectors(A, B):
    dot_product = np.dot(A, B)
    norm_a = np.linalg.norm(A)
    norm_b = np.linalg.norm(B)
    similarity = dot_product / (norm_a * norm_b)
    return similarity


def add_laplace_noise_to_vector(vector, epsilon, delta_f_new):
    vector = np.asarray(vector, dtype=np.longdouble)
    if epsilon == 0:
        beta_values = delta_f_new * 0
    else:
        beta_values = delta_f_new / (0.5 * epsilon)
    noise = np.random.laplace(loc=0, scale=beta_values, size=len(beta_values))
    noisy_vector = vector + noise

    return noisy_vector


def perturb_sentence(sent,
                     epsilon,
                     tokenizer,
                     token_to_vector_dict,
                     sorted_distance_data,
                     delta_f_new):
    tokens = tokenizer.tokenize(sent)
    new_tokens = []
    Delta_u = 1.0  
    exp_factor = epsilon / (2 * Delta_u)
    for origin_token in tokens:
        if origin_token[0] == ' ':
            origin_token = origin_token[1:]
        origin_embed = token_to_vector_dict.get(origin_token, None)
        if origin_embed is None:
            new_tokens.append(origin_token)
            continue
        noise_embed = add_laplace_noise_to_vector(origin_embed, epsilon, delta_f_new)
        similarity = cosine_similarity_vectors(origin_embed, noise_embed)
        sorted_distances_for_token = sorted_distance_data.get(origin_token, None)
        if sorted_distances_for_token is None:
            continue
        token_only = sorted_distances_for_token[0]
        similarity_only = sorted_distances_for_token[1]
        arr = np.flip(similarity_only)
        index = np.searchsorted(arr, similarity)
        index = len(arr) - index
        close_tokens = token_only[:index]
        close_similarities = similarity_only[:index]
        if len(close_tokens) == 0:
            continue
        unnormalized_probabilities = np.exp(exp_factor * np.array(close_similarities))
        total_unnormalized_prob = np.sum(unnormalized_probabilities)
        probabilities = unnormalized_probabilities / total_unnormalized_prob
        selected_token = np.random.choice(close_tokens, p=probabilities)
        new_tokens.append(selected_token)
    token_ids = tokenizer.convert_tokens_to_ids(new_tokens)
    sentence = tokenizer.decode(token_ids)
    return sentence
