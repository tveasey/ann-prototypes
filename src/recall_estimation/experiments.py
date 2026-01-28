import itertools
import json
import os
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from elasticsearch.helpers import streaming_bulk
from elasticsearch.exceptions import ConnectionError
from jsonargparse import auto_cli
from scipy.spatial.distance import cdist
from tqdm.auto import tqdm

ROOT_DIR = Path(os.getenv("ROOT_DIR", Path.cwd()))

def flatten_dict(d: dict, parent_key: str = '', sep: str = '_') -> dict:
    """Flattens a nested dictionary.

    :param d: The dictionary to flatten.
    :param parent_key: The base key string for recursion.
    :param sep: The separator between keys.
    :return: A flattened dictionary.
    """
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

class ExperimentFramework:
    """
    Client for performing ANN recall estimation experiments with Elasticsearch.
    """

    SIMILARITIES = [
        "l2_norm",
        "cosine",
        "max_inner_product",
    ]
    INDEX_TYPES = [
        "hnsw",
        "int8_hnsw",
        "int4_hnsw",
        "bbq_hnsw",
        "int8_flat",
        "int4_flat",
        "bbq_flat",
        "bbq_disk"
    ]
    CANDIDATE_M = [8, 16, 32, 64]
    CANDIDATE_EF_CONSTRUCTION = [100, 200, 400]
    CANDIDATE_CLUSTER_SIZE = [64, 96, 128, 192, 256, 384]
    CANDIDATE_DEFAULT_VISIT_PERCENTAGE = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
    CANDIDATE_OVERSAMPLE = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0]

    def __init__(self, seed: int, num_queries: int = 500) -> None:
        """Initializes the client and connects to Elasticsearch."""
        np.random.seed(seed)

        # Load the .env file for credentials if it exists
        load_dotenv()

        self.host = os.getenv("ELASTIC_HOST", "localhost")
        self.port = int(os.getenv("ELASTIC_PORT", "9200"))
        self.username = os.getenv("ELASTIC_USERNAME", "elastic")
        self.password = os.getenv("ELASTIC_PASSWORD", "")
        self.cert_fingerprint = os.getenv("CERT_FINGERPRINT", "")
        self.num_queries = num_queries

        self.es = Elasticsearch(self._https(),
                                ssl_assert_fingerprint=self.cert_fingerprint,
                                basic_auth=(self.username, self.password))
        try:
            if not self.es.ping():
                print("Error: Could not connect to Elasticsearch.")
                print(f"Please ensure Elasticsearch is running on {self._https()}")
                raise ConnectionError(f"Could not connect to Elasticsearch at {self._https()}")
            print(f"Successfully connected to Elasticsearch at {self._https()}")
        except ConnectionError as e:
            print(f"Error: Connection to {self._https()} refused.")
            print(f"Details: {e}")
            print("Please ensure Elasticsearch is running.")
            raise

    def make_field_mapping(self,
                           similarity: str,
                           index_type: str,
                           m: int | None = None,
                           ef_construction: int | None = None,
                           cluster_size: int | None = None) -> dict[str, Any]:
        """Creates the field mapping for a dense_vector field with the specified index
        type and similarity.

        :param similarity: The similarity metric to use ("l2_norm", "cosine", or "max_inner_product").
        :param index_type: The index type to use (e.g., "hnsw", "int8_hnsw", etc.).
        :param m: The 'm' parameter for HNSW indices.
        :param ef_construction: The 'ef_construction' parameter for HNSW indices.
        :param cluster_size: The 'cluster_size' parameter for BBQ indices.
        :return: A dictionary representing the field mapping.
        """
        index_options: dict[str, Any] = {"type": index_type}
        if index_type in ["hnsw", "int8_hnsw", "int4_hnsw", "bbq_hnsw"]:
            if m is not None:
                index_options["m"] = m
            if ef_construction is not None:
                index_options["ef_construction"] = ef_construction
        if index_type in ["bbq_disk"]:
            if cluster_size is not None:
                index_options["cluster_size"] = cluster_size
        return {
            "type": "dense_vector",
            "index": True,
            "similarity": similarity,
            "index_options": index_options
        }

    def make_query_params(self,
                          k: int,
                          visit_percentage: float | None = None,
                          oversample: float | None = None) -> dict[str, Any]:
        """Creates the query parameters for an approximate nearest neighbor search.

        :param k: The number of nearest neighbors to retrieve.
        :param visit_percentage: The percentage of the index to visit during the search.
        :param oversample: The oversampling factor for the search.
        :return: A dictionary representing the query parameters.
        """
        query_params: dict[str, Any] = {"k": k}
        if visit_percentage is not None:
            query_params["visit_percentage"] = visit_percentage
        if oversample is not None:
            query_params["rescore_vector"] = {"oversample": oversample}
        return query_params

    def experiment_hash(self,
                        index_name: str,
                        field_mapping: dict,
                        query_params: dict) -> str:
        """Generates a unique hash for the experiment configuration.

        :param index_name: The name of the Elasticsearch index.
        :param field_mapping: The field mapping for the dense_vector field.
        :param query_params: The query parameters for the recall experiment.
        :return: A SHA-256 hash string representing the experiment configuration.
        """
        hash_input = f"{index_name}_{field_mapping}_{query_params}"
        return sha256(hash_input.encode()).hexdigest()

    def sampling_recall_experiment(self,
                                   index_name: str,
                                   fvec_file_name: str,
                                   sample_sizes: list[int | None],
                                   field_mapping: dict,
                                   k: int,
                                   query_params: list[dict]) -> list[dict] | None:
        """Indexes vectors from a .fvec file into the specified Elasticsearch index.

        :param index_name: The name of the Elasticsearch index.
        :param fvec_file_name: The path to the .fvec file containing the vectors.
        :param sample_sizes: The list of numbers of vectors to sample from the .fvec file.
        :param field_mapping: The field mapping for the dense_vector field.
        :param k: The number of nearest neighbors to retrieve.
        :param query_params: The query parameters for the recall experiment.
        :return: A list of average recall values for each sample size and query parameter set,
        or None if an error occurs.
        """
        average_recalls = []
        for sample_size in sample_sizes:
            print(f"Running recall experiments with sample size: {sample_size}")

            sample_index_name = (f"{index_name}_{sample_size}"
                                 if sample_size is not None else index_name)
            created, already_exists = self._create_index(sample_index_name, field_mapping)
            if not created:
                print(f"Skipping indexing for '{sample_index_name}' due to index creation failure.")
                return None

            corpus = self._read_fvecs(fvec_file_name, sample_size)
            corpus, queries = self._random_queries(corpus, num_queries=self.num_queries)

            if not already_exists and not self._index_vectors(sample_index_name, corpus):
                print(f"Indexing failed for index '{sample_index_name}'.")
                return None

            brute_force_top_k = self._brute_force_indices(
                queries, corpus, similarity=field_mapping["similarity"], k=k
            )

            for params in query_params:
                ann_top_k = self._ann_indices(sample_index_name, queries, params)

                # Calculate recall
                recalls = []
                num_queries = queries.shape[0]
                for i in range(num_queries):
                    ann_set = set(ann_top_k[i].tolist())
                    brute_force_set = set(brute_force_top_k[i].tolist())
                    intersection_size = len(ann_set.intersection(brute_force_set))
                    recall = intersection_size / len(brute_force_set)
                    recalls.append(recall)
                
                average_recall = np.mean(recalls)
                recall_std = np.std(recalls)
                result = {
                    "hash": self.experiment_hash(sample_index_name, field_mapping, params),
                    "index_name": sample_index_name,
                    **flatten_dict({"index_params": field_mapping["index_options"]}),
                    **flatten_dict({"query_params": params}),
                    "sample_size": sample_size,
                    "average_recall": average_recall,
                    "recall_std": recall_std
                }
                average_recalls.append(result)

        return average_recalls

    def canary_recall_experiment(self,
                                 index_name: str,
                                 fvec_file_name: str,
                                 field_mapping: dict,
                                 k: int,
                                 query_params: list[dict]) -> list[dict] | None:
        """Performs a canary-based recall estimation experiment.

        :param index_name: The name of the Elasticsearch index.
        :param fvec_file_name: The path to the .fvec file containing the vectors.
        :param field_mapping: The field mapping for the dense_vector field.
        :param k: The number of nearest neighbors to retrieve.
        :param query_params: The query parameters for the recall experiment.
        :return: A list of average recall values for each query parameter set,
        or None if an error occurs.
        """

        # The idea is to construct known nearest neighbour sets for canary queries and use
        # these to estimate recall for other vectors. We do this as follows:
        #   1. Index random half of the corpus.
        #   2. Pick random query vectors from the other 0.25 of the corpus.
        #   3. For each query vector, find its approximate 4 * sqrt(dim) nearest neighbors
        #      in the indexed half.
        #   4. Compute the minimum distance d_min of any vector found to the query vector.
        #   5. For each query, create a set of k "vectors" by sampling pairs (i, j) i < j
        #      from its nearest neighbors and constructing a * v_i + (1 - a) * v_j where "a"
        #      is random in [0, 1] and normalizing.
        #   6. For each query, uniformly sample inside a sphere which centered on the query
        #      along each neighbour vector to create its nearest neighbours.
        #   7. Index all remaining vectors (including canary nearest neighbours).
        #   8. Measure recall over the canary nearest neighbour sets.

        if not self._create_index(index_name, field_mapping)[0]:
            print(f"Skipping indexing for '{index_name}' due to index creation failure.")
            return None

        corpus = self._read_fvecs(fvec_file_name, sample_size=None)
        np.random.shuffle(corpus)
        sample_index_corpus = corpus[:corpus.shape[0] // 4]
        if not self._index_vectors(index_name, sample_index_corpus):
            print(f"Indexing failed for index '{index_name}'.")
            return None

        sample_size = corpus.shape[0] // 4
        remaining_corpus = corpus[sample_size:]
        _, query_vectors = self._random_queries(remaining_corpus, num_queries=self.num_queries)

        dim = corpus.shape[1]
        num_neighbors = int(4 * np.sqrt(dim))
        ann_top_k = self._ann_indices(index_name, query_vectors, query_params={"k": num_neighbors})

        canary_neighbors = []
        for i, query in enumerate(tqdm(
            query_vectors, total=len(query_vectors), desc="Generating canary neighbors"
        )):
            neighbor_indices = ann_top_k[i]
            neighbor_vectors = sample_index_corpus[neighbor_indices]
            distances = cdist(
                query.reshape(1, -1), neighbor_vectors, metric='euclidean'
            ).flatten()
            d_min = np.min(distances)

            canary_set: list[np.ndarray] = []
            while len(canary_set) < k:
                i, j = np.random.choice(len(neighbor_vectors), size=2, replace=False)
                a = np.random.uniform(0, 1)
                avg_vector = a * neighbor_vectors[i] + (1 - a) * neighbor_vectors[j]
                avg_vector /= np.linalg.norm(avg_vector) + 1e-10
                # The volume of a d-dimensional sphere scales as r^d so to get a uniform
                # distribution within the sphere we need to sample with probability
                # proportional to r^(d-1). This can be done using the inverse transform
                # method.
                scale = np.random.uniform(0.1 * d_min, 0.9 * d_min) * 0.95**dim
                scale = scale ** (1 / dim)
                canary_vector = query + scale * avg_vector
                canary_set.append(canary_vector)
            canary_neighbors.append(np.array(canary_set))

        # We want to randomly spread the canary neighbors throughout the remaining corpus
        # at index time to avoid any indexing edge case.
        remaining_corpus = np.vstack([remaining_corpus] + canary_neighbors)
        indices = np.array(range(sample_index_corpus.shape[0], 
                                 sample_index_corpus.shape[0] + remaining_corpus.shape[0]))
        np.random.shuffle(indices)
        inverse_indices = np.empty_like(indices)
        inverse_indices[indices] = np.arange(len(indices))
        if not self._index_vectors(index_name, remaining_corpus, 
                                  offset=sample_index_corpus.shape[0]):
            print(f"Indexing failed for index '{index_name}'.")
            return None

        exact_top_k = []
        for i in range(len(query_vectors)):
            canary_indices = {
                inverse_indices[sample_index_corpus.shape[0] + i * k + j]
                for j in range(k)
            }
            exact_top_k.append(canary_indices)

        average_recalls = []
        for params in query_params:
            recalls = []
            ann_top_k = self._ann_indices(index_name, query_vectors, query_params=params)
            for i in range(len(query_vectors)):
                ann_set = set(ann_top_k[i].tolist())
                intersection_size = len(ann_set.intersection(exact_top_k[i]))
                recall = intersection_size / len(exact_top_k[i])
                recalls.append(recall)

            average_recall = np.mean(recalls)
            recall_std = np.std(recalls)
            result = {
                "hash": self.experiment_hash(index_name, field_mapping, params),
                "index_name": index_name,
                **flatten_dict({"index_params": field_mapping["index_options"]}),
                **flatten_dict({"query_params": params}),
                "sample_size": sample_size,
                "average_recall": average_recall,
                "recall_std": recall_std
            }
            average_recalls.append(result)

        return average_recalls

    def delete_all_indices(self, dry_run: bool = False) -> None:
        """Deletes all indices in the Elasticsearch cluster."""
        try:
            # Fetch all indices
            # We use "*" to get everything, then filter in Python
            indices_to_delete = self.es.indices.get_alias(index="*,-.*").keys()

            if not indices_to_delete:
                print("\nNo non-system indices found to delete.")
                return
            if dry_run:
                print("Dry run mode: The following indices would be deleted:")
                for index in indices_to_delete:
                    print(f"  {index}")
                return
            for idx in indices_to_delete:
                try:
                    self.es.indices.delete(index=idx)
                    print(f"✔ Deleted: {idx}")
                except Exception as e:
                    print(f"✖ Failed to delete {idx}: {e}")
        except Exception as e:
            print(f"Error fetching indices: {e}")

    def _https(self) -> str:
        return f"https://{self.host}:{self.port}"

    def _create_index(self,
                      index_name: str,
                      field_mapping: dict) -> tuple[bool, bool]:
        # Create the mappings for the index
        mappings = {
            "properties": {
                "vec": {
                    "type": "dense_vector",
                    "index": True,
                    **field_mapping
                }
            }
        }
        try:
            # Create the index (ignore 400 errors if the index already exists)
            response = self.es.options(ignore_status=[400]).indices.create(
                index=index_name, mappings=mappings
            )
            print(f"Create index response: {response}")
            if "acknowledged" in response and response["acknowledged"]:
                print(f"Index '{index_name}' created successfully.")
                return True, False
            if "error" in response:
                if response["error"].get("type") == "resource_already_exists_exception":
                    print(f"Index '{index_name}' already exists. Skipping creation.")
                    return True, True
                print(f"Error creating index '{index_name}': {response['error']}")
                return False, False
            print(f"Unknown response while creating index: {response}")
            return False, False
        except Exception as e:
            # Catch other errors and skip this index
            print(f"Error creating index: {e}")
            return False, False

    def _read_fvecs(self,
                    fvec_file_name: str,
                    sample_size: int | None) -> np.ndarray:
        x = np.fromfile(fvec_file_name, dtype='int32')
        d = x[0]
        x = x.reshape(-1, d + 1)[:, 1:].copy()
        x = x.view('float32')
        if sample_size is not None and sample_size < x.shape[0]:
            selected_indices = np.random.choice(
                x.shape[0], size=sample_size, replace=False
            )
            x = x[selected_indices]
        return x

    def _index_vectors(self,
                       index_name: str,
                       corpus: np.ndarray,
                       offset: int = 0) -> bool:
        actions = (
            {
                "_index": index_name,
                "_id": f"{offset + id}",
                "_source": {"vec": vector.tolist()}
            }
            for id, vector in enumerate(tqdm(
                corpus, total=len(corpus), desc=f"Indexing vectors into '{index_name}'")
            )
        )
        for success, info in streaming_bulk(
            self.es.options(request_timeout=120),
            actions,
            chunk_size=10000,
            raise_on_error=False
        ):
            if not success:
                print(f"Failed to index document: {info}")
                return False
        # Ensure that the index is refreshed before searching.
        self.es.indices.refresh(index=index_name)
        return True

    def _random_queries(self,
                        corpus: np.ndarray,
                        num_queries: int) -> tuple[np.ndarray, np.ndarray]:
        """Selects random queries from the corpus.

        :param corpus: The corpus of vectors.
        :param num_queries: The number of random queries to select.
        :return: A tuple (x, y) of randomly selected queries y and corpus minus those queries x.
        """
        indices = np.random.choice(corpus.shape[0], size=num_queries, replace=False)
        return corpus[np.setdiff1d(np.arange(corpus.shape[0]), indices)], corpus[indices]

    def _ann_indices(self,
                     index_name: str,
                     queries: np.ndarray,
                     query_params: dict) -> np.ndarray:
        """Performs an approximate nearest neighbor search using Elasticsearch.

        :param index_name: The name of the Elasticsearch index.
        :param queries: The query vectors.
        :param k: The number of nearest neighbors to retrieve.
        :return: A list of document IDs representing the nearest neighbors.
        """
        # We use the position of each vector in the corpus collection as its document ID.
        # Thus, the brute-force top-k indices correspond directly to document IDs.
        ann_indices = []
        for query in tqdm(
            queries, total=len(queries), desc="Performing ANN searches"
        ):
            query_body = {
                "size": query_params.get("k", 10),
                "query": {
                    "knn": {
                        "field": "vec",
                        "query_vector": query.tolist(),
                        **query_params
                    }
                }
            }
            response = self.es.search(index=index_name, body=query_body)
            hits = response["hits"]["hits"]
            ann_indices.append([int(hit["_id"]) for hit in hits])
        return np.array(ann_indices)

    def _brute_force_indices(self,
                             queries: np.ndarray,
                             corpus: np.ndarray,
                             similarity: str,
                             k: int) -> np.ndarray:
        """Performs a brute-force search to find the top-k nearest neighbors.

        :param corpus: The corpus of vectors.
        :param queries: The query vectors.
        :param similarity: The similarity metric to use.
        :param k: The number of nearest neighbors to retrieve.
        :return: A list of document IDs representing the nearest neighbors.
        """
        if similarity == "l2_norm":
            scores = 1 / (1 + cdist(queries, corpus, metric='sqeuclidean'))
            topk_indices = np.argpartition(-scores, k, axis=1)[:, :k]
        elif similarity in ["cosine", "dot_product"]:
            # Compute the outer product of the vector norms
            norm_product = np.outer(np.linalg.norm(queries, axis=1),
                                    np.linalg.norm(corpus, axis=1)) + 1e-10
            # Match Lucene's cost function for cosine similarity
            scores = (0.5 + 0.5 * queries @ corpus.T / norm_product)
            topk_indices = np.argpartition(-scores, k, axis=1)[:, :k]
        elif similarity == "max_inner_product":
            # The score is 1 / (1 - inner_product) if inner_product < 0
            # otherwise it's 1 + inner_product
            scores = np.where(
                queries @ corpus.T < 0,
                1 / (1 - (queries @ corpus.T)),
                1 + (queries @ corpus.T)
            )
            topk_indices = np.argpartition(-scores, k, axis=1)[:, :k]
        else:
            raise ValueError(f"Unsupported similarity metric: {similarity}")
        return topk_indices

def _run_experiments(fvec_file_name: str,
                     similarity: str,
                     index_type: str,
                     clear_caches: bool,
                     hashes: set[str] | None = None,
                     seed: int = 42,
                     m: int | None = None,
                     ef_construction: int | None = None,
                     cluster_size: int | None = None,
                     k: int | None = None,
                     visit_percentage: list[float] | None = None,
                     oversample: list[float] | None = None) -> list[dict]:

    client = ExperimentFramework(seed=seed)

    if clear_caches:
        client.delete_all_indices()

    # Concatenate the string representations of the collection name and build parameters
    # (If the build parameters are different we shouldn't reuse the same index name.)
    index_name = (
        f"{Path(fvec_file_name).stem}_{index_type}_{m}_{ef_construction}_{cluster_size}"
    ).lower()

    field_mapping = client.make_field_mapping(
        similarity=similarity,
        index_type=index_type,
        m=m,
        ef_construction=ef_construction,
        cluster_size=cluster_size
    )

    def _base_query_params():
        yield from (
            client.make_query_params(
                k=k or 10,
                visit_percentage=visit_percentage,
                oversample=oversample
            )
            for visit_percentage, oversample in itertools.product(
                visit_percentage or [None], # type: ignore
                oversample or [None] # type: ignore
            )
        )

    def _hash(index_name_: str, params: dict) -> str:
        return client.experiment_hash(
            index_name=index_name_,
            field_mapping=field_mapping,
            query_params=params
        )

    recalls = []

    # Remove any experiments that have already been run
    non_null_hashes: set = hashes or set()
    query_params = [
        params for params in _base_query_params()
        if _hash(index_name, params) not in non_null_hashes
    ]
    if not query_params:
        print(f"Skipping all sampling experiments for index '{index_name}' ('{index_type}') "
              "as they have already been run.")
    else:
        recalls += client.sampling_recall_experiment(
            index_name=index_name,
            fvec_file_name=fvec_file_name,
            sample_sizes=[None, 2000, 4000, 8000, 16000],
            field_mapping=field_mapping,
            k=k or 10,
            query_params=query_params
        ) or []

    query_params = [
        params for params in _base_query_params()
        if _hash(f"{index_name}_canary", params) not in non_null_hashes
    ]
    if not query_params:
        print(f"Skipping all canary experiments for index '{index_name}' ('{index_type}') "
              "as they have already been run.")
    else:
        recalls += client.canary_recall_experiment(
            index_name=f"{index_name}_canary",
            fvec_file_name=fvec_file_name,
            field_mapping=field_mapping,
            k=k or 10,
            query_params=query_params
        ) or []

    print(f"Experiments completed for index '{index_name}' ('{index_type}').")
    print(f"Recalls: {json.dumps(recalls, indent=2)}")
    return recalls

def _visualize_correlation(results: pd.DataFrame) -> None:
    """Calculates the percentage difference between the recall estimate and the exact recall.

    :param recall_estimate: The estimated recall from the ANN search.
    :param exact_recall: The exact recall from the brute-force search.
    :return: The percentage difference between the two recall values.
    """
    from matplotlib import pyplot as plt

    # Get the unique non-nan values of sample_size
    unique_sample_sizes = results["sample_size"].unique()
    unique_sample_sizes = [s for s in unique_sample_sizes if pd.notna(s)]

    # Create a map "comparable experiment" -> [(sample size, average recall)]
    parameter_columns = [
        "source_file",
        "index_params_type",
        "index_params_m",
        "index_params_ef_construction",
        "index_params_cluster_size",
        "query_params_k",
        "query_params_rescore_vector_oversample",
        "query_params_visit_percentage",
    ]
    results[parameter_columns] = results[parameter_columns].fillna("n/a")
    results["sample_size"] = results["sample_size"].fillna("full")
    for i, sample_size in enumerate(unique_sample_sizes):
        plt.figure(i, figsize=(12, 8))
    for i, (filter_results, label) in enumerate([
        (results, "All"),
        (results[((results["index_params_type"] == "hnsw") |
                  (results["index_params_type"] == "int8_hnsw") |
                  (results["index_params_type"] == "int4_hnsw") |
                  (results["index_params_type"] == "bbq_hnsw"))], "HNSW"),
        (results[((results["index_params_type"] == "int8_flat") |
                  (results["index_params_type"] == "int4_flat") |
                  (results["index_params_type"] == "bbq_flat"))], "Flat"),
        (results[(results["index_params_type"] == "bbq_disk")], "DiskBBQ"),
    ]):
        pivot_results = filter_results.pivot_table(
            index=parameter_columns,
            columns="sample_size",
            values="average_recall"
        )
        all_results = {}
        for col in pivot_results.columns:
            all_results[col] = pivot_results[col].dropna().to_dict()

        # Create a scatter plot of recall for sample size vs full for each sample size.
        for j, sample_size in enumerate(unique_sample_sizes):
            sample_size_results = all_results.get(sample_size, {})
            full_results = all_results.get("full", {})
            estimated_recalls = []
            exact_recalls = []
            for params in sample_size_results.keys():
                if params in full_results:
                    estimated_recalls.append(sample_size_results[params])
                    exact_recalls.append(full_results[params])
            # Compute R^2 for the scatter plot
            r2 = np.corrcoef(np.array(estimated_recalls), np.array(exact_recalls))[0, 1] ** 2
            fig = plt.figure(j)
            fig.add_subplot(2, 2, i + 1)
            plt.scatter(estimated_recalls, exact_recalls, alpha=0.6, label=label)
            plt.plot([0, 1], [0, 1], 'r--')
            plt.xlabel(f"Estimated Recall (Sample Size = {sample_size})")
            plt.ylabel("Exact Recall (Full Dataset)")
            plt.title(f"Recall Estimate vs Exact Recall (R² = {r2:.4f})")
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.legend()
    plt.show()

def main(fvec_file_name: str | None = None,
         similarity: str | None = None,
         seed: int = 42,
         k: int | None = 10,
         all_params: bool = True,
         visualize: bool = False,
         clear_caches: bool = True,
         index_type: str = "hnsw",
         m: int | None = None,
         ef_construction: int | None = None,
         cluster_size: int | None = None,
         visit_percentage: float | None = None,
         oversample: float | None = None) -> None:
    """Main function to run recall estimation experiments.

    Args:
        fvec_file_name: The path to the .fvec file containing the vectors.
        similarity: The similarity metric to use.
        seed: The random seed for reproducibility.
        k: The number of nearest neighbors to retrieve.
        all_params: Whether to run experiments for all parameter combinations.
        visualize: Whether to visualize existing experiment results.
        clear_caches: Whether to clear all existing indices before running the experiment.
        index_type: The index type to use.
        m: The 'm' parameter for HNSW indices.
        ef_construction: The 'ef_construction' parameter for HNSW indices.
        cluster_size: The 'cluster_size' parameter for BBQ indices.
        visit_percentage: The percentage of the index to visit during the search.
        oversample: The oversampling factor for the search.
    """

    if visualize:
        # If fvec_file_name is not supplied, load all files in the results directory
        if fvec_file_name is None:
            results_files = list((Path("results")).glob("*_experiment_results.csv"))
            if not results_files:
                print("No results files found in the 'results' directory.")
                return
            all_results = pd.DataFrame()
            for file in results_files:
                df = pd.read_csv(file)
                # Add a column for the source file
                df["source_file"] = file.name
                all_results = pd.concat([all_results, df], ignore_index=True)
            _visualize_correlation(all_results)
            return

        output_file = Path("results") / (Path(fvec_file_name).stem + "_experiment_results.csv")
        if not output_file.exists():
            print(f"Output file '{output_file}' does not exist.")
            return

        _visualize_correlation(pd.read_csv(output_file))
        return

    if fvec_file_name is None or similarity is None:
        print("Error: fvec_file_name and similarity must be specified when not visualizing.")
        return

    if all_params:
        output_file = Path("results") / (Path(fvec_file_name).stem + "_experiment_results.csv")
        def _flush_results(initial_results: pd.DataFrame,
                           results: list[dict],
                           last_output_size: int) -> int:
            if len(results) - last_output_size >= 50:
                df = pd.concat([initial_results, pd.DataFrame(results)], ignore_index=True)
                df.to_csv(output_file, index=False)
                last_output_size = len(results)
                print(f"Intermediate results saved to {output_file}")
            return last_output_size

        hnsw_experiments = itertools.product(
            ["hnsw", "int8_hnsw", "int4_hnsw", "bbq_hnsw"],
            ExperimentFramework.CANDIDATE_M,
            ExperimentFramework.CANDIDATE_EF_CONSTRUCTION
        )
        flat_experiments = [
            "int8_flat", "int4_flat", "bbq_flat"
        ]
        disk_experiments = itertools.product(
            ["bbq_disk"],
            ExperimentFramework.CANDIDATE_CLUSTER_SIZE
        )

        initial_results = (
            pd.read_csv(output_file) if output_file.exists() else pd.DataFrame()
        )
        hashes = set(initial_results["hash"].to_list()) if not initial_results.empty else None

        results: list[dict] = []
        last_output_size = 0
        for index_type_, m_, ef_construction_ in hnsw_experiments:
            results += _run_experiments(
                fvec_file_name=fvec_file_name,
                similarity=similarity,
                index_type=index_type_,
                clear_caches=clear_caches,
                hashes=hashes,
                seed=seed,
                k=k,
                m=m_,
                ef_construction=ef_construction_,
                cluster_size=None,
                visit_percentage=None,
                oversample=ExperimentFramework.CANDIDATE_OVERSAMPLE,
            )
            last_output_size = _flush_results(initial_results, results, last_output_size)

        for index_type_ in flat_experiments:
            results += _run_experiments(
                fvec_file_name=fvec_file_name,
                similarity=similarity,
                index_type=index_type_,
                clear_caches=clear_caches,
                hashes=hashes,
                seed=seed,
                m=None,
                ef_construction=None,
                cluster_size=None,
                k=k,
                visit_percentage=None,
                oversample=ExperimentFramework.CANDIDATE_OVERSAMPLE,
            )
            last_output_size = _flush_results(initial_results, results, last_output_size)

        for index_type_, cluster_size_ in disk_experiments:
            results += _run_experiments(
                fvec_file_name=fvec_file_name,
                similarity=similarity,
                index_type=index_type_,
                seed=seed,
                clear_caches=clear_caches,
                hashes=hashes,
                m=None,
                ef_construction=None,
                cluster_size=cluster_size_,
                k=k,
                visit_percentage=ExperimentFramework.CANDIDATE_DEFAULT_VISIT_PERCENTAGE,
                oversample=ExperimentFramework.CANDIDATE_OVERSAMPLE,
            )
            last_output_size = _flush_results(initial_results, results, last_output_size)

        print(f"All experiments completed. Have {len(results)} results")

        # Concatenate with the initial results and write out as a pandas DataFrame
        df = pd.concat([initial_results, pd.DataFrame(results)], ignore_index=True)
        df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
        return

    # Single run with specified parameter
    print("Running single experiment with specified parameters.")
    print(f"Index type: {index_type}, m: {m}, ef_construction: {ef_construction}, "
          f"cluster_size: {cluster_size}, visit_percentage: {visit_percentage}, "
          f"oversample: {oversample}")
    _run_experiments(
        fvec_file_name=fvec_file_name,
        similarity=similarity,
        index_type=index_type,
        seed=seed,
        clear_caches=clear_caches,
        m=m,
        ef_construction=ef_construction,
        cluster_size=cluster_size,
        k=k,
        visit_percentage=[visit_percentage] if visit_percentage is not None else None,
        oversample=[oversample] if oversample is not None else None
    )

if __name__ == "__main__":
    auto_cli(main)
