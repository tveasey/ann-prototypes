import itertools
import os
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from elasticsearch.exceptions import ConnectionError
from jsonargparse import auto_cli
from scipy.spatial.distance import cdist
from tqdm.auto import tqdm

ROOT_DIR = Path(os.getenv("ROOT_DIR", Path.cwd()))


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
    CANDIDATE_M = [8, 16, 24, 32, 40, 48, 56, 64]
    CANDIDATE_EF_CONSTRUCTION = [100, 200, 300, 400, 500]
    CANDIDATE_CLUSTER_SIZE = [64, 96, 128, 192, 256, 384]
    CANDIDATE_DEFAULT_VISIT_PERCENTAGE = [0.5, 0.75, 1.0, 1.25, 1.5]
    CANDIDATE_OVERSAMPLE = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0]

    def __init__(self) -> None:
        """Initializes the client and connects to Elasticsearch."""
        # Load the .env file for credentials if it exists
        load_dotenv()

        self.host = os.getenv("ELASTIC_HOST", "localhost")
        self.port = int(os.getenv("ELASTIC_PORT", "9200"))
        self.username = os.getenv("ELASTIC_USERNAME", "elastic")
        self.password = os.getenv("ELASTIC_PASSWORD", "")
        self.cert_fingerprint = os.getenv("CERT_FINGERPRINT", "")

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

    def recall_experiment(self,
                          index_name: str,
                          fvec_file_name: str,
                          sample_sizes: list[int | None],
                          field_mapping: dict,
                          query_params: list[dict]) -> list[float] | None:
        """Indexes vectors from a .fvec file into the specified Elasticsearch index.

        :param index_name: The name of the Elasticsearch index.
        :param fvec_file_name: The path to the .fvec file containing the vectors.
        :param sample_sizes: The list of numbers of vectors to sample from the .fvec file.
        :param field_mapping: The field mapping for the dense_vector field.
        :param query_params: The query parameters for the recall experiment.
        :return: A list of average recall values for each sample size, or None if an error occurs.
        """
        average_recalls = []
        for sample_size in sample_sizes:
            print(f"Running recall experiment with sample size: {sample_size}")

            sample_index_name = (f"{index_name}_{sample_size}"
                                 if sample_size is not None else index_name)
            if not self._create_index(sample_index_name, field_mapping):
                print(f"Skipping indexing for '{sample_index_name}' due to index creation failure.")
                return None

            corpus = self._read_fvecs(fvec_file_name, sample_size)

            if not self._index_vectors(sample_index_name, corpus):
                print(f"Indexing failed for index '{sample_index_name}'.")
                return None

            queries = self._random_samples(corpus, num_samples=500)

            brute_force_top_k = self._brute_force_indices(
                queries, corpus, similarity=field_mapping["similarity"], k=10
            )

            for query_param in query_params:
                print(f"Using query parameters: {query_param}")
                ann_top_k = self._ann_indices(
                    sample_index_name, queries, query_param
                )

                # Calculate recall
                total_recall = 0.0
                num_queries = queries.shape[0]
                for i in range(num_queries):
                    ann_set = set(ann_top_k[i])
                    brute_force_set = set(brute_force_top_k[i])
                    intersection_size = len(ann_set.intersection(brute_force_set))
                    recall = intersection_size / len(brute_force_set)
                    total_recall += recall

                average_recall = total_recall / num_queries
                print(f"Average recall: {average_recall:.4f}")
                average_recalls.append(average_recall)

        return average_recalls

    def delete_all_indices(self) -> None:
        """Deletes all indices in the Elasticsearch cluster."""
        try:
            indices = self.es.indices.get_alias(name="*")
            for index in indices:
                self.es.indices.delete(index=index)
                print(f"Deleted index: {index}")
            print("All indices have been deleted.")
        except Exception as e:
            print(f"Error deleting indices: {e}")

    def _https(self) -> str:
        return f"https://{self.host}:{self.port}"

    def _create_index(self, index_name: str, field_mapping: dict) -> bool:
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
                return True
            if "error" in response:
                if response["error"].get("type") == "resource_already_exists_exception":
                    print(f"Index '{index_name}' already exists. Skipping creation.")
                    return True
                print(f"Error creating index '{index_name}': {response['error']}")
                return False
            print(f"Unknown response while creating index: {response}")
            return False
        except Exception as e:
            # Catch other errors and skip this index
            print(f"Error creating index: {e}")
            return False

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
                       corpus: np.ndarray) -> bool:
        actions = (
            {
                "_index": index_name,
                "_id": f"{id}",
                "_source": {"vec": vector.tolist()}
            }
            for id, vector in enumerate(tqdm(
                corpus, total=len(corpus), desc=f"Indexing vectors into '{index_name}'")
            )
        )
        try:
            success, failed = bulk(
                self.es, actions, stats_only=True, chunk_size=10000
            )
            print(f"Successfully indexed {success} vectors.")
            if failed and (isinstance(failed, list) and len(failed) > 0):
                print(f"Failed to index {len(failed)} vectors.")
                # Log the first 5 failures for debugging
                for i, item in enumerate(failed[:5]):
                    print(f"  Failure {i+1}: {item['index']['error']}")
                return False
            if failed:
                print(f"Failed to index {failed} vectors.")
                return False
        except Exception as e:
            print(f"An error occurred during bulk indexing: {e}")
            return False
        return True

    def _random_samples(self, corpus: np.ndarray, num_samples: int) -> np.ndarray:
        """Selects random samples from the corpus.

        :param corpus: The corpus of vectors.
        :param num_samples: The number of random samples to select.
        :return: A numpy array of randomly selected samples.
        """
        indices = np.random.choice(corpus.shape[0], size=num_samples, replace=False)
        return corpus[indices]

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
                        "vec": {
                            "vector": query.tolist(),
                            **query_params
                        }
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
            dists = cdist(queries, corpus, metric='sqeuclidean')
            topk_indices = np.argpartition(dists, k, axis=1)[:, :k]
        elif similarity == "cosine":
            similarities = corpus @ queries / (
                np.linalg.norm(corpus, axis=1) * np.linalg.norm(queries, axis=1) + 1e-10
            )
            topk_indices = np.argpartition(-similarities, k, axis=1)[:, :k]
        elif similarity == "max_inner_product":
            inner_products = corpus @ queries
            topk_indices = np.argpartition(-inner_products, k, axis=1)[:, :k]
        else:
            raise ValueError(f"Unsupported similarity metric: {similarity}")
        return topk_indices

def _run_experiment(fvec_file_name: str,
                    similarity: str,
                    index_type: str,
                    clear_caches: bool,
                    m: int | None = None,
                    ef_construction: int | None = None,
                    cluster_size: int | None = None,
                    k: int | None = None,
                    visit_percentage: list[float] | None = None,
                    oversample: list[float] | None = None) -> None:
    client = ExperimentFramework()

    if clear_caches:
        client.delete_all_indices()

    # Concatenate the string representations of the collection name and build parameters
    # (If the build parameters are different we shouldn't reuse the same index name.)
    index_name = (
        f"{Path(fvec_file_name).stem}_{index_type}_{m}_{ef_construction}_{cluster_size}"
    )

    field_mapping = client.make_field_mapping(
        similarity=similarity,
        index_type=index_type,
        m=m,
        ef_construction=ef_construction,
        cluster_size=cluster_size
    )

    query_params = [
        client.make_query_params(
            k=k or 10,
            visit_percentage=visit_percentage,
            oversample=oversample
        )
        for visit_percentage, oversample in itertools.product(
            visit_percentage or [None], # type: ignore
            oversample or [None] # type: ignore
        )
    ]

    recalls = client.recall_experiment(
        index_name=index_name,
        fvec_file_name=fvec_file_name,
        sample_sizes=[None, 5000, 6000, 7000, 8000, 9000, 10000],
        field_mapping=field_mapping,
        query_params=query_params
    )
    print(f"Experiment completed for index '{index_name}'/{index_type}.")
    print(f"Recalls: {recalls}")

def main(fvec_file_name: str,
         similarity: str,
         k: int | None = 10,
         run_all: bool = True,
         clear_caches: bool = True,
         index_type: str = "hnsw",
         m: int | None = None,
         ef_construction: int | None = None,
         cluster_size: int | None = None,
         visit_percentage: float | None = None,
         oversample: float | None = None) -> None:
    """Main function to run recall estimation experiments.

    :param fvec_file_name: The path to the .fvec file containing the vectors.
    :param similarity: The similarity metric to use.
    :param run_all: Whether to run experiments for all parameter combinations.
    :param clear_caches: Whether to clear all existing indices before running the experiment.
    :param index_type: The index type to use.
    :param m: The 'm' parameter for HNSW indices.
    :param ef_construction: The 'ef_construction' parameter for HNSW indices.
    :param cluster_size: The 'cluster_size' parameter for BBQ indices.
    :param k: The number of nearest neighbors to retrieve.
    :return: None
    """
    if run_all:
        # Create an outer product of all parameter combinations iterator

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
        for index_type_, m_, ef_construction_ in hnsw_experiments:
            _run_experiment(
                fvec_file_name=fvec_file_name,
                similarity=similarity,
                index_type=index_type_,
                k=k,
                clear_caches=clear_caches,
                m=m_,
                ef_construction=ef_construction_,
                cluster_size=None,
                visit_percentage=None,
                oversample=ExperimentFramework.CANDIDATE_OVERSAMPLE,
            )
        for index_type_ in flat_experiments:
            _run_experiment(
                fvec_file_name=fvec_file_name,
                similarity=similarity,
                index_type=index_type_,
                clear_caches=clear_caches,
                m=None,
                ef_construction=None,
                cluster_size=None,
                k=k,
                visit_percentage=None,
                oversample=ExperimentFramework.CANDIDATE_OVERSAMPLE,
            )
        for index_type_, cluster_size_ in disk_experiments:
            _run_experiment(
                fvec_file_name=fvec_file_name,
                similarity=similarity,
                index_type=index_type_,
                clear_caches=clear_caches,
                m=None,
                ef_construction=None,
                cluster_size=cluster_size_,
                k=k,
                visit_percentage=ExperimentFramework.CANDIDATE_DEFAULT_VISIT_PERCENTAGE,
                oversample=ExperimentFramework.CANDIDATE_OVERSAMPLE,
            )
        return

    _run_experiment(
        fvec_file_name=fvec_file_name,
        similarity=similarity,
        index_type=index_type,
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
