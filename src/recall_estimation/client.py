import json
import os
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from elasticsearch.exceptions import BadRequestError, ConnectionError

ROOT_DIR = Path(os.getenv("ROOT_DIR", Path.cwd()))


class ESClient:
    """
    A simple Elasticsearch client to create an index and bulk-insert documents.
    """
    SIMILARITIES = [
        "l2_norm",
        "cosine", "max_inner_product"
        "max_inner_product"
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
    CANDIDATE_DEFAULT_VISIT_PERCENTAGE = [0.5, 0.75, 1.0, 1.25, 1.5]
    CANDIDATE_OVERSAMPLE = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0]

    def make_field_mapping(self,
                           similarity: str,
                           index_type: str,
                           m: int | None = None,
                           ef_construction: int | None = None,
                           default_visit_percentage: float | None = None,
                           oversample: float | None = None) -> dict[str, Any]:
        """Creates the field mapping for a dense_vector field with the specified index type
        and similarity.

        :param similarity: The similarity metric to use ("l2_norm", "cosine", or "max_inner_product").
        :param index_type: The index type to use (e.g., "hnsw", "int8_hnsw", etc.).
        :param m: The 'm' parameter for HNSW indices.
        :param ef_construction: The 'ef_construction' parameter for HNSW indices.
        :param default_visit_percentage: The default visit percentage for disk-based indices.
        :param oversample: The oversampling factor for disk-based indices.
        :return: A dictionary representing the field mapping.
        """
        index_options: dict[str, Any] = {"type": index_type}
        if index_type in ["hnsw", "int8_hnsw", "int4_hnsw", "bbq_hnsw"]:
            if m is not None:
                index_options["m"] = m
            if ef_construction is not None:
                index_options["ef_construction"] = ef_construction
        elif index_type in ["bbq_disk"]:
            if default_visit_percentage is not None:
                index_options["default_visit_percentage"] = default_visit_percentage

        if oversample is not None:
            index_options["rescore_vector"] = {"oversample": oversample}
        return {
            "type": "dense_vector",
            "index": True,
            "similarity": similarity,
            "index_options": index_options
        }


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

    def index_vectors(self,
            index_name: str,
            fvec_file_name: str,
            field_mapping: dict,
        ) -> None:
        """Indexes vectors from a .fvec file into the specified Elasticsearch index.

        :param index_name: The name of the Elasticsearch index.
        :param field_name: The name of the field to store the vectors.
        :param fvec_file_name: The path to the .fvec file containing the vectors.
        """

        def read_fvecs(fname):
            x = np.fromfile(fname, dtype='int32')
            d = x[0]
            x = x.reshape(-1, d + 1)[:, 1:].copy()
            x = x.view('float32')
            yield from x

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
            elif "error" in response:
                already_exists = response["error"].get("type") == "resource_already_exists_exception"
                if already_exists:
                    print(f"Index '{index_name}' already exists. Skipping creation.")
                    return
                else:
                    print(f"Error creating index '{index_name}': {response['error']}")
                    return
            else:
                print(f"Unknown response while creating index: {response}")
                return

        except Exception as e:
            # Catch other errors and skip this index
            print(f"Error creating index: {e}")
            return
    
        vectors = read_fvecs(fvec_file_name)

        actions = [
            {
                "_index": index_name,
                "_source": {"vec": vector.tolist()}
            }
            for vector in vectors
        ]

        print(f"Indexing {len(actions)} vectors into index '{index_name}'...")
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
            elif failed:
                print(f"Failed to index {failed} vectors.")
        except Exception as e:
            print(f"An error occurred during bulk indexing: {e}")

    def _https(self) -> str:
        return f"https://{self.host}:{self.port}"

