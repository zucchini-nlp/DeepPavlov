import faiss
import numpy as np
import torch
from tqdm import trange
from transformers import AutoTokenizer
from bpr import BiEncoder, FaissBinaryIndex, FaissIndex, Retriever
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.serializable import Serializable


class Retriever(object):
    def __init__(self, index: FaissIndex, biencoder: BiEncoder):
        self.index = index
        self._biencoder = biencoder
        self._tokenizer = AutoTokenizer.from_pretrained(biencoder.hparams.base_pretrained_model, use_fast=True)

    def encode_queries(self, queries, batch_size: int = 256) -> np.ndarray:
        embeddings = []
        with torch.no_grad():
            for start in trange(0, len(queries), batch_size):
                model_inputs = self._tokenizer.batch_encode_plus(
                    queries[start : start + batch_size],
                    return_tensors="pt",
                    max_length=self._biencoder.hparams.max_query_length,
                    pad_to_max_length=True,
                )

                model_inputs = {k: v.to(self._biencoder.device) for k, v in model_inputs.items()}
                emb = self._biencoder.query_encoder(**model_inputs).cpu().numpy()
                embeddings.append(emb)

        return np.vstack(embeddings)

    def search(self, query_embeddings: np.ndarray, k: int, **faiss_index_options):
        scores_list, ids_list = self.index.search(query_embeddings, k, **faiss_index_options)
        return scores_list, ids_list


@register('bpr')
class BPR(Component, Serializable):
    def __init__(self, load_path, bpr_checkpoint, bpr_index, top_n=100, *args, **kwargs):
        super().__init__(save_path=None, load_path=load_path)
        self.bpr_checkpoint = bpr_checkpoint
        self.bpr_index = bpr_index
        self.top_n = top_n
        self.load()
        self.index = FaissBinaryIndex(self.base_index)
        self.retriever = Retriever(self.index, self.biencoder)
    
    def load(self):
        self.biencoder = BiEncoder.load_from_checkpoint(str(self.load_path / self.bpr_checkpoint))
        self.biencoder.eval()
        self.biencoder.freeze()
        self.base_index = faiss.read_index_binary(str(self.load_path / self.bpr_index))
    
    def save(self) -> None:
        pass

    def __call__(self, queries):
        queries = list(queries)
        query_embeddings = self.retriever.encode_queries(queries)
        scores, ids = self.retriever.search(query_embeddings, k=self.top_n)
        return ids.tolist()
