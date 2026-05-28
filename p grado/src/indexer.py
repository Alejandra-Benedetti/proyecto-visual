import numpy as np
import pickle
from pathlib import Path
from config import CACHE_DIR, TOP_K_SIMILAR


class SimilarityIndexer:
    def __init__(self):
        self.index = None
        self.image_paths: list = []
        self._index_file = CACHE_DIR / "faiss.index"
        self._paths_file = CACHE_DIR / "faiss_paths.pkl"

    def build(self, embeddings: np.ndarray, image_paths: list):
        import faiss

        emb = embeddings.astype(np.float32)
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(emb)
        self.image_paths = list(image_paths)

        faiss.write_index(self.index, str(self._index_file))
        with open(self._paths_file, "wb") as f:
            pickle.dump(self.image_paths, f)

    def load(self) -> bool:
        import faiss

        if self._index_file.exists() and self._paths_file.exists():
            self.index = faiss.read_index(str(self._index_file))
            with open(self._paths_file, "rb") as f:
                self.image_paths = pickle.load(f)
            return True
        return False

    def search(self, query: np.ndarray, k: int = TOP_K_SIMILAR) -> list:
        scores, indices = self.index.search(query.astype(np.float32).reshape(1, -1), k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append({
                    "path": self.image_paths[idx],
                    "score": float(score),
                    "index": int(idx),
                })
        return results

    def cluster(self, embeddings: np.ndarray, n_clusters: int = 5) -> np.ndarray:
        import faiss

        emb = embeddings.astype(np.float32)
        km = faiss.Kmeans(emb.shape[1], n_clusters, niter=30, verbose=False, seed=42)
        km.train(emb)
        _, labels = km.index.search(emb, 1)
        return labels.flatten()
