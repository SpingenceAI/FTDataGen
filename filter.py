import torch
from typing import List
from sentence_transformers import SentenceTransformer


class SemanticFilter:
    """Semantic filter for filtering similar sentences"""

    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embedding(self, sentences: List[str]) -> torch.tensor:
        """_summary_
        Args:
            sentences (List[str]): input sentences ex.["Hi", "Hello", "你好"]

        Returns:
            torch.tensor: embedding vector [n*384]
        """
        embeddings = self.model.encode(sentences)
        return embeddings

    def similarity(self, sentences: List[str]) -> torch.tensor:
        """_summary_

        Args:
            sentences (List[str]): input sentences ex.["Hi", "bye", "你好"]

        Returns:
            torch.tensor: similarity [n,n] tensor([[1.0,0.2,0.8],[0.2,1.0,0.1],[0.8,0.1,1.0]])
        """
        embeddings = self.embedding(sentences)
        similarities = self.model.similarity(embeddings, embeddings)
        return similarities

    def filter(self, sentences: List[str], thresh: float = 0.7) -> List[str]:
        """Filter out similar sentences by similarity threshold if similarity is greater than threshold, the sentence will be filtered out"""
        sim_tensor = self.similarity(sentences)
        remove_idx_list = []
        for i, sim_arr in enumerate(sim_tensor):
            for j, sim in enumerate(sim_arr):
                if (j < i) and (sim > thresh):
                    remove_idx_list.append(i)
        return [sentences[i] for i in range(len(sentences)) if i not in remove_idx_list]



if __name__ == "__main__":

    # test SemanticFilter
    test_data = [
        "這本書提供了許多關於成功的實用建議，讓讀者可以輕鬆掌握達成目標的方法。",
        "書中詳細介紹了成功的策略，幫助讀者輕鬆理解如何實現自己的目標。",
        "這本書充滿了實用的成功技巧，能幫助讀者清楚地了解達成目標的途徑。",
        "這本書缺乏實用的指導，讀者可能難以找到實現目標的具體方法。",
    ]
    sf = SemanticFilter("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    res = sf.filter(test_data)
    print(res)
    # return:
    # tensor([[1.0000, 0.3857, 0.8717],
    #     [0.3857, 1.0000, 0.2625],
    #     [0.8717, 0.2625, 1.0000]])