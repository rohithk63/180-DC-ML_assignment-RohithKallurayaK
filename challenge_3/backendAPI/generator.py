import os
import json
import random
from collections import defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class QueryAwareCaseGenerator:

    def __init__(self, data_file: str, n: int = 2):
        self.n = n
        with open(data_file, "r", encoding="utf-8") as f:
            self.cases = json.load(f)

        # Use summaries as the corpus
        self.case_texts = [c.get("summary", "") for c in self.cases]
        self.vectorizer = TfidfVectorizer()
        self.embeddings = self.vectorizer.fit_transform(self.case_texts)

    # ---- retrieval ----
    def _retrieve_relevant_texts(self, query: str, top_k: int = 3):
        if not query.strip():
            return random.sample(self.case_texts, k=min(top_k, len(self.case_texts)))

        qv = self.vectorizer.transform([query])
        sims = cosine_similarity(qv, self.embeddings).flatten()
        idx = sims.argsort()[-top_k:][::-1]
        return [self.case_texts[i] for i in idx]

    # ---- small Markov builder ----
    def _build_markov(self, texts):
        chain = defaultdict(list)
        for text in texts:
            words = text.split()
            if len(words) < self.n + 1:
                continue
            for i in range(len(words) - self.n):
                key = tuple(words[i:i + self.n])
                chain[key].append(words[i + self.n])
        return chain

    # ---- generation ----
    def generate(self, query: str, max_words: int = 24) -> str:
        relevant_texts = self._retrieve_relevant_texts(query, top_k=3)
        chain = self._build_markov(relevant_texts)
        if not chain:
            return "Based on prior rulings, the matter hinges on evidence and applicable precedent."

        start = random.choice(list(chain.keys()))
        out = list(start)

        for _ in range(max_words - self.n):
            key = tuple(out[-self.n:])
            next_options = chain.get(key)
            if not next_options:
                break
            out.append(random.choice(next_options))

        sentence = " ".join(out).strip()

        if not sentence.endswith((".", "!", "?")):
            sentence += "."
        return sentence

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "legal_cases.json")
generator = QueryAwareCaseGenerator(DATA_PATH, n=2)
