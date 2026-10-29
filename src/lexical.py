from collections import Counter
import math
import re
import unicodedata


def tokenize(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    return re.findall(r"\w+(?:\.\d+)*", text)


class BM25Index:
    def __init__(self, chunks, k1=1.5, b=0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.terms = [Counter(tokenize(chunk["text"])) for chunk in chunks]
        self.lengths = [sum(terms.values()) for terms in self.terms]
        self.average_length = sum(self.lengths) / len(chunks) if chunks else 0
        frequencies = Counter(term for terms in self.terms for term in terms)
        self.idf = {term: math.log(1 + (len(chunks) - count + 0.5) / (count + 0.5))
                    for term, count in frequencies.items()}

    def retrieve(self, query, top_k=20):
        if top_k <= 0 or not self.average_length:
            return []
        query_terms = set(tokenize(query))
        scored = []
        for index, terms in enumerate(self.terms):
            norm = self.k1 * (1 - self.b + self.b * self.lengths[index] / self.average_length)
            score = sum(self.idf[term] * terms[term] * (self.k1 + 1) / (terms[term] + norm)
                        for term in query_terms if term in terms)

            if score > 0:
                scored.append((index, score))
        scored.sort(key=lambda item: (-item[1], item[0]))
        return [{**self.chunks[index], "bm25_score": score}
                for index, score in scored[:top_k]]
