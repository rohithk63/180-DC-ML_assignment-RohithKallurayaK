import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CASES_SAMPLE = [
    {"case_text": "A man sues a parrot for defamation."},
    {"case_text": "A cat inherits a fortune from its owner."},
    {"case_text": "A robot refuses to pay taxes."},
    {"case_text": "A woman accuses her cat of stealing her Wi-Fi signal."},
{"case_text": "A man sues a vending machine for emotional distress."},
{"case_text": "A baker is taken to court for overbaking bread that exploded."},
{"case_text": "A teenager sues gravity after tripping on a skateboard."},
{"case_text": "A man blames his goldfish for insider trading."},
{"case_text": "A teacher sues a chalkboard for public humiliation."},
{"case_text": "A city sues clouds for causing rain during the marathon."},
{"case_text": "A woman accuses her toaster of breaching contract."},
{"case_text": "A man sues his alarm clock for waking him up too early."},
{"case_text": "A company is taken to court for selling invisible hats."},
{"case_text": "A child sues a tree for hitting them with falling leaves."},
{"case_text": "A man sues his mirror for identity theft."},
{"case_text": "A woman blames her socks for causing a slip-and-fall."},
{"case_text": "A man accuses his shoes of trespassing in his living room."},
{"case_text": "A cat is sued for harassment after knocking over furniture."},
{"case_text": "A woman sues a cloud for blocking sunlight during photosynthesis."},
{"case_text": "A man blames his pillow for insomnia."},
{"case_text": "A dog is sued for barking without a permit."},
{"case_text": "A man takes his houseplants to court for negligence."},
{"case_text": "A child sues their blanket for emotional abandonment."},
{"case_text": "A man blames his fridge for spoiling leftovers."},
{"case_text": "A woman accuses her carpet of defamation after a stain incident."},
{"case_text": "A man sues the moon for influencing tides."},
{"case_text": "A person blames their shoes for losing a race."},
{"case_text": "A bird is taken to court for disturbing the peace in the morning."}
]

def random_case():
    return random.choice(CASES_SAMPLE)["case_text"]

def semantic_search(query, corpus_texts, top_k=1):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(corpus_texts + [query])
    sims = cosine_similarity(vectors[-1], vectors[:-1]).flatten()
    top_indices = sims.argsort()[-top_k:][::-1]
    return top_indices

def hybrid_retrieval(query, cases, top_k=1):
    corpus_texts = [c['summary'] for c in cases]
    top_idx = semantic_search(query, corpus_texts, top_k)
    return [cases[i] for i in top_idx]
