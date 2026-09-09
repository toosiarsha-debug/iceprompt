import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from config import CATEGORIES, MODEL_CONFIG


class IECOptimizer:
    def __init__(self):
        self.encoder = SentenceTransformer(MODEL_CONFIG["embedding_model"])
        self.categories = CATEGORIES
        self.category_embeddings = self._precompute_embeddings()
        self.num_candidates = MODEL_CONFIG["num_candidates"]
        self.baseline_score = MODEL_CONFIG["baseline_score"]
        self.current_population = []
        self.current_vectors = []

    def _precompute_embeddings(self):
        embeddings = {}
        for cat, words in self.categories.items():
            embeddings[cat] = {
                word: self.encoder.encode(word, normalize_embeddings=True)
                for word in words
            }
        return embeddings

    def build_prompt_text(self, word_dict):
        return (
            f"A {word_dict['Atmosphere']} {word_dict['Genre']} music "
            f"featuring {word_dict['Instrument']}, evoking a {word_dict['Emotion']} emotion"
        )

    def initialize_population(self, seed_selection):
        self.current_population = []
        self.current_vectors = []

        base_prompt = seed_selection.copy()
        self.current_population.append(base_prompt)
        base_vec = self.encoder.encode(
            self.build_prompt_text(base_prompt), 
            normalize_embeddings=True
        )
        self.current_vectors.append(base_vec)

        for _ in range(self.num_candidates - 1):
            mutated = base_prompt.copy()
            cats_to_mutate = np.random.choice(
                list(self.categories.keys()), 
                size=2, 
                replace=False
            )
            for cat in cats_to_mutate:
                mutated[cat] = np.random.choice(self.categories[cat])
            self.current_population.append(mutated)
            vec = self.encoder.encode(
                self.build_prompt_text(mutated), 
                normalize_embeddings=True
            )
            self.current_vectors.append(vec)

        return self.current_population

    def find_nearest_word(self, target_vec, category, exclude_word=None):
        best_word = None
        min_dist = float("inf")
        for word, vec in self.category_embeddings[category].items():
            if word == exclude_word:
                continue
            dist = cosine(target_vec, vec)
            if dist < min_dist:
                min_dist = dist
                best_word = word
        return best_word if best_word else exclude_word

    def evolve_step(self, scores):
        scores = np.array(scores, dtype=float)
        weights = scores - self.baseline_score

        if np.all(weights == 0) or np.sum(np.abs(weights)) == 0:
            weights = np.ones_like(weights)

        weighted_sum = np.zeros_like(self.current_vectors[0])
        for w, vec in zip(weights, self.current_vectors):
            weighted_sum += w * vec

        norm = np.linalg.norm(weighted_sum)
        if norm > 0:
            target_vector = weighted_sum / norm
        else:
            target_vector = self.current_vectors[np.argmax(scores)]

        best_idx = int(np.argmax(scores))
        best_candidate = self.current_population[best_idx].copy()

        new_population = [best_candidate]
        new_vectors = [self.current_vectors[best_idx]]

        for i in range(self.num_candidates - 1):
            next_cand = {}
            for cat in self.categories.keys():
                current_word = best_candidate[cat]
                curr_vec = self.category_embeddings[cat][current_word]
                step_direction = target_vector[:len(curr_vec)]
                
                mutation_noise = np.random.normal(0, 0.05, size=curr_vec.shape)
                step_direction += mutation_noise
                
                step_norm = np.linalg.norm(step_direction)
                if step_norm > 0:
                    step_direction = step_direction / step_norm
                
                if np.random.rand() > 0.3:
                    next_cand[cat] = self.find_nearest_word(step_direction, cat)
                else:
                    next_cand[cat] = np.random.choice(self.categories[cat])

            new_population.append(next_cand)
            vec = self.encoder.encode(
                self.build_prompt_text(next_cand), 
                normalize_embeddings=True
            )
            new_vectors.append(vec)

        self.current_population = new_population
        self.current_vectors = new_vectors
        return self.current_population
