import numpy as np
from sentence_transformers import SentenceTransformer

class FreeTextIECOptimizer:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(model_name)
        self.current_base_vector = None
        self.positive_prompts_history = []
        self.negative_prompts_history = []

    def compute_embedding(self, text):
        return self.encoder.encode(text, convert_to_numpy=True, normalize_embeddings=True)

    def initialize_base(self, initial_user_prompt):
        self.current_base_vector = self.compute_embedding(initial_user_prompt)
        return self.current_base_vector

    def update_objective_and_get_target(self, population_prompts, user_scores, temperature=1.5, baseline=5.0):
        scores = np.array(user_scores, dtype=float)

        for prompt, score in zip(population_prompts, scores):
            if score >= 8:
                self.positive_prompts_history.append(prompt)
            elif score <= 2:
                self.negative_prompts_history.append(prompt)

        shifted_scores = (scores - baseline) / temperature
        exp_weights = np.exp(shifted_scores - np.max(shifted_scores))
        weights = exp_weights / np.sum(exp_weights)

        prompt_vectors = np.array([self.compute_embedding(p) for p in population_prompts])

        target_direction = np.sum(weights[:, np.newaxis] * prompt_vectors, axis=0)
        target_direction /= np.linalg.norm(target_direction)

        learning_rate = 0.6
        new_base = (1.0 - learning_rate) * self.current_base_vector + learning_rate * target_direction
        self.current_base_vector = new_base / np.linalg.norm(new_base)

        return self.current_base_vector
