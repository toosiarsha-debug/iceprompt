import numpy as np
import random
from sentence_transformers import SentenceTransformer
from config import CATEGORIES

class FreeTextIECOptimizer:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(model_name)
        self.current_base_vector = None
        self.positive_prompts_history = []
        self.negative_prompts_history = []
        self.categories = CATEGORIES
        
        # پیش‌محاسبه امبدینگ کلمات برای انتخاب بهینه
        self.category_embeddings = {}
        for cat, words in self.categories.items():
            self.category_embeddings[cat] = {
                word: self.compute_embedding(word) for word in words
            }

    def compute_embedding(self, text):
        emb = self.encoder.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return emb

    def initialize_base(self, initial_user_prompt):
        self.current_base_vector = self.compute_embedding(initial_user_prompt)
        return self.current_base_vector

    def _sample_prompt_near_target(self, target_vector, explore_rate=0.3):
        selected_words = {}
        for cat, word_dict in self.category_embeddings.items():
            words = list(word_dict.keys())
            if random.random() < explore_rate or target_vector is None:
                selected_words[cat] = random.choice(words)
            else:
                # انتخاب کلماتی با بیشترین شباهت کسینوسی به جهت هدف
                sims = [np.dot(target_vector, word_dict[w]) for w in words]
                best_indices = np.argsort(sims)[-3:]
                chosen_idx = random.choice(best_indices)
                selected_words[cat] = words[chosen_idx]

        return f"Generate {selected_words['Atmosphere']} {selected_words['Genre']} music with {selected_words['Instrument']}, evoking {selected_words['Emotion']}"

    def initialize_population(self, initial_user_prompt, pop_size=3):
        self.initialize_base(initial_user_prompt)
        prompts = [initial_user_prompt]
        for _ in range(pop_size - 1):
            prompts.append(self._sample_prompt_near_target(self.current_base_vector, explore_rate=0.4))
        return prompts

    def update_objective_and_get_target(self, population_prompts, user_scores, temperature=1.5, baseline=3.0):
        scores = np.array(user_scores, dtype=float)

        for prompt, score in zip(population_prompts, scores):
            if score >= 4:
                self.positive_prompts_history.append(prompt)
            elif score <= 2:
                self.negative_prompts_history.append(prompt)

        shifted_scores = (scores - baseline) / temperature
        exp_weights = np.exp(shifted_scores - np.max(shifted_scores))
        weights = exp_weights / np.sum(exp_weights)

        prompt_vectors = np.array([self.compute_embedding(p) for p in population_prompts])
        target_direction = np.sum(weights[:, np.newaxis] * prompt_vectors, axis=0)
        target_norm = np.linalg.norm(target_direction)
        if target_norm > 0:
            target_direction /= target_norm

        learning_rate = 0.6
        if self.current_base_vector is not None:
            new_base = (1.0 - learning_rate) * self.current_base_vector + learning_rate * target_direction
        else:
            new_base = target_direction

        self.current_base_vector = new_base / np.linalg.norm(new_base)
        return self.current_base_vector

    def evolve(self, current_prompts, ratings, pop_size=3):
        # به‌روزرسانی بردار هدف بر اساس امتیازهای کاربر (از ۱ تا ۵)
        target_vec = self.update_objective_and_get_target(current_prompts, ratings, baseline=3.0)
        
        # حفظ بهترین پرامپت (Elitism)
        best_idx = int(np.argmax(ratings))
        best_prompt = current_prompts[best_idx]
        
        new_prompts = [best_prompt]
        for _ in range(pop_size - 1):
            new_prompts.append(self._sample_prompt_near_target(target_vec, explore_rate=0.25))
            
        return new_prompts
