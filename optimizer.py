import numpy as np
import random
from sentence_transformers import SentenceTransformer
from config import CATEGORIES

class FreeTextIECOptimizer:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(model_name)
        self.current_base_vector = None
        self.categories = CATEGORIES
        
        # پیش‌محاسبه امبدینگ تمام کلمات کاتالوگ
        self.category_embeddings = {}
        for cat, words in self.categories.items():
            self.category_embeddings[cat] = {
                word: self.compute_embedding(word) for word in words
            }

    def compute_embedding(self, text):
        emb = self.encoder.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return emb

    def _find_best_word_for_category(self, cat, target_vec, temperature=0.5):
        """انتخاب کلمه متناسب با بردار هدف با احتمال رولت و دما"""
        word_dict = self.category_embeddings[cat]
        words = list(word_dict.keys())
        sims = np.array([np.dot(target_vec, word_dict[w]) for w in words])
        
        # تبدیل شباهت به احتمال (Softmax با دما)
        exp_sims = np.exp((sims - np.max(sims)) / max(temperature, 0.1))
        probs = exp_sims / np.sum(exp_sims)
        return np.random.choice(words, p=probs)

    def generate_full_prompt(self, target_vec, mutation_rate=0.2):
        """تولید پرامپت کامل ۴ بخشی منطبق بر مقاله"""
        parts = {}
        for cat in ["Atmosphere", "Genre", "Instrument", "Emotion"]:
            # در صورت جهش، یک کلمه کاملاً تصادفی انتخاب می‌شود
            if random.random() < mutation_rate:
                parts[cat] = random.choice(list(self.category_embeddings[cat].keys()))
            else:
                parts[cat] = self._find_best_word_for_category(cat, target_vec)
                
        return f"Generate {parts['Atmosphere']} {parts['Genre']} music with {parts['Instrument']}, evoking {parts['Emotion']}"

    def initialize_population(self, initial_user_prompt, pop_size=3):
        """تبدیل ورودی کاربر به ۳ پرامپت کامل ساختاریافته"""
        self.current_base_vector = self.compute_embedding(initial_user_prompt)
        
        prompts = []
        # ۳ ترکیب متنوع حول بردار ورودی کاربر با نرخ جهش‌های متفاوت
        prompts.append(self.generate_full_prompt(self.current_base_vector, mutation_rate=0.0))
        prompts.append(self.generate_full_prompt(self.current_base_vector, mutation_rate=0.2))
        prompts.append(self.generate_full_prompt(self.current_base_vector, mutation_rate=0.4))
        return prompts

    def evolve(self, current_prompts, ratings, pop_size=3):
        """الگوریتم تکاملی: ترکیب برداری پرامپت‌ها بر اساس امتیازدهی کاربر"""
        scores = np.array(ratings, dtype=float)
        # نرمال‌سازی وزن‌ها (امتیاز بالاتر = تاثیر بیشتر روی بردار جدید)
        weights = np.exp(scores - np.max(scores))
        weights = weights / np.sum(weights)
        
        prompt_vecs = np.array([self.compute_embedding(p) for p in current_prompts])
        weighted_vector = np.sum(weights[:, np.newaxis] * prompt_vecs, axis=0)
        weighted_vector = weighted_vector / np.linalg.norm(weighted_vector)
        
        # حرکت به سمت بردار مورد علاقه کاربر (یادگیری گام به گام)
        learning_rate = 0.7
        self.current_base_vector = (1.0 - learning_rate) * self.current_base_vector + learning_rate * weighted_vector
        self.current_base_vector = self.current_base_vector / np.linalg.norm(self.current_base_vector)
        
        # تولید جمعیت نسل جدید (متنوع با کلمات ارتقایافته)
        new_prompts = []
        new_prompts.append(self.generate_full_prompt(self.current_base_vector, mutation_rate=0.05))
        new_prompts.append(self.generate_full_prompt(self.current_base_vector, mutation_rate=0.20))
        new_prompts.append(self.generate_full_prompt(self.current_base_vector, mutation_rate=0.35))
        
        return new_prompts
