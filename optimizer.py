import numpy as np
import random
from sentence_transformers import SentenceTransformer
from config import CATEGORIES


class FreeTextIECOptimizer:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(model_name)
        self.current_base_vector = None
        self.categories = CATEGORIES
        self.generation = 0  # برای کاهش تدریجی learning_rate بین نسل‌ها

        # پیش‌محاسبه امبدینگ تمام کلمات کاتالوگ
        self.category_embeddings = {}
        for cat, words in self.categories.items():
            self.category_embeddings[cat] = {
                word: self.compute_embedding(word) for word in words
            }

    def compute_embedding(self, text):
        emb = self.encoder.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return emb

    def _find_best_word_for_category(self, cat, target_vec, temperature=0.12, deterministic=False):
        """
        انتخاب کلمه متناسب با بردار هدف.

        نکته مهم (باگ اصلی نسخه قبلی): شباهت کسینوسی بین امبدینگ یک
        کلمه‌ی تنها (مثلا "piano") و امبدینگ یک جمله‌ی کامل (پرامپت
        کاربر یا میانگین وزنی پرامپت‌های تولیدشده) همیشه در یک بازه‌ی
        خیلی باریک می‌افتد (معمولا حدود 0.15 تا 0.25)، نه در بازه‌ی
        کامل [-1, 1]. اگر این مقادیر باریک مستقیم وارد softmax با
        temperature=0.5 شوند، توزیع احتمال تقریبا یکنواخت می‌شود و
        انتخاب کلمه عملا رندوم می‌شود - یعنی امتیاز کاربر بی‌اثر می‌ماند.

        راه‌حل: قبل از softmax، شباهت‌ها را min-max normalize می‌کنیم تا
        فاصله‌ی نسبی بین کلمات بزرگ‌نمایی شود، بعد یک temperature کوچک‌تر
        روی آن اعمال می‌کنیم.

        deterministic=True: به‌جای نمونه‌گیری احتمالاتی، مستقیم نزدیک‌ترین
        کلمه (argmax شباهت) برگردانده می‌شود. این برای نمونه‌ی "لنگر"
        (anchor) هر نسل استفاده می‌شود تا حداقل یک نمونه واقعاً دقیق‌ترین
        تطبیق ممکن با ورودی کاربر باشد، نه یک انتخاب رندوم از بین گزینه‌های
        نسبتاً محتمل.
        """
        word_dict = self.category_embeddings[cat]
        words = list(word_dict.keys())
        sims = np.array([np.dot(target_vec, word_dict[w]) for w in words])

        if deterministic:
            return words[int(np.argmax(sims))]

        sim_range = sims.max() - sims.min()
        if sim_range < 1e-8:
            # همه کلمات تقریبا یک‌اندازه نزدیکند -> انتخاب واقعا یکنواخت
            return random.choice(words)

        norm_sims = (sims - sims.min()) / sim_range
        exp_sims = np.exp(norm_sims / max(temperature, 0.02))
        probs = exp_sims / np.sum(exp_sims)
        return np.random.choice(words, p=probs)

    def generate_full_prompt(self, target_vec, mutation_rate=0.2, deterministic=False):
        """تولید پرامپت کامل ۴ بخشی منطبق بر مقاله"""
        parts = {}
        for cat in ["Atmosphere", "Genre", "Instrument", "Emotion"]:
            # در صورت جهش، یک کلمه کاملاً تصادفی انتخاب می‌شود
            if not deterministic and random.random() < mutation_rate:
                parts[cat] = random.choice(list(self.category_embeddings[cat].keys()))
            else:
                parts[cat] = self._find_best_word_for_category(
                    cat, target_vec, deterministic=deterministic
                )

        return f"Generate {parts['Atmosphere']} {parts['Genre']} music with {parts['Instrument']}, evoking {parts['Emotion']}"

    def _resolve_category_word(self, cat, target_vec, llm_word=None):
        """
        بین پیشنهاد LLM و بهترین گزینه‌ی embedding (argmax) داور می‌کند.

        نکته‌ی مهم: اگر llm_word از قبل در کاتالوگ نباشد (یعنی LLM یک
        کلمه‌ی کاملاً جدید پیشنهاد داده، نه یکی از گزینه‌های موجود)،
        به‌جای رد کردنش، امبدینگش همین‌جا محاسبه و به کاتالوگ همین دسته
        اضافه می‌شود. این یعنی کاتالوگ به‌مرور با کلمات جدیدی که LLM
        کشف می‌کند بزرگ می‌شود و دیگر لازم نیست هر ژانر/ساز/احساس را از
        قبل و دستی پیش‌بینی و به config.py اضافه کنیم.

        چرا لازم است: مدل‌های کوچک همیشه دستور "دقیقا یکی از این
        گزینه‌ها را انتخاب کن" را درست دنبال نمی‌کنند و ممکن است حتی
        وقتی ورودی کاربر دقیقا با یکی از کلمات کاتالوگ یکی است، یک
        گزینه‌ی نامرتبط‌تر برگردانند. برای همین شباهت هر دو گزینه با
        target_vec محاسبه و بهترین انتخاب می‌شود - LLM فقط وقتی واقعا
        اثر می‌گذارد که پیشنهادش از نظر معنایی هم‌ارز یا بهتر باشد.
        """
        word_dict = self.category_embeddings[cat]
        words = list(word_dict.keys())
        sims = np.array([np.dot(target_vec, word_dict[w]) for w in words])
        best_idx = int(np.argmax(sims))
        embedding_word = words[best_idx]
        embedding_score = sims[best_idx]

        if llm_word:
            if llm_word not in word_dict:
                # کلمه‌ی جدید: امبدینگش را حساب و به کاتالوگ اضافه کن
                word_dict[llm_word] = self.compute_embedding(llm_word)
            llm_score = np.dot(target_vec, word_dict[llm_word])
            if llm_score >= embedding_score:
                return llm_word

        return embedding_word

    def initialize_population(self, initial_user_prompt, pop_size=3, llm_choices=None):
        """
        تبدیل ورودی کاربر به pop_size پرامپت کامل ساختاریافته.

        llm_choices (اختیاری): دیکشنری {category: word_or_None} که از
        LLMCategoryMapper می‌آید. برای نمونه‌ی لنگر، بین پیشنهاد LLM و
        بهترین گزینه‌ی embedding داوری می‌شود (نگاه کنید به
        _resolve_category_word) - یعنی هیچ‌وقت یک جواب ضعیف‌تر از LLM
        جایگزین یک تطبیق واضح embedding نمی‌شود.
        """
        self.current_base_vector = self.compute_embedding(initial_user_prompt)
        self.generation = 0

        anchor_parts = {}
        for cat in ["Atmosphere", "Genre", "Instrument", "Emotion"]:
            llm_word = (llm_choices or {}).get(cat)
            anchor_parts[cat] = self._resolve_category_word(
                cat, self.current_base_vector, llm_word=llm_word
            )

        anchor_prompt = (
            f"Generate {anchor_parts['Atmosphere']} {anchor_parts['Genre']} "
            f"music with {anchor_parts['Instrument']}, evoking {anchor_parts['Emotion']}"
        )

        prompts = [anchor_prompt]
        # بقیه‌ی جمعیت برای تنوع و اکتشاف، با نرخ جهش صعودی
        if pop_size > 1:
            mutation_rates = np.linspace(0.1, 0.4, pop_size - 1)
            prompts += [
                self.generate_full_prompt(self.current_base_vector, mutation_rate=float(m))
                for m in mutation_rates
            ]
        return prompts

    def evolve(self, current_prompts, ratings, pop_size=3):
        """الگوریتم تکاملی: ترکیب برداری پرامپت‌ها بر اساس امتیازدهی کاربر"""
        self.generation += 1

        scores = np.array(ratings, dtype=float)
        # نرمال‌سازی وزن‌ها (امتیاز بالاتر = تاثیر بیشتر روی بردار جدید)
        weights = np.exp(scores - np.max(scores))
        weights = weights / np.sum(weights)

        prompt_vecs = np.array([self.compute_embedding(p) for p in current_prompts])
        weighted_vector = np.sum(weights[:, np.newaxis] * prompt_vecs, axis=0)
        weighted_vector = weighted_vector / np.linalg.norm(weighted_vector)

        # learning_rate کاهشی: نسخه قبلی همیشه 0.7 بود که در ۳ نسل
        # عملا بردار اولیه‌ی کاربر را محو می‌کرد. حالا هرچه جلوتر می‌رویم
        # محتاط‌تر حرکت می‌کنیم تا "تکامل تدریجی" واقعا حس شود.
        base_lr = 0.45
        learning_rate = base_lr / (1 + 0.5 * self.generation)

        self.current_base_vector = (
            (1.0 - learning_rate) * self.current_base_vector
            + learning_rate * weighted_vector
        )
        self.current_base_vector = self.current_base_vector / np.linalg.norm(self.current_base_vector)

        # تولید جمعیت نسل جدید: نمونه‌ی اول همیشه لنگر دقیق (argmax) است،
        # بقیه برای اکتشاف با نرخ جهش صعودی
        prompts = [
            self.generate_full_prompt(self.current_base_vector, deterministic=True)
        ]
        if pop_size > 1:
            mutation_rates = np.linspace(0.1, 0.35, pop_size - 1)
            prompts += [
                self.generate_full_prompt(self.current_base_vector, mutation_rate=float(m))
                for m in mutation_rates
            ]
        return prompts
