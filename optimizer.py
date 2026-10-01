import re
import random
import numpy as np
from sentence_transformers import SentenceTransformer
from config import CATEGORIES

CATEGORY_ORDER = ["Atmosphere", "Genre", "Instrument", "Emotion"]

# برای استخراج کلمه‌ی هر دسته از متن پرامپت قالب‌بندی‌شده
PROMPT_RE = re.compile(
    r"^Generate (?P<Atmosphere>.+?) (?P<Genre>.+?) music with "
    r"(?P<Instrument>.+?), evoking (?P<Emotion>.+?)$"
)


class FreeTextIECOptimizer:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(model_name)
        self.categories = CATEGORIES
        self.generation = 0  # برای کاهش تدریجی learning_rate بین نسل‌ها

        # بردار سلیقه‌ی جداگانه برای هر دسته (نه یک بردار مشترک برای کل
        # جمله). این مهم‌ترین تفاوت نسبت به نسخه‌ی قبلی است: قبلا وقتی
        # یک پرامپت کامل نمره‌ی بالا می‌گرفت، کل جمله باهم میانگین‌گیری
        # می‌شد و معلوم نبود دقیقا کدام دسته (مثلا ژانر) باعث آن امتیاز
        # بالا بوده - این باعث می‌شد سیگنال دسته‌ها با هم قاطی شود و
        # حتی نمونه‌ی "لنگر" هم گاهی به سمت چیزی نامرتبط برود. حالا هر
        # دسته فقط از کلمه‌ی خودش در پرامپت‌های امتیازگرفته یاد می‌گیرد.
        self.category_vectors = {cat: None for cat in CATEGORY_ORDER}

        # پیش‌محاسبه امبدینگ تمام کلمات کاتالوگ
        self.category_embeddings = {}
        for cat, words in self.categories.items():
            self.category_embeddings[cat] = {
                word: self.compute_embedding(word) for word in words
            }

    def compute_embedding(self, text):
        return self.encoder.encode(text, convert_to_numpy=True, normalize_embeddings=True)

    def _parse_prompt(self, prompt):
        """
        از متن پرامپت قالب‌بندی‌شده، کلمه‌ی انتخابی هر دسته را استخراج
        می‌کند تا بتوان بردار هر دسته را جداگانه و دقیق آپدیت کرد.
        """
        m = PROMPT_RE.match(prompt)
        if m:
            return {cat: m.group(cat) for cat in CATEGORY_ORDER}

        # fallback محتاط‌تر: اگر قالب دقیق مطابقت نداشت (مثلا به‌خاطر
        # یک کلمه‌ی چندبخشی جدید)، سعی می‌کنیم طولانی‌ترین کلمه‌ی
        # شناخته‌شده‌ی هر دسته را داخل متن پیدا کنیم
        parts = {}
        low = prompt.lower()
        for cat in CATEGORY_ORDER:
            found = None
            for word in self.category_embeddings[cat]:
                if word.lower() in low:
                    if found is None or len(word) > len(found):
                        found = word
            parts[cat] = found
        return parts

    def _find_best_word_for_category(self, cat, target_vec, temperature=0.3, deterministic=False):
        """
        انتخاب کلمه متناسب با بردار هدف.

        نکته مهم: شباهت کسینوسی بین امبدینگ یک کلمه‌ی تنها (مثلا
        "piano") و امبدینگ یک جمله‌ی کامل همیشه در بازه‌ای خیلی باریک
        می‌افتد، نه در بازه‌ی کامل [-1, 1]. اگر این مقادیر باریک مستقیم
        وارد softmax شوند، توزیع احتمال تقریبا یکنواخت و انتخاب کلمه
        عملا رندوم می‌شود. راه‌حل: قبل از softmax، شباهت‌ها را
        min-max normalize می‌کنیم تا فاصله‌ی نسبی بین کلمات بزرگ‌نمایی
        شود.

        deterministic=True: به‌جای نمونه‌گیری احتمالاتی، مستقیم
        نزدیک‌ترین کلمه (argmax شباهت) برگردانده می‌شود.
        """
        word_dict = self.category_embeddings[cat]
        words = list(word_dict.keys())
        sims = np.array([np.dot(target_vec, word_dict[w]) for w in words])

        if deterministic:
            return words[int(np.argmax(sims))]

        sim_range = sims.max() - sims.min()
        if sim_range < 1e-8:
            return random.choice(words)

        norm_sims = (sims - sims.min()) / sim_range
        exp_sims = np.exp(norm_sims / max(temperature, 0.02))
        probs = exp_sims / np.sum(exp_sims)
        return np.random.choice(words, p=probs)

    def generate_full_prompt(self, target_vecs, mutation_rate=0.2, deterministic=False):
        """
        target_vecs: دیکشنری {category: vector} - بردار هدف جداگانه
        برای هر دسته (خروجی self.category_vectors).
        """
        parts = {}
        for cat in CATEGORY_ORDER:
            if not deterministic and random.random() < mutation_rate:
                parts[cat] = random.choice(list(self.category_embeddings[cat].keys()))
            else:
                parts[cat] = self._find_best_word_for_category(
                    cat, target_vecs[cat], deterministic=deterministic
                )
        return (
            f"Generate {parts['Atmosphere']} {parts['Genre']} music with "
            f"{parts['Instrument']}, evoking {parts['Emotion']}"
        )

    def _generate_diverse_prompts(self, target_vecs, count, exclude=None, max_retries=8):
        """
        count پرامپت متفاوت از هم (و از exclude) تولید می‌کند. اگر
        پرامپت تولیدشده تکراری بود، با نرخ جهش بالاتر (تا سقف ۱.۰) دوباره
        تولید می‌شود تا واقعا متفاوت باشد - این برای جلوگیری از این است
        که دو نمونه‌ی مختلف دقیقا یک genotype را نمایندگی کنند و امتیاز
        کاربر را مخدوش کنند.
        """
        seen = set(exclude or [])
        prompts = []
        if count <= 0:
            return prompts

        base_rates = np.linspace(0.1, 0.4, count)
        for base_rate in base_rates:
            rate = float(base_rate)
            candidate = self.generate_full_prompt(target_vecs, mutation_rate=rate)
            attempt = 0
            while candidate in seen and attempt < max_retries:
                attempt += 1
                rate = min(1.0, rate + 0.15)
                candidate = self.generate_full_prompt(target_vecs, mutation_rate=rate)
            prompts.append(candidate)
            seen.add(candidate)

        return prompts

    def _resolve_category_word(self, cat, target_vec, llm_word=None):
        """
        بین پیشنهاد LLM و بهترین گزینه‌ی embedding (argmax) داور می‌کند.
        اگر llm_word از قبل در کاتالوگ نباشد، امبدینگش همین‌جا محاسبه و
        به کاتالوگ همین دسته اضافه می‌شود (کاتالوگ به‌مرور بزرگ‌تر
        می‌شود). LLM فقط وقتی واقعا اثر می‌گذارد که پیشنهادش از نظر
        معنایی هم‌ارز یا بهتر از نزدیک‌ترین تطبیق embedding باشد.
        """
        word_dict = self.category_embeddings[cat]
        words = list(word_dict.keys())
        sims = np.array([np.dot(target_vec, word_dict[w]) for w in words])
        best_idx = int(np.argmax(sims))
        embedding_word = words[best_idx]
        embedding_score = sims[best_idx]

        if llm_word:
            if llm_word not in word_dict:
                word_dict[llm_word] = self.compute_embedding(llm_word)
            llm_score = np.dot(target_vec, word_dict[llm_word])
            if llm_score >= embedding_score:
                return llm_word

        return embedding_word

    def initialize_population(self, initial_user_prompt, pop_size=3, llm_choices=None):
        """
        تبدیل ورودی کاربر به pop_size پرامپت کامل ساختاریافته.

        چون هنوز هیچ تاریخچه‌ای از کلمات امتیازگرفته نداریم، همه‌ی ۴
        بردار دسته را با امبدینگ کل پرامپت اولیه کاربر seed می‌کنیم.
        از نسل بعد (evolve)، هر بردار دسته جدا و فقط از کلمه‌ی خودش
        در پرامپت‌های امتیازگرفته‌شده به‌روزرسانی می‌شود.
        """
        self.generation = 0
        base_vec = self.compute_embedding(initial_user_prompt)
        for cat in CATEGORY_ORDER:
            self.category_vectors[cat] = base_vec.copy()

        anchor_parts = {}
        for cat in CATEGORY_ORDER:
            llm_word = (llm_choices or {}).get(cat)
            anchor_parts[cat] = self._resolve_category_word(
                cat, self.category_vectors[cat], llm_word=llm_word
            )
        anchor_prompt = (
            f"Generate {anchor_parts['Atmosphere']} {anchor_parts['Genre']} "
            f"music with {anchor_parts['Instrument']}, evoking {anchor_parts['Emotion']}"
        )

        rest = self._generate_diverse_prompts(
            self.category_vectors, pop_size - 1, exclude=[anchor_prompt]
        )
        return [anchor_prompt] + rest

    def evolve(self, current_prompts, ratings, pop_size=3):
        """
        الگوریتم تکاملی: هر دسته جداگانه، فقط بر اساس کلمه‌ی خودش در
        پرامپت‌های امتیازگرفته، به‌روزرسانی می‌شود.

        سه تکنیک برای محسوس‌تر شدن اثر امتیاز کاربر:
        1. فشار انتخاب تندتر (selection_temperature پایین): تفاوت
           نمرات را بزرگ‌نمایی می‌کند.
        2. learning_rate با پایه‌ی بالا و کاهش ملایم.
        3. الیتیسم: بهترین‌نمره‌گرفته‌ی همین نسل عینا به نسل بعد منتقل
           می‌شود.
        """
        self.generation += 1
        scores = np.array(ratings, dtype=float)

        selection_temperature = 0.35
        weights = np.exp((scores - np.max(scores)) / selection_temperature)
        weights = weights / np.sum(weights)

        parsed = [self._parse_prompt(p) for p in current_prompts]

        base_lr = 0.6
        learning_rate = base_lr / (1 + 0.3 * self.generation)

        for cat in CATEGORY_ORDER:
            word_vecs = []
            for parts in parsed:
                word = parts.get(cat)
                if word and word in self.category_embeddings[cat]:
                    word_vecs.append(self.category_embeddings[cat][word])
                else:
                    # fallback نادر: کلمه قابل‌استخراج نبود -> بی‌اثر بمان
                    word_vecs.append(self.category_vectors[cat])
            word_vecs = np.array(word_vecs)

            weighted_vector = np.sum(weights[:, np.newaxis] * word_vecs, axis=0)
            weighted_vector = weighted_vector / np.linalg.norm(weighted_vector)

            updated = (
                (1.0 - learning_rate) * self.category_vectors[cat]
                + learning_rate * weighted_vector
            )
            self.category_vectors[cat] = updated / np.linalg.norm(updated)

        # الیتیسم: بهترین‌نمره‌گرفته عینا حفظ می‌شود
        elite_idx = int(np.argmax(scores))
        elite_prompt = current_prompts[elite_idx]

        # نمونه‌ی لنگر: بهترین تطبیق دقیق (argmax) با بردارهای به‌روزشده
        anchor_prompt = self.generate_full_prompt(self.category_vectors, deterministic=True)

        seen = {elite_prompt}
        prompts = [elite_prompt]
        if anchor_prompt not in seen:
            prompts.append(anchor_prompt)
            seen.add(anchor_prompt)

        remaining = pop_size - len(prompts)
        rest = self._generate_diverse_prompts(self.category_vectors, remaining, exclude=seen)
        return prompts + rest
