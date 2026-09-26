import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LLMCategoryMapper:
    """
    پرامپت آزاد کاربر رو می‌گیره و برای هر دسته (Atmosphere/Genre/
    Instrument/Emotion) با کمک یک LLM محلی، بهترین گزینه رو از کاتالوگ
    ثابت (CATEGORIES) انتخاب می‌کنه. این همون لایه‌ای‌ست که پرامپت آزاد
    رو به فرمت مورد نیاز روش IEC تبدیل می‌کنه.

    مدل پیش‌فرض: HuggingFaceTB/SmolLM3-3B — کاملا باز (Apache 2.0)،
    بدون gate و بدون نیاز به توکن/لایسنس، و برای این نوع دستورالعمل‌های
    محدودکننده (انتخاب از یک لیست کوتاه) به‌طور محسوسی بهتر از مدل‌های
    قبلی (flan-t5-base) عمل می‌کند.

    اگر مدل لود نشود یا خروجی قابل‌تشخیص نباشد، برای همان دسته مقدار
    None برمی‌گردد تا فراخوان (optimizer) بتواند به روش embedding فالبک
    کند - یعنی هیچ‌وقت کل سیستم به این LLM وابسته‌ی صرف نمی‌ماند. علاوه
    بر این، optimizer.py حتی وقتی LLM جواب معتبری بدهد، آن را با بهترین
    گزینه‌ی embedding مقایسه می‌کند و فقط اگر واقعا بهتر یا هم‌ارز باشد
    قبولش می‌کند (نگاه کنید به _resolve_category_word).
    """

    def __init__(self, categories, model_name="HuggingFaceTB/SmolLM3-3B"):
        self.categories = categories
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
            self.is_loaded = True
        except Exception as e:
            print(f"[LLMCategoryMapper] Could not load {model_name}, will fallback to embeddings: {e}")
            self.is_loaded = False

    def _ask_llm_for_category(self, user_prompt, cat, words):
        options_str = ", ".join(words)
        user_msg = (
            f"Music description: \"{user_prompt}\"\n"
            f"(Note: the description may reference a real artist, band, or song - "
            f"if so, use your knowledge of their sound/style to judge this.)\n"
            f"Which one of these {cat.lower()} options best fits this description? "
            f"Options: {options_str}. "
            f"If one of these options fits well, reply with exactly that word or phrase. "
            f"If none of them really fit, reply with a single better word or short phrase "
            f"of your own instead. Reply with only the word or phrase, nothing else."
        )
        # "/no_think" حالت استدلال طولانی SmolLM3 رو خاموش می‌کنه تا
        # جواب کوتاه و مستقیم بدیم (برای این کار ساده لازم نیست فکر کنه)
        messages = [
            {"role": "system", "content": "/no_think"},
            {"role": "user", "content": user_msg},
        ]
        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs, max_new_tokens=15, do_sample=False
                )
            output_ids = generated_ids[0][len(inputs.input_ids[0]):]
            raw = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip().lower()
        except Exception:
            return None

        if not raw:
            return None

        # اول تلاش می‌کنیم ببینیم آیا خروجی دقیقاً یکی از گزینه‌های
        # موجود کاتالوگ است (برای جلوگیری از تکثیر مترادف‌های نزدیک)
        for w in sorted(words, key=len, reverse=True):
            if re.search(rf"\b{re.escape(w.lower())}\b", raw):
                return w

        # اگر هیچ‌کدام از گزینه‌های کاتالوگ مطابقت نداشت، خروجی خام مدل
        # را به‌عنوان یک کلمه/عبارت جدید برمی‌گردانیم (نه رد کردنش).
        # optimizer.py مسئول اضافه کردن این کلمه‌ی جدید به کاتالوگ است.
        cleaned = re.sub(r'["\'.!?]', '', raw).strip()
        word_count = len(cleaned.split())
        if 0 < word_count <= 3:
            return cleaned
        return None

    def expand_prompt(self, user_prompt):
        """
        اگر پرامپت به یک هنرمند/بند/آهنگ واقعی اشاره کند (مثلا
        "music like queen")، آن را به یک توصیف صریح صوتی (ژانر، حس،
        سازبندی، دوره‌ی زمانی) تبدیل می‌کند - مثلا برای "queen" باید
        چیزی شبیه "glam/arena rock with layered vocal harmonies, electric
        guitar and piano, theatrical and energetic" برگرداند.

        چرا لازم است: بدون این مرحله، هم تطبیق embedding و هم انتخاب
        دسته‌ها مستقیم روی متن خام و مبهم انجام می‌شد. کلماتی مثل
        "queen" به تنهایی معنای دیگری هم دارند (ملکه) و ممکن است مدل
        (یا embedding عمومی) را به سمت تداعی‌های غلط ببرند (مثلا به‌جای
        "rock"، به سمت "reggae" به‌خاطر عبارت‌هایی مثل "dancehall
        queen"). با اجبار مدل به بیان صریح دانش واقعی‌اش درباره‌ی
        هنرمند، این ریسک به‌شدت کم می‌شود.

        عمدا حالت تفکر (thinking) مدل را خاموش نکرده‌ایم (بدون
        "/no_think")، چون این یک فراخوان است (نه ۴ بار تکراری مثل
        انتخاب دسته‌ها) و دقت اینجا از سرعت مهم‌تر است.

        اگر پرامپت اصلا به هنرمند/بند/آهنگی اشاره نکند، یا مدل لود نشده
        باشد، همان متن اصلی کاربر بدون تغییر برگردانده می‌شود.
        """
        if not self.is_loaded:
            return user_prompt

        msg = (
            f"Music request: \"{user_prompt}\"\n"
            f"If this request references a real, existing music artist, band, or song, "
            f"briefly describe (in one or two sentences, in plain descriptive terms) their "
            f"typical genre, mood, era, and instrumentation, using your own knowledge. "
            f"If it does not reference any real artist, band, or song, just repeat the "
            f"request exactly as it is. "
            f"Reply with only the final description, nothing else - no preamble."
        )
        messages = [{"role": "user", "content": msg}]
        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs, max_new_tokens=120, do_sample=False
                )
            output_ids = generated_ids[0][len(inputs.input_ids[0]):]
            raw = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        except Exception:
            return user_prompt

        return raw if raw else user_prompt

    def map_to_categories(self, user_prompt):
        """
        خروجی: تاپل (دیکشنری {category: word_or_None}, expanded_prompt)

        اگر مدل لود نشده باشد، دیکشنری با همه‌ی مقادیر None و همان
        user_prompt اصلی برمی‌گردد تا فراخوان به‌طور کامل به embedding
        فالبک کند.
        """
        expanded_prompt = self.expand_prompt(user_prompt)

        if not self.is_loaded:
            return {cat: None for cat in self.categories}, expanded_prompt

        result = {}
        for cat, words in self.categories.items():
            result[cat] = self._ask_llm_for_category(expanded_prompt, cat, words)
        return result, expanded_prompt

    def suggest_similar_real_music(self, final_prompt, original_user_prompt=""):
        """
        بر اساس پرامپت نهاییِ بهینه‌شده (و پرامپت اولیه‌ی کاربر، اگر
        اسم هنرمند/بندی در آن بوده)، چند آهنگ/خواننده/سبک واقعی و
        مشابه پیشنهاد می‌دهد.

        هشدار مهم: چون از یک مدل نسبتاً کوچک و بدون دسترسی به اینترنت
        استفاده می‌کنیم، ممکن است اسم آهنگ/خواننده اشتباه یا حتی کاملا
        ساختگی (hallucination) باشد. این خروجی باید به‌عنوان «پیشنهاد
        برای بررسی بیشتر» به کاربر نمایش داده شود، نه یک منبع تضمینی.
        """
        if not self.is_loaded:
            return "متاسفانه مدل زبانی برای پیشنهاد آهنگ در دسترس نیست."

        user_msg = (
            f"A user built a custom music style through an interactive process. "
            f"Their final preferred style, described in words: \"{final_prompt}\"\n"
        )
        if original_user_prompt:
            user_msg += f"Their original starting description was: \"{original_user_prompt}\"\n"
        user_msg += (
            "Suggest 4 to 6 real, existing songs (with real artist names) and 2 to 3 "
            "real artists or bands that closely match this style. "
            "Format as a simple bulleted list, songs first then artists. "
            "Only suggest music you are confident actually exists - do not invent song "
            "or artist names."
        )
        messages = [
            {"role": "system", "content": "/no_think"},
            {"role": "user", "content": user_msg},
        ]
        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs, max_new_tokens=300, do_sample=True, temperature=0.7
                )
            output_ids = generated_ids[0][len(inputs.input_ids[0]):]
            raw = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        except Exception as e:
            return f"خطا در تولید پیشنهاد: {e}"

        return raw if raw else "پیشنهادی تولید نشد."
