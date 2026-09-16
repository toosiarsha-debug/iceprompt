# IEC Prompt Optimization for Music Generation

سامانه‌ی تولید موسیقی با بازخورد تعاملی کاربر (Interactive Evolutionary
Computation) روی پرامپت‌های `facebook/musicgen-small`.

## فایل‌های فعال (این‌ها را import می‌کنند و اجرا می‌شوند)

| فایل | نقش |
|---|---|
| `config.py` | کاتالوگ کلمات (`CATEGORIES`) و تنظیمات مدل (`AppConfig`) |
| `generator.py` | تولید صوت با MusicGen (`AudioGenerator` / `MusicGenerator`) |
| `optimizer.py` | هسته‌ی IEC: `FreeTextIECOptimizer` — ساخت جمعیت اولیه و تکامل بر اساس امتیاز کاربر |
| `app.py` | رابط Gradio؛ حلقه‌ی نسل‌ها و امتیازدهی |
| `requirements.txt` | وابستگی‌های پایتون |

## فایل‌های مرده (در پروژه هستند ولی هیچ‌جا import نمی‌شوند)

- `llm_generator.py` (کلاس `LLMPromptMutator`) — یک مسیر جایگزین برای
  تولید واریانت پرامپت با یک مدل text2text (flan-t5-base) که هرگز به
  `app.py` یا `optimizer.py` وصل نشده است. پارامتر `positive_history`
  را می‌گیرد ولی هیچ‌وقت استفاده نمی‌کند.
- `generator.py.backup` — نسخه‌ی ساده‌تر و قدیمی‌تر `MusicGenerator`
  (بدون batch، بدون مدیریت انعطاف‌پذیر `config`).

پیشنهاد: اگر قصد استفاده‌ی آینده از این دو ندارید، حذفشان کنید تا در
import و دیباگ آینده اشتباهی پیش نیاید:

```bash
git rm llm_generator.py generator.py.backup
git commit -m "chore: remove unused dead-code files"
```

اگر می‌خواهید مسیر LLM-based mutation را واقعاً به IEC وصل کنید (به‌جای
انتخاب کلمه از کاتالوگ ثابت)، بگویید تا `optimizer.evolve` را طوری
بازنویسی کنم که از `positive_history` واقعی استفاده کند.

## اجرا

```bash
pip install -r requirements.txt
python app.py
```

## خلاصه‌ی اصلاحات نسبت به نسخه‌ی قبلی

جزئیات کامل در `CHANGELOG_FIX.md`. خلاصه:

1. **باگ اصلی (optimizer.py):** شباهت کسینوسی بین امبدینگ یک کلمه‌ی
   تنها و یک جمله‌ی کامل در بازه‌ای خیلی باریک می‌افتد؛ بدون
   min-max normalize قبل از softmax، انتخاب کلمه عملا رندوم بود و
   امتیاز کاربر تاثیر واقعی نداشت. حالا اصلاح شده.
2. `pop_size` قبلا نادیده گرفته می‌شد (همیشه دقیقا ۳ پرامپت هاردکد
   تولید می‌شد)؛ حالا با `pop_size` واقعی هماهنگ است.
3. `learning_rate=0.7` ثابت خیلی تهاجمی بود و در ۳ نسل بردار اولیه‌ی
   کاربر را محو می‌کرد؛ حالا کاهشی است (`0.45 / (1 + 0.5*generation)`).
4. `app.py` مقادیر `MAX_GENERATIONS`/`POPULATION_SIZE` را از
   `config.py` می‌خواند، نه هاردکد جدا.
