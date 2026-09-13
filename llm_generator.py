import torch
from transformers import pipeline

class LLMPromptMutator:
    def __init__(self, model_name="google/flan-t5-base"):
        device = 0 if torch.cuda.is_available() else -1
        try:
            self.generator = pipeline(
                "text2text-generation",
                model=model_name,
                device=device
            )
            self.is_loaded = True
        except Exception as e:
            print(f"Loading fallback mutator due to: {e}")
            self.is_loaded = False

    def generate_variations(self, base_prompt, positive_history, count=4):
        styles = [
            "with cinematic ambient pads and orchestral depth",
            "featuring upbeat tempo, punchy drums, and deep bass synth",
            "in lo-fi chillhop style with warm vinyl crackle and rhodes piano",
            "acoustic organic vibe with delicate guitar and warm reverb",
            "dreamy atmospheric soundscape with slow attack pads",
            "energetic modern electronic beat with clean percussion"
        ]

        if not self.is_loaded:
            return [f"{base_prompt}, {style}" for style in styles[:count]]

        variations = []
        for i in range(count):
            style = styles[i % len(styles)]
            input_text = f"Expand and enhance this music prompt creatively: '{base_prompt}' {style}"
            
            try:
                res = self.generator(
                    input_text,
                    max_length=60,
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=0.7
                )
                generated = res[0]['generated_text'].strip()
                if len(generated) > 10:
                    variations.append(f"{base_prompt}, {generated}")
                else:
                    variations.append(f"{base_prompt}, {style}")
            except Exception:
                variations.append(f"{base_prompt}, {style}")

        return variations[:count]
