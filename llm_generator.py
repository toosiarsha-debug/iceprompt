import os
import numpy as np

class LLMPromptMutator:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

    def generate_variations(self, base_prompt, positive_history, count=4):
        system_instruction = (
            "You are an expert AI music prompt engineer. "
            "Given a base music idea and preferred styles, generate variations that explore "
            "different instruments, tempos, atmospheres, and production qualities."
        )

        history_context = ""
        if positive_history:
            history_context = f"User previously highly rated prompts like: {'; '.join(positive_history[-3:])}"

        user_instruction = f"""
        Base Idea: '{base_prompt}'
        {history_context}
        
        Generate {count} unique, detailed, natural-language music generation prompts.
        Return only the prompts, one per line.
        """

        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_instruction}
                ],
                temperature=0.8
            )
            prompts = response.choices[0].message.content.strip().split("\n")
            return [p.strip("- ").strip() for p in prompts if p.strip()][:count]
        except Exception:
            modifiers = [
                "rich cinematic orchestration and ambient pads",
                "upbeat rhythm with deep bass and melodic synths",
                "lo-fi chill vibes with smooth electric piano",
                "acoustic organic textures with warm atmospheric reverb"
            ]
            return [f"{base_prompt}, {mod}" for mod in modifiers[:count]]
