import os
import torch
import gradio as gr
from config import AppConfig
from generator import AudioGenerator
from optimizer import FreeTextIECOptimizer

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

config = AppConfig()
generator = AudioGenerator(config)
optimizer = FreeTextIECOptimizer(config.MODEL_CONFIG['embedding_model'])

current_prompts = []
current_audios = []
history = []
generation_count = 0
MAX_GENERATIONS = 3

def initial_generation(base_prompt):
    global current_prompts, current_audios, history, generation_count
    generation_count = 1
    history = []
    
    current_prompts = optimizer.initialize_population(base_prompt)
    current_audios = [generator.generate(p) for p in current_prompts]
    
    status_text = f"نسل {generation_count} از {MAX_GENERATIONS} تولید شد. لطفاً به هر قطعه از ۱ تا ۵ امتیاز دهید."
    
    return [
        current_audios[0], current_prompts[0],
        current_audios[1], current_prompts[1],
        current_audios[2], current_prompts[2],
        status_text,
        gr.update(interactive=True, value="تکامل و تولید نسل بعدی")
    ]

def evolve_generation(r1, r2, r3):
    global current_prompts, current_audios, history, generation_count
    
    ratings = [int(r1), int(r2), int(r3)]
    history.append(list(zip(current_prompts, ratings)))
    
    if generation_count >= MAX_GENERATIONS:
        best_prompt = ""
        best_score = -1
        for gen in history:
            for p, s in gen:
                if s > best_score:
                    best_score = s
                    best_prompt = p
                    
        summary = f"🏁 روند تکامل در ۳ نسل به پایان رسید!\nبهترین پرامپت با امتیاز {best_score}: {best_prompt}"
        return [
            current_audios[0], current_prompts[0],
            current_audios[1], current_prompts[1],
            current_audios[2], current_prompts[2],
            summary,
            gr.update(interactive=False, value="تکامل به پایان رسید")
        ]
        
    generation_count += 1
    current_prompts = optimizer.evolve(current_prompts, ratings)
    current_audios = [generator.generate(p) for p in current_prompts]
    
    status_text = f"نسل {generation_count} از {MAX_GENERATIONS} تولید شد. لطفاً امتیاز دهید."
    
    return [
        current_audios[0], current_prompts[0],
        current_audios[1], current_prompts[1],
        current_audios[2], current_prompts[2],
        status_text,
        gr.update(interactive=True)
    ]

with gr.Blocks(title="سامانه آهنگسازی تکاملی تعاملی (IEC)") as demo:
    gr.Markdown("## سامانه تولید موسیقی با الگوریتم ژنتیک بر بستر پرامپت")
    gr.Markdown("یک پرامپت اولیه وارد کنید، به خروجی‌ها امتیاز دهید تا سیستم در ۳ نسل سبک دلخواه شما را بیاموزد.")
    
    with gr.Row():
        base_prompt_input = gr.Textbox(
            label="پرامپت اولیه (توضیح موزیک دلخواه به انگلیسی)",
            value="upbeat lo-fi chillhop beat with soft electric piano and warm bass"
        )
        init_btn = gr.Button("شروع و تولید نسل اول", variant="primary")
        
    status_box = gr.Textbox(label="وضعیت", interactive=False)
    
    with gr.Row():
        with gr.Column():
            audio_1 = gr.Audio(label="نمونه ۱")
            prompt_1 = gr.Textbox(label="پرامپت ۱", interactive=False)
            rating_1 = gr.Slider(minimum=1, maximum=5, step=1, value=3, label="امتیاز (۱ تا ۵)")
        with gr.Column():
            audio_2 = gr.Audio(label="نمونه ۲")
            prompt_2 = gr.Textbox(label="پرامپت ۲", interactive=False)
            rating_2 = gr.Slider(minimum=1, maximum=5, step=1, value=3, label="امتیاز (۱ تا ۵)")
        with gr.Column():
            audio_3 = gr.Audio(label="نمونه ۳")
            prompt_3 = gr.Textbox(label="پرامپت ۳", interactive=False)
            rating_3 = gr.Slider(minimum=1, maximum=5, step=1, value=3, label="امتیاز (۱ تا ۵)")
            
    evolve_btn = gr.Button("تکامل و تولید نسل بعدی", variant="secondary")
    
    init_btn.click(
        fn=initial_generation,
        inputs=[base_prompt_input],
        outputs=[audio_1, prompt_1, audio_2, prompt_2, audio_3, prompt_3, status_box, evolve_btn]
    )
    
    evolve_btn.click(
        fn=evolve_generation,
        inputs=[rating_1, rating_2, rating_3],
        outputs=[audio_1, prompt_1, audio_2, prompt_2, audio_3, prompt_3, status_box, evolve_btn]
    )

if __name__ == "__main__":
    demo.launch(share=True)
