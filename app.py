import gradio as gr
import numpy as np
from optimizer import FreeTextIECOptimizer
from llm_generator import LLMPromptMutator
from generator import MusicGenerator

optimizer = FreeTextIECOptimizer()
mutator = LLMPromptMutator()
music_gen = MusicGenerator()

current_prompts = []

def start_session(user_input_prompt):
    global current_prompts
    optimizer.initialize_base(user_input_prompt)
    current_prompts = mutator.generate_variations(user_input_prompt, optimizer.positive_prompts_history, count=4)
    
    audio_paths = [music_gen.generate(p) for p in current_prompts]
    
    return (
        current_prompts[0], audio_paths[0],
        current_prompts[1], audio_paths[1],
        current_prompts[2], audio_paths[2],
        current_prompts[3], audio_paths[3]
    )

def next_generation(s1, s2, s3, s4):
    global current_prompts
    scores = [s1, s2, s3, s4]
    
    optimizer.update_objective_and_get_target(current_prompts, scores)
    
    base_text = current_prompts[int(np.argmax(scores))]
    current_prompts = mutator.generate_variations(base_text, optimizer.positive_prompts_history, count=4)
    
    audio_paths = [music_gen.generate(p) for p in current_prompts]
    
    return (
        current_prompts[0], audio_paths[0],
        current_prompts[1], audio_paths[1],
        current_prompts[2], audio_paths[2],
        current_prompts[3], audio_paths[3]
    )

with gr.Blocks(title="Free-Text IEC Music Generator") as demo:
    gr.Markdown("# سامانه تولید موسیقی بر پایه بهینه‌سازی پرامپت آزاد (IEC + LLM)")
    
    with gr.Row():
        user_input = gr.Textbox(label="پرامپت اولیه خود را به زبان آزاد بنویسید", placeholder="مثال: A chill lo-fi beat with soft piano for studying on a rainy day")
        start_btn = gr.Button("شروع تولید نسل ۱")

    prompts_box = []
    audios_box = []
    scores_box = []

    for i in range(4):
        with gr.Group():
            gr.Markdown(f"### گزینه {i+1}")
            p_text = gr.Textbox(label="پرامپت تولید شده", interactive=False)
            a_play = gr.Audio(label="پخش موسیقی")
            s_slider = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="امتیاز شما (۱ تا ۱۰)")
            
            prompts_box.append(p_text)
            audios_box.append(a_play)
            scores_box.append(s_slider)

    next_btn = gr.Button("اعمال امتیازها و رفتن به نسل بعد")

    start_btn.click(
        start_session,
        inputs=[user_input],
        outputs=[
            prompts_box[0], audios_box[0],
            prompts_box[1], audios_box[1],
            prompts_box[2], audios_box[2],
            prompts_box[3], audios_box[3]
        ]
    )

    next_btn.click(
        next_generation,
        inputs=scores_box,
        outputs=[
            prompts_box[0], audios_box[0],
            prompts_box[1], audios_box[1],
            prompts_box[2], audios_box[2],
            prompts_box[3], audios_box[3]
        ]
    )

if __name__ == "__main__":
    demo.launch()
