import gradio as pd_ui
from config import CATEGORIES
from optimizer import IECOptimizer
from generator import MusicGenerator

optimizer = IECOptimizer()
generator = MusicGenerator()
current_generation_idx = 0


def start_system(atmosphere, genre, instrument, emotion):
    global current_generation_idx
    current_generation_idx = 1
    seed = {
        "Atmosphere": atmosphere,
        "Genre": genre,
        "Instrument": instrument,
        "Emotion": emotion
    }
    candidates = optimizer.initialize_population(seed)
    
    audios = []
    prompts = []
    for cand in candidates:
        p_text = optimizer.build_prompt_text(cand)
        prompts.append(p_text)
        sr, audio = generator.generate(p_text)
        audios.append((sr, audio))

    gen_label = f"Generation {current_generation_idx}"
    return (
        gen_label,
        prompts[0], audios[0],
        prompts[1], audios[1],
        prompts[2], audios[2]
    )


def next_generation(s1, s2, s3):
    global current_generation_idx
    current_generation_idx += 1
    scores = [s1, s2, s3]
    candidates = optimizer.evolve_step(scores)

    audios = []
    prompts = []
    for cand in candidates:
        p_text = optimizer.build_prompt_text(cand)
        prompts.append(p_text)
        sr, audio = generator.generate(p_text)
        audios.append((sr, audio))

    gen_label = f"Generation {current_generation_idx}"
    return (
        gen_label,
        prompts[0], audios[0],
        prompts[1], audios[1],
        prompts[2], audios[2]
    )


with pd_ui.Blocks(title="IEC Prompt Optimizer for Music Generation") as demo:
    pd_ui.Markdown("## Interactive Evolutionary Music Prompt Optimization")
    pd_ui.Markdown("Adjust scores to guide the system toward your preferred musical style.")

    with pd_ui.Row():
        at_input = pd_ui.Dropdown(CATEGORIES["Atmosphere"], value="cinematic", label="Atmosphere")
        gn_input = pd_ui.Dropdown(CATEGORIES["Genre"], value="ambient", label="Genre")
        in_input = pd_ui.Dropdown(CATEGORIES["Instrument"], value="piano", label="Instrument")
        em_input = pd_ui.Dropdown(CATEGORIES["Emotion"], value="serene", label="Emotion")

    start_btn = pd_ui.Button("Initialize Generation 1", variant="primary")
    status_header = pd_ui.Markdown("### Generation Status: Ready")

    with pd_ui.Row():
        with pd_ui.Column():
            p1_text = pd_ui.Textbox(label="Candidate 1 Prompt", interactive=False)
            a1_audio = pd_ui.Audio(label="Candidate 1 Audio")
            s1_score = pd_ui.Slider(1, 10, value=5, step=1, label="Score (1-10)")

        with pd_ui.Column():
            p2_text = pd_ui.Textbox(label="Candidate 2 Prompt", interactive=False)
            a2_audio = pd_ui.Audio(label="Candidate 2 Audio")
            s2_score = pd_ui.Slider(1, 10, value=5, step=1, label="Score (1-10)")

        with pd_ui.Column():
            p3_text = pd_ui.Textbox(label="Candidate 3 Prompt", interactive=False)
            a3_audio = pd_ui.Audio(label="Candidate 3 Audio")
            s3_score = pd_ui.Slider(1, 10, value=5, step=1, label="Score (1-10)")

    evolve_btn = pd_ui.Button("Evolve to Next Generation", variant="secondary")

    start_btn.click(
        fn=start_system,
        inputs=[at_input, gn_input, in_input, em_input],
        outputs=[status_header, p1_text, a1_audio, p2_text, a2_audio, p3_text, a3_audio]
    )

    evolve_btn.click(
        fn=next_generation,
        inputs=[s1_score, s2_score, s3_score],
        outputs=[status_header, p1_text, a1_audio, p2_text, a2_audio, p3_text, a3_audio]
    )

if __name__ == "__main__":
    demo.launch(share=False)
