CATEGORIES = {
    "Atmosphere": [
        "cinematic", "ambient", "ethereal", "futuristic",
        "dark", "mystical", "nostalgic", "peaceful", "dreamy"
    ],
    "Genre": [
        "lo-fi", "classical", "jazz", "synthwave",
        "ambient", "folk", "electronic", "soundtrack"
    ],
    "Instrument": [
        "piano", "acoustic guitar", "synthesizer", "violin",
        "flute", "cello", "harp", "electric guitar"
    ],
    "Emotion": [
        "melancholic", "serene", "joyful", "hopeful",
        "mysterious", "calm", "relaxing", "uplifting"
    ]
}

MODEL_CONFIG = {
    "musicgen_model": "facebook/musicgen-small",
    "embedding_model": "all-MiniLM-L6-v2",
    "generation_duration": 8,
    "num_candidates": 3,
    "baseline_score": 5.0
}


class AppConfig:
    MODEL_NAME = MODEL_CONFIG["musicgen_model"]
    DURATION = MODEL_CONFIG["generation_duration"]
    GUIDANCE_SCALE = 3.0
    POPULATION_SIZE = MODEL_CONFIG["num_candidates"]
    MAX_GENERATIONS = 3
    CATEGORIES = CATEGORIES
    MODEL_CONFIG = MODEL_CONFIG
