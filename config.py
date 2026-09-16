CATEGORIES = {
    "Atmosphere": [
        "cinematic", "ambient", "ethereal", "futuristic",
        "dark", "mystical", "nostalgic", "peaceful", "dreamy",
        "energetic", "gritty", "intense", "warm", "raw"
    ],
    "Genre": [
        "lo-fi", "classical", "jazz", "synthwave",
        "ambient", "folk", "electronic", "soundtrack",
        "rock", "pop", "hip-hop", "blues", "metal",
        "funk", "reggae", "punk", "country"
    ],
    "Instrument": [
        "piano", "acoustic guitar", "synthesizer", "violin",
        "flute", "cello", "harp", "electric guitar",
        "drums", "bass guitar", "saxophone", "trumpet", "organ"
    ],
    "Emotion": [
        "melancholic", "serene", "joyful", "hopeful",
        "mysterious", "calm", "relaxing", "uplifting",
        "aggressive", "rebellious", "triumphant", "nostalgic", "powerful"
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
