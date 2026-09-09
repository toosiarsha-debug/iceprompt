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
