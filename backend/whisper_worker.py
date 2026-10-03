"""Whisper running in its own process.

Inside the main backend process Whisper competes with face/object recognition and the web server
for the Python interpreter, and a 1.5 s chunk took 1.3-6.5 s to transcribe. In a separate process
the same model takes ~0.8 s. Keep this module light: it must not import TensorFlow/PyTorch.
"""

_model = None


def load_model(size: str, threads: int):
    """Process initializer: load the model once when the worker process starts."""
    global _model
    from faster_whisper import WhisperModel
    _model = WhisperModel(size, device="cpu", compute_type="int8", cpu_threads=threads)


def ready() -> bool:
    return _model is not None


def transcribe(audio_path: str, hint: str) -> str:
    """Transcribe an audio file and return the cleaned text."""
    segments, _info = _model.transcribe(
        audio_path,
        language="en",
        beam_size=3,
        vad_filter=True,          # skip silence segments
        vad_parameters=dict(min_silence_duration_ms=500),
        hotwords=hint,            # without it Whisper writes "Kinley"/"Kennedy"/"kindly"
    )
    return " ".join(seg.text.strip() for seg in segments).strip()
