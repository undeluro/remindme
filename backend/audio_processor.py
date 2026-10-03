import base64
import os
import re
import tempfile
import time
import asyncio
import multiprocessing
import ssl
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

import whisper_worker

ssl._create_default_https_context = ssl._create_unverified_context

# Wake phrase: "Hey Kinly" (also "Hi/Okay Kinly"). Whisper doesn't know the name and hears
# "Kinley"/"Kennedy"/"kindly", so it gets the name as a hint; accents can still merge it into
# one word ("HIKINLY"), so we also match spellings with the spaces removed.
WAKE_HINT = "Kinly"
_WAKE_PATTERN = re.compile(r"\b(hey|hi|hay|okay|ok)\s*kin(ly|li|ley|lee)\b")
_WAKE_MERGED = ("heykinl", "hikinl", "haykinl", "okaykinl", "okkinl")
# Backup phrase — whole phrase only ("remind" alone appears in normal conversation)
_BACKUP_PATTERN = re.compile(r"\bremind me\b")

# "base" hears "Hey Kinly" as reliably as "small" in our tests (10/10, no false wakes) but is ~3x
# faster on CPU (~0.7 s vs ~2 s per chunk) — that was most of the delay before Command Mode opened
WHISPER_MODEL = "base"
# Skip near-empty recordings (just the WebM header). 1.5 s chunks of quiet speech can be only a few KB,
# so this must stay well below that; silence is filtered by Whisper's VAD anyway
MIN_CHUNK_BYTES = 2000


def is_wake_phrase(text: str) -> bool:
    """True if a transcript contains "Hey Kinly" (or a variant) or "Remind me"."""
    words = re.sub(r"[^a-z ]", " ", text.lower())
    words = re.sub(r"\s+", " ", words).strip()
    merged = words.replace(" ", "")
    return bool(
        _WAKE_PATTERN.search(words)
        or any(m in merged for m in _WAKE_MERGED)
        or _BACKUP_PATTERN.search(words)
    )


class AudioProcessor:
    def __init__(self):
        print(f"[AudioProcessor] Starting faster-whisper {WHISPER_MODEL} (int8) in its own process...")
        self.model = None
        self._executor = None
        try:
            self._start_worker()
            self.model = True
            print(f"[AudioProcessor] faster-whisper {WHISPER_MODEL} model loaded successfully.")
        except Exception as e:
            print(f"[AudioProcessor] CRITICAL ERROR loading faster-whisper: {e}")
        # Previous chunk's text: a wake phrase can be split between two audio chunks ("Hey" | "Kinly")
        self._previous_text = ""

    def _start_worker(self):
        """Run Whisper in a separate process (see whisper_worker.py) and wait until its model is loaded.

        "spawn" starts a clean interpreter: forking a process that already runs TensorFlow/PyTorch
        threads can deadlock.
        """
        self._executor = ProcessPoolExecutor(
            max_workers=1,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=whisper_worker.load_model,
            initargs=(WHISPER_MODEL, int(os.environ.get("MODEL_THREADS", "4"))),
        )
        if not self._executor.submit(whisper_worker.ready).result(timeout=300):
            raise RuntimeError("Whisper worker failed to load the model")

    async def _transcribe(self, audio_path: str) -> str:
        """Transcribe in the Whisper process; restart it if it crashed."""
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(self._executor, whisper_worker.transcribe, audio_path, WAKE_HINT)
        except BrokenProcessPool:
            print("[AudioProcessor] Whisper process crashed — restarting it")
            await asyncio.to_thread(self._start_worker)
            return ""

    async def detect_hotword(self, chunk_data_b64: str) -> tuple[bool, str]:
        """
        Expects a base64 encoded webm chunk containing audio.
        Transcribes it and checks for the hotword.
        Returns (is_hotword, transcribed_text).
        """
        if not self.model:
            return False, ""
        chunk_data = base64.b64decode(chunk_data_b64)
        
        # We need a temporary file for whisper and ffmpeg to process
        fh, temp_path = tempfile.mkstemp(suffix=".webm")
        try:
            with os.fdopen(fh, 'wb') as f:
                f.write(chunk_data)
                
            if os.path.getsize(temp_path) < MIN_CHUNK_BYTES:
                return False, ""

            started = time.perf_counter()
            clean_text = await self._transcribe(temp_path)
            took_ms = (time.perf_counter() - started) * 1000
            text = clean_text.lower()

            # Filter out empty/garbage transcriptions
            if not clean_text or clean_text in [".", "...", " "]:
                clean_text = ""

            # Check this chunk together with the previous one, in case the phrase was split between them
            combined = f"{self._previous_text} {text}"
            self._previous_text = text
            if is_wake_phrase(combined):
                print(f"[AudioProcessor] Hotword detected ({took_ms:.0f} ms) -> {combined.strip()}")
                self._previous_text = ""  # don't trigger twice on the same words
                return True, clean_text

            if clean_text:
                print(f"[AudioProcessor] Transcribed in {took_ms:.0f} ms (no hotword): {clean_text[:80]}")

            return False, clean_text
                
        except Exception as e:
            print(f"[AudioProcessor] Transcription error: {e}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
        return False, ""

    async def transcribe(self, chunk_data_b64: str) -> str:
        """Transcribe audio chunk and return the text. Returns empty string on failure."""
        if not self.model:
            return ""
        
        chunk_data = base64.b64decode(chunk_data_b64)
        
        fh, temp_path = tempfile.mkstemp(suffix=".webm")
        try:
            with os.fdopen(fh, 'wb') as f:
                f.write(chunk_data)
                
            if os.path.getsize(temp_path) < MIN_CHUNK_BYTES:
                return ""

            text = await self._transcribe(temp_path)
            
            if text and text not in [".", "...", " "]:
                return text
                
        except Exception as e:
            print(f"Audio transcription error: {e}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
        return ""

