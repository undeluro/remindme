import base64
import tempfile
import os
import subprocess
import time
import shutil

# Resolve ffmpeg path once at import time (Windows venv/uv may strip PATH)
_FFMPEG_PATH = shutil.which("ffmpeg") or "ffmpeg"
print(f"[VoiceProcessor] Using ffmpeg at: {_FFMPEG_PATH}")
import db

MAX_VECTORS_PER_PERSON = 20


def _patch_symlinks_for_windows():
    """On Windows, replace os.symlink with a copy fallback to avoid WinError 1314."""
    if os.name == 'nt':
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        _original_symlink = os.symlink

        def _symlink_or_copy(src, dst, target_is_directory=False, *args, **kwargs):
            try:
                _original_symlink(src, dst, target_is_directory=target_is_directory)
            except OSError:
                # Symlink failed (no privilege) — fall back to copy
                if os.path.isdir(src):
                    if not os.path.exists(dst):
                        shutil.copytree(src, dst)
                else:
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copy2(src, dst)

        os.symlink = _symlink_or_copy


class VoiceProcessor:
    def __init__(self):
        print("[VoiceProcessor] Loading SpeechBrain ECAPA-TDNN model...")
        try:
            _patch_symlinks_for_windows()
            from speechbrain.inference.speaker import EncoderClassifier
            self.encoder = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/spkrec-ecapa-voxceleb",
                run_opts={"device": "cpu"}
            )
            print("[VoiceProcessor] ECAPA-TDNN model loaded successfully.")
        except Exception as e:
            print(f"[VoiceProcessor] CRITICAL ERROR loading SpeechBrain: {e}")
            self.encoder = None

    def _webm_to_wav(self, webm_path: str) -> str | None:
        """Convert a webm audio file to 16kHz mono WAV using ffmpeg."""
        wav_path = webm_path.replace(".webm", ".wav")
        try:
            subprocess.run(
                [
                    _FFMPEG_PATH, "-y", "-i", webm_path,
                    "-ar", "16000", "-ac", "1", "-f", "wav", wav_path
                ],
                capture_output=True, check=True, timeout=10
            )
            return wav_path
        except subprocess.CalledProcessError as e:
            print(f"[VoiceProcessor] ffmpeg conversion error: {e.stderr.decode()[:200]}")
            return None
        except Exception as e:
            print(f"[VoiceProcessor] ffmpeg error: {e}")
            return None

    def _load_audio(self, wav_path: str):
        """Load a WAV file as a tensor suitable for SpeechBrain.
        Uses stdlib wave module to avoid torchaudio's TorchCodec dependency.
        """
        import wave
        import torch
        import numpy as np

        with wave.open(wav_path, 'rb') as wf:
            frames = wf.readframes(wf.getnframes())

        # Convert raw PCM bytes to float32 tensor (shape: [1, num_samples])
        signal = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        signal = torch.tensor(signal).unsqueeze(0)
        return signal

    def generate_embedding(self, audio_b64: str) -> list[float] | None:
        """Generate a 192-dimensional voice embedding from base64-encoded webm audio."""
        if not self.encoder:
            return None

        t0 = time.time()
        fh, temp_webm = tempfile.mkstemp(suffix=".webm")
        temp_wav = None
        try:
            # Write base64 audio to temp webm file
            audio_bytes = base64.b64decode(audio_b64)
            with os.fdopen(fh, 'wb') as f:
                f.write(audio_bytes)

            # Skip tiny audio chunks (< 10KB usually means silence)
            if os.path.getsize(temp_webm) < 10000:
                print("[VoiceProcessor] Audio chunk too small, skipping embedding")
                return None

            # Convert webm to wav
            temp_wav = self._webm_to_wav(temp_webm)
            if not temp_wav:
                return None

            # Load audio and generate embedding
            signal = self._load_audio(temp_wav)
            embedding = self.encoder.encode_batch(signal)
            embedding_list = embedding.squeeze().tolist()

            elapsed = (time.time() - t0) * 1000
            print(f"[VoiceProcessor] Embedding generated in {elapsed:.0f}ms (dim={len(embedding_list)})")
            return embedding_list

        except Exception as e:
            print(f"[VoiceProcessor] Embedding generation error: {e}")
            return None
        finally:
            if os.path.exists(temp_webm):
                os.remove(temp_webm)
            if temp_wav and os.path.exists(temp_wav):
                os.remove(temp_wav)

    def identify_speaker(self, audio_b64: str) -> tuple[str | None, list[float] | None]:
        """Identify a speaker from audio. Returns (name_or_none, embedding_or_none)."""
        embedding = self.generate_embedding(audio_b64)
        if not embedding:
            return None, None

        meta = db.query_voice(embedding)
        if meta:
            name = meta.get("name", "Unknown")
            print(f"[VoiceProcessor] Speaker identified: {name}")
            return name, embedding
        else:
            print("[VoiceProcessor] Speaker not recognized (no match in DB)")
            return None, embedding

    def enroll_speaker(self, audio_b64: str, name: str) -> dict:
        """Enroll a speaker's voice into ChromaDB. Generates embedding and stores it."""
        embedding = self.generate_embedding(audio_b64)
        if not embedding:
            return {"error": "Could not generate voice embedding from audio"}

        vec_count = db.count_voice_vectors(name)
        if vec_count >= MAX_VECTORS_PER_PERSON:
            print(f"[VoiceProcessor] Voice enrollment skipped for '{name}': already at max ({vec_count} vectors)")
            return {"message": f"'{name}' already has maximum voice vectors ({vec_count})"}

        db.add_voice(embedding, {"name": name})
        new_count = vec_count + 1
        print(f"[VoiceProcessor] Enrolled voice for '{name}' ({new_count}/{MAX_VECTORS_PER_PERSON} vectors)")
        return {"message": f"Voice enrolled for '{name}' ({new_count} vectors)"}
