import base64
import os
import tempfile
import asyncio
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


class AudioProcessor:
    def __init__(self):
        print("[AudioProcessor] Loading faster-whisper small model (int8)...")
        try:
            from faster_whisper import WhisperModel
            self.model = WhisperModel(
                "small",
                device="cpu",
                compute_type="int8",  # quantized for speed on CPU
            )
            print("[AudioProcessor] faster-whisper small model loaded successfully.")
        except Exception as e:
            print(f"[AudioProcessor] CRITICAL ERROR loading faster-whisper: {e}")
            self.model = None

    def _transcribe_sync(self, audio_path: str) -> str:
        """Run faster-whisper transcription synchronously. Returns cleaned text."""
        segments, _info = self.model.transcribe(
            audio_path,
            language="en",
            beam_size=3,
            vad_filter=True,          # skip silence segments
            vad_parameters=dict(min_silence_duration_ms=500),
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return text

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
                
            if os.path.getsize(temp_path) < 10000:
                return False, ""

            clean_text = await asyncio.to_thread(self._transcribe_sync, temp_path)
            text = clean_text.lower()
            
            # Filter out empty/garbage transcriptions
            if not clean_text or clean_text in [".", "...", " "]:
                clean_text = ""
            
            if "remind me" in text or "remind" in text:
                print(f"[AudioProcessor] Hotword detected -> {text}")
                return True, clean_text
            
            if clean_text:
                print(f"[AudioProcessor] Transcribed (no hotword): {clean_text[:80]}")
            
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
                
            if os.path.getsize(temp_path) < 10000:
                return ""

            text = await asyncio.to_thread(self._transcribe_sync, temp_path)
            
            if text and text not in [".", "...", " "]:
                return text
                
        except Exception as e:
            print(f"Audio transcription error: {e}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
        return ""

