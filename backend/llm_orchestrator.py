import os
import asyncio
import base64
import json
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import Literal, Optional

import db


class CommandAction(BaseModel):
    """Structured output schema for command processing."""
    user_transcript: str = Field(
        description="The exact text of what the user said in the audio."
    )
    action: Literal["add_note", "set_reminder", "retrieve_memory", "recognize_object", "respond"] = Field(
        description="The action to take based on the user's command."
    )
    note_text: Optional[str] = Field(
        None, description="The text content of the note, if action is add_note or recognize_object."
    )
    reminder_time: Optional[str] = Field(
        None, description="When to remind, if action is set_reminder."
    )
    reminder_task: Optional[str] = Field(
        None, description="What to remind about, if action is set_reminder."
    )
    memory_query: Optional[str] = Field(
        None, description="The query to search memories, if action is retrieve_memory."
    )
    object_name: Optional[str] = Field(
        None, description="The name or label the user gives to the object they want to save, if action is recognize_object."
    )
    spoken_response: str = Field(
        "", description="A friendly text response to show the user in the chat."
    )


class LLMOrchestrator:
    def __init__(self, ws_send_callback):
        self.ws_send_callback = ws_send_callback
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            print("WARNING: GEMINI_API_KEY missing. LLM features will not work.")
            self.client = None
        else:
            try:
                self.client = genai.Client(api_key=api_key)
            except Exception as e:
                print(f"Error initializing Gemini Client: {e}")
                self.client = None

    async def process_command(self, audio_b64: str, face_context: list) -> dict | None:
        """Process a spoken voice command directly from audio using Gemini structured output."""
        if not self.client:
            return None
        
        try:
            # Build context
            patient_context = db.build_context_string()
            
            # Format visible faces
            face_info = ""
            if face_context:
                face_names = [f"{f.get('name', 'Unknown')} ({f.get('relation', 'N/A')})" for f in face_context]
                face_info = f"\nCurrently visible people: {', '.join(face_names)}"
            
            system_prompt = f"""You are a memory companion assistant for people with memory difficulties or cognitive problem.
Based on the user's spoken command, determine what action to take.

{patient_context}
{face_info}

Actions:
- add_note: Save information the user wants to remember
- set_reminder: Set a reminder for a specific time/event
- retrieve_memory: Search past memories/notes for information
- recognize_object: The user wants to save/remember an object currently visible in front of the camera. Extract the note/description they want to attach (note_text) and optionally the name they use for the object (object_name). The system will detect and photograph the central object automatically.
- respond: Just respond conversationally (no special action needed)

Always provide a friendly spoken_response.
When the user mentions wanting to remember, save, or recognize an object/item/thing in front of them (e.g. medicine, keys, food), use recognize_object."""

            audio_bytes = base64.b64decode(audio_b64)
            response = await self.client.aio.models.generate_content(
                model="gemini-3.1-flash-lite-preview",
                contents=[
                    types.Part.from_text(text=system_prompt),
                    types.Part.from_bytes(data=audio_bytes, mime_type="audio/webm")
                ],
                config={
                    "response_mime_type": "application/json",
                    "response_json_schema": CommandAction.model_json_schema(),
                }
            )
            
            result = json.loads(response.text)
            print(f"[LLM] Structured output: {result}")
            return result
            
        except Exception as e:
            print(f"Error processing command: {e}")
            return {"action": "respond", "spoken_response": f"Sorry, I had trouble understanding that. ({e})"}

    async def generate_ambient_summary(self, base64_image: str) -> str:
        if not self.client:
            return ""
        try:
            image_bytes = base64.b64decode(base64_image)
            response = await self.client.aio.models.generate_content(
                model="gemini-3.1-flash-lite-preview",
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                    "Describe the patient's surroundings in 1 short sentence."
                ]
            )
            return response.text
        except Exception as e:
            print(f"Error generating ambient summary: {e}")
            return ""

    async def generate_session_summary(self, session_log: list[str]) -> str:
        if not self.client or not session_log:
            return ""
        try:
            prompt = "Summarize the recent session with the patient focusing on key events and contexts:\n" + "\n".join(session_log)
            response = await self.client.aio.models.generate_content(
                model="gemini-3.1-flash-lite-preview",
                contents=prompt
            )
            return response.text
        except Exception as e:
            print(f"Error generating session summary: {e}")
            return ""

    def _init_gemma(self):
        """Lazy-load Gemma 3 1B for topic extraction. Uses MPS on Apple Silicon, CPU otherwise."""
        if not hasattr(self, '_gemma_pipe') or self._gemma_pipe is None:
            import torch
            from transformers import pipeline

            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

            print(f"[LLM] Loading Gemma 3 1B on {device}...")
            self._gemma_pipe = pipeline(
                "text-generation",
                model="google/gemma-3-1b-it",
                device=device,
                torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
            )
            print(f"[LLM] Gemma 3 1B loaded successfully on {device}.")
        return self._gemma_pipe

    def extract_topic_sync(self, transcript: str) -> str | None:
        """Extract a short readable conversation topic using Gemma 3 1B (local)."""
        if not transcript.strip() or len(transcript.strip()) < 15:
            return None
        try:
            pipe = self._init_gemma()

            # Cap transcript to avoid excessive input
            text = transcript.strip()[:1500]

            messages = [
                {
                    "role": "user",
                    "content": (
                        "Extract the main conversation topic from this transcript in 3-6 words. "
                        "Reply with ONLY the topic, nothing else.\n\n"
                        f"Transcript: {text}"
                    ),
                }
            ]

            result = pipe(messages, max_new_tokens=20, do_sample=False)
            topic = result[0]["generated_text"][-1]["content"].strip()

            # Clean up: remove quotes, periods, prefixes
            topic = topic.strip('"\'.,!').strip()
            if topic.lower().startswith("topic:"):
                topic = topic[6:].strip()

            if topic and 3 < len(topic) < 80:
                print(f"[LLM] Gemma topic extracted: '{topic}'")
                return topic

            print(f"[LLM] Gemma topic rejected (len={len(topic)}): '{topic}'")
            return None
        except Exception as e:
            print(f"[LLM] Gemma topic extraction error: {e}")
            return None
