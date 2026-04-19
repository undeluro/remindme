# RemindMe: AI Memory Companion

<p align="center">
  <img src="media/ui.png" alt="RemindMe UI" width="40%">
</p>
<p align="center">
  <img src="media/ui2.png" alt="RemindMe UI Command Mode">
</p>


RemindMe is an ambient AI memory companion designed to support individuals experiencing memory difficulties or cognitive impairments. Acting as a silent observer and helpful assistant, RemindMe continuously processes the user's environment and conversations, instantly springing to action upon hearing its hotword, ready to intelligently assist with daily life.

## 💡 The Idea
For individuals dealing with cognitive decline, interacting with complex phone apps to save a note or set a reminder can be incredibly daunting. RemindMe removes this friction entirely through **ambient intelligence**.

Our application runs passively in the background. It continuously views the environment using the camera, recognizes who the user is interacting with, and listens for the wake word: **"Remind me."** When triggered, RemindMe enters **Command Mode**. The user can speak complex, completely unscripted commands into the app—like "Remind me to call John back tomorrow" (while John is dynamically recognized in the room)—and RemindMe processes the direct raw audio to intelligently perform the requested action.

It's not just a voice assistant; it is a contextual memory aide that *sees* what you see and *remembers* what you forget.

---

## Key Features

- **Ambient Contextual Awareness:** The system passively captures the environment every 35 seconds, utilizing Vision LLMs to build a rolling memory log of the patient's surroundings and activities.
- **Robust Multi-Vector Face Tracking:** Uses a localized DeepFace and Vector DB architecture to identify family, friends, and caretakers. It runs a probabilistic continuous learning loop, saving multiple embedding vectors per person under different lighting to achieve bulletproof recognition (via majority voting).
- **Offline Hotword Detection:** Uses a lightweight `whisper-tiny` model running strictly locally to provide 100% offline, privacy-first, lightning-fast wake-word detection. 
- **Direct Audio-to-Action Intelligence:** When Command Mode activates, the system bypasses traditional, error-prone Speech-to-Text chains. Instead, it accumulates raw audio blobs and feeds them *natively* into **Gemini 3.1 Flash-Lite**. Gemini concurrently understands the audio, recognizes the intent, formulates a friendly spoken response, and fires off structured JSON actions (like `add_note` or `set_reminder`).
- **Reactive, Accessible UI:** Features a high-contrast, distraction-free React interface. The edges of the screen feature a neon glow border tightly coupled to the browser's Web Audio API—pulsing beautifully in real-time according to the user's microphone volume.

---

## Technical Details & Architecture

RemindMe relies on a powerful multi-modal separation between a modern React frontend and a heavyweight Python AI backend over WebSockets.

### Frontend
- **Framework:** React + Vite
- **Web APIs:** Heavy usage of `navigator.mediaDevices` for raw `audio/webm` and frame extraction. Integrates `AudioContext` and `AnalyserNode` for live Fast Fourier Transform (FFT) extraction, binding voice frequency data to CSS variables to animate the glowing UI natively.
- **Real-Time Communication:** A persistent WebSocket connection actively streams chunks and syncs UI state without HTTP overhead.

### Backend
- **Core Server:** FastAPI + Uvicorn handling heavy concurrency via asynchronous event loops. 
- **AI Orchestration Framework:** Powered heavily by the `google-genai` SDK. Implements structured outputs (Pydantic schemas) combined with Gemini 3.1 Flash-Lite's multi-modal capabilities (`types.Part.from_bytes`) to fuse raw environmental images, historical text context, and raw user audio into an intelligent output.
- **Audio Processing:** Local OpenAI Whisper (`tiny`) handles ambient hot-word detection to preserve API limits and reduce latencies.
- **Vector Database:** Integrates **ChromaDB**. Stores face embeddings created by **DeepFace** (using cosine similarity space). Built a custom K-Nearest Neighbors (KNN) top-5 majority voting loop for ultra-reliable facial recognition.

### Application Flow
1. **Ambient Mode:** UI captures 2.5s audio chunks and a frame per 1.25 sec. Sends chunks to local Whisper. Identifies people and updates the room's visual summary.
2. **Trigger:** Whisper detects "remind me". The backend sends a `STATE_CHANGE` to the frontend.
3. **Command Mode:** The frontend stops chunking and opens a continuous audio recorder. The UI's glow activates, pulsing to the user's voice.
4. **Execution:** User taps the screen to stop. The full audio payload is serialized via Base64 to FastAPI. Gemini processes the acoustic input, cross-references identified faces and memory logs, and executes actions.

---

### Running the Project Locally

**Requirements:**
- Python 3.10+
- Node.js > 18
- Gemini API Key

**Terminal 1 (Backend):**
```bash
cd backend
export GEMINI_API_KEY="your-api-key"
uv run fastapi dev
```
Or you can add the API key to .env file.

**Terminal 2 (Frontend):**
```bash
cd frontend
npm install
npm run dev
```

Navigate to `http://localhost:5173` to start using your Memory Companion!