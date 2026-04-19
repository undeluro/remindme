from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import datetime
import json
import asyncio
import os
import ssl
from dotenv import load_dotenv

ssl._create_default_https_context = ssl._create_unverified_context

load_dotenv()

from face_processor import FaceProcessor
from audio_processor import AudioProcessor
from llm_orchestrator import LLMOrchestrator
from object_processor import ObjectProcessor
import edge_tts
import base64
import db

app = FastAPI()

async def async_generate_tts(text: str, voice: str = "en-US-AriaNeural") -> str:
    """Generate TTS audio and return base64 encoded MP3 string."""
    try:
        communicate = edge_tts.Communicate(text, voice)
        audio_data = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]
        return base64.b64encode(audio_data).decode('utf-8')
    except Exception as e:
        print(f"TTS generation error: {e}")
        return ""

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

face_processor = FaceProcessor()
audio_processor = AudioProcessor()
object_processor = ObjectProcessor()

class AddFaceRequest(BaseModel):
    name: str
    relation: str
    frame_b64: str
    face_box: dict | None = None  # {x, y, w, h} as percentages — to crop a specific face

@app.post("/api/faces")
async def add_new_face(req: AddFaceRequest):
    result = await asyncio.to_thread(face_processor.register_new_face, req.frame_b64, req.name, req.relation, req.face_box)
    if "error" in result:
        return {"success": False, "error": result["error"]}
    return {"success": True, "message": result.get("message", "Success")}

@app.get("/api/faces")
async def get_all_faces():
    import db
    faces = db.list_faces()
    return {"faces": faces}

@app.delete("/api/faces/{face_id}")
async def remove_face(face_id: str):
    import db
    db.delete_face(face_id)
    return {"success": True}

@app.get("/api/notes")
async def get_notes():
    import db
    notes = db.list_notes()
    return {"notes": notes}

class AddNoteRequest(BaseModel):
    text: str

@app.post("/api/notes")
async def create_note(req: AddNoteRequest):
    import db
    db.add_note(req.text)
    return {"success": True}

@app.delete("/api/notes/{index}")
async def remove_note(index: int):
    import db
    db.delete_note(index)
    return {"success": True}

# ========== Routines Endpoints ==========

class AddRoutineRequest(BaseModel):
    text: str

@app.get("/api/routines")
async def get_routines():
    import db
    routines = db.list_routines()
    return {"routines": routines}

@app.post("/api/routines")
async def create_routine(req: AddRoutineRequest):
    import db
    db.add_routine(req.text)
    return {"success": True}

@app.delete("/api/routines/{index}")
async def remove_routine(index: int):
    import db
    db.delete_routine(index)
    return {"success": True}

# ========== Reminders Endpoints ==========

class AddReminderRequest(BaseModel):
    time: str
    task: str

@app.get("/api/reminders")
async def get_reminders():
    import db
    reminders = db.list_reminders()
    return {"reminders": reminders}

@app.post("/api/reminders")
async def create_reminder(req: AddReminderRequest):
    import db
    db.add_reminder(req.time, req.task)
    return {"success": True}

@app.delete("/api/reminders/{index}")
async def remove_reminder(index: int):
    import db
    db.delete_reminder(index)
    return {"success": True}

# ========== Object Endpoints ==========

class AddObjectRequest(BaseModel):
    name: str
    notes: str = ""
    dosage: str = ""
    schedule: str = ""
    category: str = ""
    frame_b64: str
    object_box: dict  # {x, y, w, h} as percentages

class UpdateObjectRequest(BaseModel):
    notes: str | None = None
    dosage: str | None = None
    schedule: str | None = None
    category: str | None = None

@app.post("/api/objects")
async def add_object(req: AddObjectRequest):
    result = await asyncio.to_thread(
        object_processor.register_object,
        req.frame_b64, req.name, req.notes, req.dosage, req.schedule, req.category, req.object_box
    )
    if "error" in result:
        return {"success": False, "error": result["error"]}
    return {"success": True, "message": result.get("message", "Success")}

@app.get("/api/objects")
async def get_objects():
    import db
    objects = db.list_objects()
    return {"objects": objects}

@app.delete("/api/objects/{name}")
async def remove_object(name: str):
    import db
    db.delete_object(name)
    return {"success": True}

@app.put("/api/objects/{name}")
async def update_object(name: str, req: UpdateObjectRequest):
    import db
    updates = {k: v for k, v in req.model_dump().items() if v is not None}
    if updates:
        db.update_object_info(name, updates)
    return {"success": True}

# ========== Topic Endpoints ==========

class AddTopicRequest(BaseModel):
    topic: str

@app.get("/api/faces/{name}/topics")
async def get_face_topics(name: str):
    import db
    topics_data = db.get_face_topics(name)
    return {"topics": topics_data.get("topics", []) if topics_data else [], "last_topic": topics_data.get("last_topic", "") if topics_data else ""}

@app.post("/api/faces/{name}/topics")
async def add_face_topic(name: str, req: AddTopicRequest):
    import db
    if not req.topic.strip():
        return {"success": False, "error": "Topic cannot be empty"}
    db.update_face_topics(name, req.topic.strip())
    return {"success": True}

@app.delete("/api/faces/{name}/topics/{index}")
async def delete_face_topic(name: str, index: int):
    import db
    ok = db.delete_face_topic(name, index)
    if not ok:
        return {"success": False, "error": "Topic not found"}
    return {"success": True}

@app.get("/api/topics")
async def get_all_topics():
    import db
    all_topics = db.list_all_topics()
    return {"topics": all_topics}

class ConnectionManager:
    def __init__(self):
        self.active_connections = []
        self.mode = "AMBIENT"
        self.yolo_debug = False

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        # Send initial state
        await self.set_mode("AMBIENT", websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_json(self, message: dict, websocket: WebSocket):
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            print(f"WS send_json error: {e}")

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            await self.send_json(message, connection)
            
    async def set_mode(self, mode: str, websocket: WebSocket):
        self.mode = mode
        await self.send_json({"type": "STATE_CHANGE", "mode": mode}, websocket)

manager = ConnectionManager()

@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    async def _ws_send(payload):
        await manager.send_json(payload, websocket)
        
    llm = LLMOrchestrator(ws_send_callback=_ws_send)
    
    latest_frame = None
    session_log = []
    is_active = True  # Per-connection flag to control ambient loop
    current_faces = []  # Last known visible faces from FRAME handler
    
    async def ambient_loop():
        try:
            while is_active:
                await asyncio.sleep(30)
                if not is_active:
                    break
                if manager.mode == "AMBIENT" and latest_frame:
                    print("Running 30-sec Ambient Context Gather...")
                    summary = await llm.generate_ambient_summary(latest_frame)
                    if summary:
                        print(f"Ambient context: {summary}")
                        session_log.append(f"At {datetime.datetime.now().strftime('%H:%M:%S')}: {summary}")
                        await _ws_send({"type": "CHAT", "role": "ambient", "content": summary})
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"Ambient loop error: {e}")

    async def object_recognition_loop():
        last_matched_names = set()  # Track previous state to avoid wasteful sends
        try:
            while is_active:
                if not is_active:
                    break

                if manager.yolo_debug and latest_frame:
                    await asyncio.sleep(0.1) # Super fast refresh in debug mode
                    matched = await asyncio.to_thread(object_processor.detect_all_yolo_debug, latest_frame)
                    await _ws_send({"type": "OBJECTS", "objects": matched or []})
                    last_matched_names = set()

                elif manager.mode == "COMMAND" and latest_frame:
                    await asyncio.sleep(0.5)  # Faster refresh in command mode
                    matched = await asyncio.to_thread(object_processor.detect_central_yolo_object_preview, latest_frame)
                    print(f"[DEBUG] COMMAND mode YOLO detected {len(matched or [])} objects")
                    await _ws_send({"type": "OBJECTS", "objects": matched or []})
                    last_matched_names = set()  # Force ambient to refresh when switching back

                elif manager.mode == "AMBIENT" and latest_frame:
                    await asyncio.sleep(0.5)
                    # Skip entirely if no objects registered yet
                    if db.objects_count() == 0:
                        if last_matched_names != "empty":
                            await _ws_send({"type": "OBJECTS", "objects": []})
                            last_matched_names = "empty"
                        continue
                    
                    matched = await asyncio.to_thread(object_processor.recognize_known_objects, latest_frame)
                    current_names = frozenset(m["name"] for m in matched) if matched else frozenset()
                    
                    # Only send when the set of matched objects changes
                    if current_names != last_matched_names:
                        await _ws_send({"type": "OBJECTS", "objects": matched or []})
                        last_matched_names = current_names
                else:
                    await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"Object recognition loop error: {e}")

    bg_task = None
    obj_task = None
    
    try:
        await manager.connect(websocket)
        bg_task = asyncio.create_task(ambient_loop())
        obj_task = asyncio.create_task(object_recognition_loop())
        
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            
            p_type = payload.get("type")
            
            if p_type == "FRAME":
                img_data = payload.get("data")
                latest_frame = img_data
                
                # 1. Run Face Processor locally
                faces = await asyncio.to_thread(face_processor.process_frame, img_data)
                
                # Sort by x-position for consistent ordering (prevents tag swapping)
                faces.sort(key=lambda f: f["x"])
                
                # Update current visible faces for voice auto-enrollment
                current_faces.clear()
                current_faces.extend(faces)
                
                # Emit recognized faces back to the React UI
                await manager.send_json({"type": "FACES", "faces": faces}, websocket)

            elif p_type == "YOLO_DEBUG":
                manager.yolo_debug = payload.get("enabled", False)
                print(f"[Debug] YOLO Debug Mode set to {manager.yolo_debug}")

            elif p_type == "AUDIO":
                audio_data = payload.get("data")
                if manager.mode == "AMBIENT":
                    # Hotword detection
                    is_hotword, transcript_text = await audio_processor.detect_hotword(audio_data)
                    if is_hotword:
                        print("========== COMMAND MODE ACTIVATED ==========")
                        await manager.set_mode("COMMAND", websocket)
                        continue

            elif p_type == "COMMAND_AUDIO_FULL":
                audio_data = payload.get("data")
                print("[Native Audio] Sending complete command chunk to Gemini...")
                
                # Build face context for the LLM
                face_context = []
                if latest_frame:
                    face_context = await asyncio.to_thread(face_processor.process_frame, latest_frame)
                
                # Process command with structured output natively
                result = await llm.process_command(audio_data, face_context)
                action = None
                detection = None
                if result:
                    # Log the LLM's transcription
                    user_transcript = result.get("user_transcript", "[Uncertain Audio]")
                    await _ws_send({"type": "CHAT", "role": "user", "content": user_transcript})
                    
                    # Dispatch the action
                    action = result.get("action", "respond")
                    response_text = result.get("spoken_response", "")
                    
                    if action == "add_note" and result.get("note_text"):
                        db.add_note(result["note_text"])
                        response_text = response_text or f"Note added: {result['note_text']}"
                        print(f"[Command] Added note: {result['note_text']}")
                    elif action == "set_reminder" and result.get("reminder_task"):
                        db.add_reminder(result.get("reminder_time", "unspecified"), result["reminder_task"])
                        response_text = response_text or f"Reminder set: {result['reminder_task']}"
                        print(f"[Command] Set reminder: {result['reminder_task']}")
                    elif action == "retrieve_memory" and result.get("memory_query"):
                        memories = db.query_session_memories(result["memory_query"])
                        mem_text = " | ".join(memories) if memories else "No memories found."
                        response_text = response_text or f"Memory: {mem_text}"
                        print(f"[Command] Retrieved memory for: {result['memory_query']}")
                    elif action == "recognize_object":
                        print("[Command] recognize_object triggered — running YOLO detection...")
                        detection = await asyncio.to_thread(object_processor.detect_central_object, latest_frame)
                        if detection:
                            await _ws_send({
                                "type": "OBJECT_CONFIRM",
                                "crop_preview_b64": detection.get("crop_b64", ""),
                                "suggested_name": result.get("object_name", ""),
                                "note_text": result.get("note_text", ""),
                                "object_box": {
                                    "x": detection["x_pct"],
                                    "y": detection["y_pct"],
                                    "w": detection["w_pct"],
                                    "h": detection["h_pct"],
                                },
                                "frame_b64": latest_frame,
                            })
                            response_text = ""  # Don't send chat — waiting for UI confirm
                            print(f"[Command] OBJECT_CONFIRM sent to frontend")
                        else:
                            response_text = "Couldn't detect any objects, try moving the object closer."
                            print("[Command] No object detected in frame")
                    
                    if response_text:
                        await _ws_send({"type": "CHAT", "role": "assistant", "content": response_text})
                        
                        # Generate and send voice
                        print(f"[TTS] Generating audio for: {response_text}")
                        audio_b64 = await async_generate_tts(response_text)
                        if audio_b64:
                            await _ws_send({"type": "AUDIO_RESPONSE", "audio_b64": audio_b64})
                            print(f"[TTS] Sent audio payload")

                # Return to AMBIENT — UNLESS we're waiting for object confirmation
                if action == "recognize_object" and detection:
                    print("========== STAYING IN COMMAND (waiting for object confirm) ==========")
                else:
                    print("========== RETURNING TO AMBIENT MODE ==========")
                    await manager.set_mode("AMBIENT", websocket)

            elif p_type == "COMMAND" and payload.get("command") == "ACTIVATE_COMMAND":
                print("========== COMMAND MODE ACTIVATED (MANUAL) ==========")
                await manager.set_mode("COMMAND", websocket)

            elif p_type == "COMMAND" and payload.get("command") == "END_COMMAND":
                print("========== RETURNING TO AMBIENT MODE (USER CANCELED) ==========")
                await manager.set_mode("AMBIENT", websocket)

            elif p_type == "COMMAND" and payload.get("command") == "OBJECT_CONFIRM_DONE":
                print("========== OBJECT CONFIRM DONE — RETURNING TO AMBIENT ==========")
                await manager.set_mode("AMBIENT", websocket)

            elif p_type == "COMMAND" and payload.get("command") == "STOP":
                print("========== ASSISTING STOPPED ==========")
                is_active = False  # Kill the ambient loop
                if bg_task:
                    bg_task.cancel()
                    bg_task = None
                if obj_task:
                    obj_task.cancel()
                    obj_task = None
                await manager.set_mode("AMBIENT", websocket)
                
                if session_log:
                    print("Generating session summary and saving to Vector DB...")
                    final_summary = await llm.generate_session_summary(session_log)
                    if final_summary:
                        db.log_session_memory(final_summary)
                        print("Saved session summary!")
                        await _ws_send({"type": "CHAT", "role": "system", "content": f"Session Summarized: {final_summary}"})
                    session_log.clear()

            elif p_type == "COMMAND" and payload.get("command") == "START":
                print("========== ASSISTING STARTED ==========")
                is_active = True
                await manager.set_mode("AMBIENT", websocket)
                bg_task = asyncio.create_task(ambient_loop())
                obj_task = asyncio.create_task(object_recognition_loop())

    except WebSocketDisconnect:
        print("Websocket client disconnected naturally.")
    except Exception as e:
        print(f"WS Exception in streaming: {e}")
    finally:
        is_active = False
        if bg_task:
            bg_task.cancel()
        if obj_task:
            obj_task.cancel()
        manager.disconnect(websocket)

