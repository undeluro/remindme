import React, { useState, useEffect, useRef } from 'react';
import Webcam from 'react-webcam';
import { CommandBar, TopBar } from './components/CommandBar';
import DatabasePanel from './components/DatabasePanel';
import KinlyLogo from './components/KinlyLogo';
import './index.css';

const CAMERA_CONSTRAINTS = {
  facingMode: 'user',
  width: { ideal: 1280 },
  height: { ideal: 720 },
  aspectRatio: 16 / 9,
  frameRate: { ideal: 30, max: 30 }
};

const STREAM_FRAME_SIZE = { width: 960, height: 540 };
// How often a camera frame is sent for face recognition (~0.12 s per frame on the backend)
const FRAME_INTERVAL_MS = 300;
// Length of each audio chunk checked for "Hey Kinly". Shorter = Command Mode opens sooner; a phrase
// split between two chunks is still caught (the backend checks each chunk together with the previous one)
const HOTWORD_CHUNK_MS = 1500;
const CHAT_LIFETIME_MS = 10000; // keep in sync with the chat-fade-out delay in index.css
// Command mode focus area in % of the frame — keep in sync with CENTER_ZONE in backend/object_processor.py
const CENTER_ZONE_BOX = { x: 25, y: 15, w: 50, h: 70 };

// Saving an object: photos taken while the user slowly turns it
const CAPTURE_SIDES = 3;          // photos, one per side of the object
const CAPTURE_STEP_SECONDS = 2;   // countdown before each photo, so there's time to turn the object
const CAPTURE_SIDE_HINTS = [
  'Show the front of the object',
  'Turn it to another side',
  'Turn it once more',
];
const API_BASE = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000';
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
// Categories that get Dosage and Schedule fields in the save panel
const DOSAGE_CATEGORIES = ['medicine', 'supplement'];
const REGISTRATION_FRAME_SIZE = { width: 1280, height: 720 };

function App() {
  const [isMuted, setIsMuted] = useState(true);
  const [isAssisting, setIsAssisting] = useState(true);
  const [commandMode, setCommandMode] = useState(false);
  const [trackedFaces, setTrackedFaces] = useState([]);
  const [chatLogs, setChatLogs] = useState([]);
  const [yoloDebug, setYoloDebug] = useState(false);
  // Automatic conversation topics (Settings) — remembered in this browser, on by default
  const [autoTopics, setAutoTopics] = useState(() => {
    try {
      return localStorage.getItem('kinly.autoTopics') !== 'off';
    } catch {
      return true;
    }
  });
  const autoTopicsRef = useRef(autoTopics);  // read by the WebSocket handlers, which outlive renders

  useEffect(() => {
    autoTopicsRef.current = autoTopics;
    try {
      localStorage.setItem('kinly.autoTopics', autoTopics ? 'on' : 'off');
    } catch {
      // storage unavailable (private mode) — the setting just isn't remembered
    }
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'AUTO_TOPICS', enabled: autoTopics }));
    }
  }, [autoTopics]);

  // Right Panels State
  const [isAddingPerson, setIsAddingPerson] = useState(false);
  const [addPersonSnapshot, setAddPersonSnapshot] = useState(null);
  const [addPersonFaceBox, setAddPersonFaceBox] = useState(null);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isDatabaseOpen, setIsDatabaseOpen] = useState(false);

  // Object recognition state
  const [recognizedObjects, setRecognizedObjects] = useState([]);
  const [objectConfirmData, setObjectConfirmData] = useState(null);
  const [captureStep, setCaptureStep] = useState(null);  // {side, seconds, shots} while photographing an object
  const [captureFlash, setCaptureFlash] = useState(0);   // bumped on every photo to replay the flash
  const captureRunningRef = useRef(false);
  const captureAbortRef = useRef(false);
  const recognizedObjectsRef = useRef([]);  // latest object boxes, readable inside the async capture loop

  const webcamRef = useRef(null);
  const wsRef = useRef(null);
  const commandRecorderRef = useRef(null);
  const commandChunksRef = useRef([]);
  const cancelCommandRef = useRef(false);
  const objectTimeoutRef = useRef(null);  // Staleness timer for object overlays
  const chatIdRef = useRef(0);

  // Add a message to the left-hand log; it fades out and is removed after CHAT_LIFETIME_MS
  const addChatLog = (role, content) => {
    const id = ++chatIdRef.current;
    setChatLogs(prev => [
      // Only one surroundings description at a time — a new one replaces the old
      ...(role === 'ambient' ? prev.filter(log => log.role !== 'ambient') : prev).slice(-9),
      { id, role, content }
    ]);
    setTimeout(() => setChatLogs(prev => prev.filter(log => log.id !== id)), CHAT_LIFETIME_MS);
  };

  const unmountedRef = useRef(false);

  useEffect(() => {
    // Basic WebSocket setup to talk to the FastAPI backend
    unmountedRef.current = false;
    connectWebSocket();
    return () => {
      unmountedRef.current = true;
      if (wsRef.current) wsRef.current.close();
      if (commandRecorderRef.current && commandRecorderRef.current.state !== 'inactive') {
        commandRecorderRef.current.stop();
      }
    };
  }, []);

  const connectWebSocket = () => {
    const wsBase = import.meta.env.VITE_WS_URL || 'ws://localhost:8000';
    const ws = new WebSocket(`${wsBase}/ws/stream`);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('Connected to companion backend');
      // The backend keeps settings per connection — send the saved choice on every (re)connect
      ws.send(JSON.stringify({ type: 'AUTO_TOPICS', enabled: autoTopicsRef.current }));
    };

    ws.onerror = (e) => {
      console.error('WebSocket error', e);
    };

    ws.onmessage = (event) => {
      // Ignore anything still arriving on a replaced connection
      if (wsRef.current !== ws) return;
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'STATE_CHANGE') {
          setCommandMode(data.mode === 'COMMAND');
        } else if (data.type === 'FACES') {
          if (data.faces && data.faces.length > 0) {
            setTrackedFaces(data.faces.slice(0, 4));
          } else {
            setTrackedFaces([]);
          }
        } else if (data.type === 'AUDIO_RESPONSE') {
          // Playback audio TTS response (either mp3 from edge-tts or legacy webm)
          const b64Data = data.audio_b64 || data.data;
          const mimeType = data.audio_b64 ? 'audio/mp3' : 'audio/webm';
          const audioMsg = new Audio(`data:${mimeType};base64,${b64Data}`);
          audioMsg.play();
        } else if (data.type === 'CHAT') {
          addChatLog(data.role, data.content);
        } else if (data.type === 'OBJECTS') {
          // Clear any previous staleness timer
          if (objectTimeoutRef.current) clearTimeout(objectTimeoutRef.current);
          setRecognizedObjects(data.objects || []);
          recognizedObjectsRef.current = data.objects || [];
          // Auto-clear overlays if no update arrives within 6 seconds
          objectTimeoutRef.current = setTimeout(() => setRecognizedObjects([]), 6000);
        } else if (data.type === 'OBJECT_CONFIRM') {
          // Voice command "remember this ..." — photograph it the same way as the "+ Save" button
          captureObject(data.object_box, {
            suggested_name: data.suggested_name,
            note_text: data.note_text,
            detector_class: data.detector_class,
          });
        }
      } catch (e) {
        console.error('Failed to parse WS message', e);
      }
    };

    ws.onclose = () => {
      // Only the current connection reconnects — otherwise every closed socket spawns a new one and they pile up
      if (unmountedRef.current || wsRef.current !== ws) return;
      console.log('WS closed, retrying in 3s');
      setTimeout(() => {
        if (!unmountedRef.current && wsRef.current === ws) connectWebSocket();
      }, 3000);
    }
  };

  // Video Streaming Loop
  useEffect(() => {
    const frameInterval = setInterval(() => {
      if (webcamRef.current && wsRef.current && wsRef.current.readyState === WebSocket.OPEN && isAssistingRef.current) {
        const imageSrc = webcamRef.current.getScreenshot(STREAM_FRAME_SIZE);
        if (imageSrc) {
          wsRef.current.send(JSON.stringify({
            type: 'FRAME',
            data: imageSrc.split(',')[1] // send base64 chunk without data URI prefix
          }));
        }
      }
    }, FRAME_INTERVAL_MS); // backend always processes only the newest frame, so this never builds a backlog

    return () => clearInterval(frameInterval);
  }, []);

  const handleAddPerson = (face) => {
    if (!webcamRef.current) return;
    const imageSrc = webcamRef.current.getScreenshot(REGISTRATION_FRAME_SIZE);
    if (!imageSrc) return;
    setAddPersonSnapshot(imageSrc);
    setAddPersonFaceBox(face ? { x: face.x, y: face.y, w: face.w, h: face.h } : null);
    setIsAddingPerson(true);
    setIsSettingsOpen(false);
    setIsDatabaseOpen(false);
  };

  // Save an object: guide the user through photographing CAPTURE_SIDES sides of it (a countdown per
  // side), then let the backend crop it in every photo and read its label, and open the save panel.
  // `box` is where the object is (% of the frame); `extras` pre-fills the panel (voice command).
  const captureObject = async (box, extras = {}) => {
    if (!webcamRef.current || captureRunningRef.current) return;
    captureRunningRef.current = true;
    captureAbortRef.current = false;

    // Stop the command recording without sending it to Gemini
    cancelCommandRef.current = true;
    if (commandRecorderRef.current && commandRecorderRef.current.state !== 'inactive') {
      commandRecorderRef.current.stop();
    }
    setIsSettingsOpen(false);
    setIsDatabaseOpen(false);
    setIsAddingPerson(false);

    const frames = [];
    const hintBoxes = [];
    for (let side = 1; side <= CAPTURE_SIDES; side++) {
      for (let seconds = CAPTURE_STEP_SECONDS; seconds > 0; seconds--) {
        setCaptureStep({ side, seconds, shots: side - 1 });
        await sleep(1000);
        if (captureAbortRef.current) break;
      }
      if (captureAbortRef.current) break;
      const shot = webcamRef.current && webcamRef.current.getScreenshot(REGISTRATION_FRAME_SIZE);
      if (shot) {
        frames.push(shot.split(',')[1]);
        // Where the object is now — it moves when turned, so each photo gets its own box
        const target = recognizedObjectsRef.current.find(o => o.targeted);
        hintBoxes.push(target ? { x: target.x, y: target.y, w: target.w, h: target.h } : box);
        setCaptureFlash(n => n + 1);
      }
    }
    setCaptureStep(null);
    if (captureAbortRef.current) {  // "Cancel command" pressed during the countdown
      captureRunningRef.current = false;
      return;
    }

    // Back to ambient mode while the user fills in the panel
    setCommandMode(false);
    setRecognizedObjects([]);
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'COMMAND', command: 'END_COMMAND' }));
    }

    const panel = {
      suggested_name: extras.suggested_name || '',
      note_text: extras.note_text || '',
      detector_class: extras.detector_class || '',
    };
    setObjectConfirmData({ ...panel, loading: true });
    try {
      const response = await fetch(`${API_BASE}/api/objects/prepare`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ frames, hint_box: box, hint_boxes: hintBoxes }),
      });
      const data = await response.json();
      if (data.success) {
        const label = data.label || {};
        setObjectConfirmData({
          ...panel,
          loading: false,
          shots: data.shots,
          label,
          // A dosage printed on the package means it's almost certainly medicine
          category: label.dosage ? 'medicine' : '',
        });
      } else {
        setObjectConfirmData(null);
        addChatLog('system', data.error || 'Could not find the object in the photos');
      }
    } catch {
      setObjectConfirmData(null);
      addChatLog('system', 'Could not analyze the photos — is the backend running?');
    } finally {
      captureRunningRef.current = false;
    }
  };

  const isMutedRef = useRef(isMuted);
  const commandModeRef = useRef(commandMode);
  const isAssistingRef = useRef(isAssisting);

  useEffect(() => {
    isMutedRef.current = isMuted;
  }, [isMuted]);

  useEffect(() => {
    commandModeRef.current = commandMode;
  }, [commandMode]);

  useEffect(() => {
    isAssistingRef.current = isAssisting;
  }, [isAssisting]);

  // Audio Streaming Loop
  useEffect(() => {
    let stream;
    let recordingInterval;

    const initAudio = async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({ audio: true });

        // Analyser for Glow Effect
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const analyser = audioContext.createAnalyser();
        analyser.fftSize = 256;
        const source = audioContext.createMediaStreamSource(stream);
        source.connect(analyser);
        const dataArray = new Uint8Array(analyser.frequencyBinCount);

        const updateGlow = () => {
          if (commandModeRef.current && isAssistingRef.current && !isMutedRef.current) {
            analyser.getByteFrequencyData(dataArray);
            let sum = 0;
            for (let i = 0; i < dataArray.length; i++) sum += dataArray[i];
            const average = sum / dataArray.length;
            const glow = 20 + (average * 1.5); // base 20px, scales up with volume
            document.documentElement.style.setProperty('--glow-intensity', `${glow}px`);
          } else {
            document.documentElement.style.setProperty('--glow-intensity', `20px`);
          }
          requestAnimationFrame(updateGlow);
        };
        updateGlow();

        const startRecordingChunk = () => {
          if (!stream) return;

          const currentMuted = isMutedRef.current;
          const currentAssisting = isAssistingRef.current;
          const currentCmd = commandModeRef.current;

          if (!currentAssisting) return;
          if (currentMuted) return;

          stream.getAudioTracks().forEach(track => {
            track.enabled = true;
          });

          // COMMAND MODE: Continuous recording until button pressed
          if (currentCmd) {
            cancelCommandRef.current = false; // reset cancel flag when starting
            if (!commandRecorderRef.current || commandRecorderRef.current.state === 'inactive') {
              commandChunksRef.current = [];
              commandRecorderRef.current = new MediaRecorder(stream, { mimeType: 'audio/webm' });

              commandRecorderRef.current.ondataavailable = (event) => {
                if (event.data.size > 0) commandChunksRef.current.push(event.data);
              };

              commandRecorderRef.current.onstop = () => {
                if (!cancelCommandRef.current && commandChunksRef.current.length > 0 && wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
                  const blob = new Blob(commandChunksRef.current, { type: 'audio/webm' });
                  const reader = new FileReader();
                  reader.readAsDataURL(blob);
                  reader.onloadend = () => {
                    const base64Audio = reader.result.split(',')[1];
                    wsRef.current.send(JSON.stringify({ type: 'COMMAND_AUDIO_FULL', data: base64Audio }));
                  };
                }
              };

              commandRecorderRef.current.start();
            }
            return; // Skip the ambient chunk logic
          } else {
            if (commandRecorderRef.current && commandRecorderRef.current.state !== 'inactive') {
              commandRecorderRef.current.stop();
            }
          }

          // AMBIENT MODE: short chunks (HOTWORD_CHUNK_MS) for "Hey Kinly" detection
          const mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
          const chunks = [];

          mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) chunks.push(event.data);
          };

          mediaRecorder.onstop = () => {
            if (chunks.length > 0 && wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
              const blob = new Blob(chunks, { type: 'audio/webm' });
              const reader = new FileReader();
              reader.readAsDataURL(blob);
              reader.onloadend = () => {
                const base64Audio = reader.result.split(',')[1];
                wsRef.current.send(JSON.stringify({ type: 'AUDIO', data: base64Audio }));
              };
            }
          };

          mediaRecorder.start();
          setTimeout(() => {
            if (mediaRecorder.state !== 'inactive') mediaRecorder.stop();
          }, HOTWORD_CHUNK_MS);
        };

        recordingInterval = setInterval(startRecordingChunk, HOTWORD_CHUNK_MS);
      } catch (e) {
        console.error("Audio capture failed", e);
      }
    };

    initAudio();

    return () => {
      if (recordingInterval) clearInterval(recordingInterval);
      if (stream) {
        stream.getTracks().forEach(t => t.stop());
      }
    };
  }, []);

  return (
    <div className={`app-container ${commandMode ? 'command-mode' : ''}`}>
      {commandMode && <div className="glow-container"></div>}
      <KinlyLogo />
      <TopBar
        onOpenSettings={() => { setIsSettingsOpen(true); setIsDatabaseOpen(false); setIsAddingPerson(false); }}
        onOpenDatabase={() => { setIsDatabaseOpen(true); setIsSettingsOpen(false); setIsAddingPerson(false); }}
      />

      {commandMode && (
        <button
          className="end-command-btn cancel-command-btn"
          onClick={(e) => {
            e.stopPropagation(); // prevent clicking the webcam
            setCommandMode(false);
            cancelCommandRef.current = true;
            captureAbortRef.current = true;  // also stops photographing an object, if that's running
            if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
              wsRef.current.send(JSON.stringify({ type: 'COMMAND', command: 'END_COMMAND' }));
            }
            if (commandRecorderRef.current && commandRecorderRef.current.state !== 'inactive') {
              commandRecorderRef.current.stop();
            }
          }}
        >
          Cancel command
        </button>
      )}

      {commandMode && captureStep === null && <div className="click-to-send-text">Click screen to send</div>}

      <div
        className="webcam-wrapper"
        style={{ cursor: commandMode && captureStep === null ? 'pointer' : 'default' }}
        onClick={() => {
          if (commandMode && !captureRunningRef.current) {
            setCommandMode(false);
            if (commandRecorderRef.current && commandRecorderRef.current.state !== 'inactive') {
              commandRecorderRef.current.stop();  // onstop sends the command; backend returns to AMBIENT after it
            } else if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
              // Nothing was recorded (e.g. mic muted) — tell the backend, or it stays stuck in COMMAND mode
              wsRef.current.send(JSON.stringify({ type: 'COMMAND', command: 'END_COMMAND' }));
            }
          }
        }}
      >
        <Webcam
          ref={webcamRef}
          audio={false}
          className="video-feed"
          screenshotFormat="image/jpeg"
          screenshotQuality={0.92}
          videoConstraints={CAMERA_CONSTRAINTS}
        />
      </div>

      <div className="overlay-layer">

        {/* Command mode only recognizes what's inside this frame (CENTER_ZONE in object_processor.py) */}
        {commandMode && (
          <div className="center-zone">
            {/* Fallback when the detector can't find the object: save whatever is inside the frame */}
            {!recognizedObjects.some(o => o.targeted) && captureStep === null && (
              <button
                className="tag-add-btn center-zone-save"
                onClick={(e) => { e.stopPropagation(); captureObject(CENTER_ZONE_BOX); }}
              >
                + Save what's in the frame
              </button>
            )}
          </div>
        )}

        {captureStep !== null && (
          <div className="capture-overlay">
            <div className="capture-step">Side {captureStep.side} of {CAPTURE_SIDES}</div>
            <div className="capture-hint">{CAPTURE_SIDE_HINTS[captureStep.side - 1]}</div>
            <div className="capture-count">{captureStep.seconds}</div>
            <div className="capture-dots">
              {Array.from({ length: CAPTURE_SIDES }, (_, i) => (
                <span key={i} className={`capture-dot ${i < captureStep.shots ? 'done' : ''}`}></span>
              ))}
            </div>
          </div>
        )}

        {/* Brief white flash each time a photo is taken (key change replays the animation) */}
        {captureFlash > 0 && <div key={captureFlash} className="capture-flash"></div>}

        {/* Chat / Logs Window Overlay */}
        <div className="chat-window">
          {chatLogs.map((log) => (
            <div key={log.id} className={`chat-bubble ${log.role}`}>
              {log.content}
            </div>
          ))}
        </div>

        {/* Right Panels Overlay */}
        {isSettingsOpen && (
          <div className="right-panel">
            <h3>Settings</h3>
            <p>Configure application preferences, UI tokens, and API endpoints.</p>
            <div style={{ marginTop: '20px', marginBottom: '20px', display: 'flex', flexDirection: 'column', gap: '14px' }}>
              <label style={{ display: 'flex', alignItems: 'center', gap: '10px', cursor: 'pointer' }}>
                <input
                  type="checkbox"
                  checked={autoTopics}
                  onChange={(e) => setAutoTopics(e.target.checked)}
                />
                Save conversation topics automatically (audio of conversations with known people is summarized by Gemini)
              </label>
              <label style={{ display: 'flex', alignItems: 'center', gap: '10px', cursor: 'pointer' }}>
                <input
                  type="checkbox"
                  checked={yoloDebug}
                  onChange={(e) => {
                    const enabled = e.target.checked;
                    setYoloDebug(enabled);
                    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
                      wsRef.current.send(JSON.stringify({ type: 'YOLO_DEBUG', enabled }));
                    }
                  }} 
                />
                Show YOLO Debug Overlays (0.2x Conf, 0.1s Refresh)
              </label>
            </div>
            <button className="panel-close-btn" onClick={() => setIsSettingsOpen(false)}>Close Modules</button>
          </div>
        )}

        {isDatabaseOpen && (
          <DatabasePanel onClose={() => setIsDatabaseOpen(false)} />
        )}

        {isAddingPerson && addPersonSnapshot && (
          <div className="right-panel">
            <h3>Add New Person</h3>
            <p>Assign a name and relationship to the face currently tracked.</p>
            <div className="snapshot-preview" style={{ backgroundImage: `url(${addPersonSnapshot})` }}></div>
            <form onSubmit={async (e) => {
              e.preventDefault();
              const name = e.target.name.value;
              const relation = e.target.relation.value;
              if (!name || !relation) return;

              try {
                const apiBase = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000';
                const response = await fetch(`${apiBase}/api/faces`, {
                  method: "POST",
                  headers: { "Content-Type": "application/json" },
                  body: JSON.stringify({
                    name,
                    relation,
                    frame_b64: addPersonSnapshot.split(',')[1],
                    face_box: addPersonFaceBox
                  })
                });
                const data = await response.json();
                if (data.success) {
                  addChatLog("system", data.message);
                  setIsAddingPerson(false);
                  setAddPersonSnapshot(null);
                  setAddPersonFaceBox(null);
                } else {
                  alert("Error: " + data.error);
                }
              } catch (err) {
                alert("Failed to add face");
              }
            }}>
              <label>Name</label>
              <input type="text" name="name" required placeholder="e.g. John Doe" />
              <label>Relationship</label>
              <input type="text" name="relation" required placeholder="e.g. Brother" />
              <div className="panel-actions">
                <button type="submit" className="submit-btn btn-primary">Save Identity</button>
                <button type="button" className="cancel-btn btn-secondary" onClick={() => {
                  setIsAddingPerson(false);
                  setAddPersonSnapshot(null);
                  setAddPersonFaceBox(null);
                }}>Cancel</button>
              </div>
            </form>
          </div>
        )}

        {objectConfirmData && objectConfirmData.loading && (
          <div className="right-panel">
            <h3>Confirm Object</h3>
            <p>Analyzing the photos and reading the label…</p>
          </div>
        )}

        {objectConfirmData && !objectConfirmData.loading && (
          <div className="right-panel">
            <h3>Confirm Object</h3>
            <p>{objectConfirmData.shots.length} photo(s) — remove any that don't show the object.</p>
            <div className="shot-grid">
              {objectConfirmData.shots.map((shot, i) => (
                <div key={i} className="shot-thumb" style={{ backgroundImage: `url(data:image/jpeg;base64,${shot.crop_b64})` }}>
                  {objectConfirmData.shots.length > 1 && (
                    <button
                      type="button"
                      className="shot-remove"
                      title="Remove this photo"
                      onClick={() => setObjectConfirmData(prev => ({ ...prev, shots: prev.shots.filter((_, j) => j !== i) }))}
                    >
                      ×
                    </button>
                  )}
                </div>
              ))}
            </div>
            <form onSubmit={async (e) => {
              e.preventDefault();
              const name = e.target.obj_name.value;
              const notes = e.target.obj_notes.value;
              const category = e.target.obj_category.value;
              // These fields only exist for medicine/supplements and objects with readable text
              const dosage = e.target.obj_dosage ? e.target.obj_dosage.value : '';
              const schedule = e.target.obj_schedule ? e.target.obj_schedule.value : '';
              const labelText = e.target.obj_label ? e.target.obj_label.value.trim() : '';
              if (!name) return;
              const label = objectConfirmData.label || {};
              // If the user cleared the label text, don't use the label to recognize this object
              const keepLabel = labelText !== '';

              try {
                const response = await fetch(`${API_BASE}/api/objects`, {
                  method: "POST",
                  headers: { "Content-Type": "application/json" },
                  body: JSON.stringify({
                    name,
                    notes,
                    category,
                    dosage,
                    schedule,
                    detector_class: objectConfirmData.detector_class || '',
                    crops: objectConfirmData.shots.map(s => s.crop_b64),
                    label_text: labelText,
                    label_name: keepLabel ? (label.product_name || '') : '',
                    label_dosage: keepLabel ? (label.dosage || '') : '',
                  })
                });
                const data = await response.json();
                if (data.success) {
                  addChatLog("system", `Saved: ${name}${notes ? ` — ${notes}` : ""}`);
                  setObjectConfirmData(null);
                  // Signal backend to return to AMBIENT mode
                  if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
                    wsRef.current.send(JSON.stringify({ type: 'COMMAND', command: 'OBJECT_CONFIRM_DONE' }));
                  }
                } else {
                  alert("Error: " + data.error);
                }
              } catch (err) {
                alert("Failed to save object");
              }
            }}>
              {((objectConfirmData.label || {}).text || (objectConfirmData.label || {}).error) && (
                <>
                  <label>
                    {objectConfirmData.label.error
                      ? "Text on the object (couldn't read it — AI is busy, type it if there is any)"
                      : 'Text on the object (read automatically)'}
                  </label>
                  <input
                    type="text"
                    name="obj_label"
                    placeholder={objectConfirmData.label.error ? 'e.g. Ibuprom 200 mg' : 'Clear it to not use the text for recognition'}
                    defaultValue={objectConfirmData.label.text || ''}
                  />
                </>
              )}
              <label>Name</label>
              <input
                type="text"
                name="obj_name"
                required
                placeholder="e.g. Amoksiklav"
                defaultValue={objectConfirmData.suggested_name || (objectConfirmData.label || {}).product_name || ''}
              />
              <label>Note</label>
              <input type="text" name="obj_notes" placeholder="e.g. Take 3 times a day" defaultValue={objectConfirmData.note_text || ''} />
              <label>Category</label>
              <select
                name="obj_category"
                value={objectConfirmData.category || ''}
                onChange={(e) => {
                  const category = e.target.value;
                  setObjectConfirmData(prev => ({ ...prev, category }));
                }}
              >
                <option value="">— Select —</option>
                <option value="medicine">Medicine</option>
                <option value="supplement">Supplement</option>
                <option value="device">Device</option>
                <option value="food">Food</option>
                <option value="other">Other</option>
              </select>
              {DOSAGE_CATEGORIES.includes(objectConfirmData.category) && (
                <>
                  <label>Dosage</label>
                  <input type="text" name="obj_dosage" placeholder="e.g. 500mg" defaultValue={(objectConfirmData.label || {}).dosage || ''} />
                  <label>Schedule</label>
                  <input type="text" name="obj_schedule" placeholder="e.g. 3x daily after meals" />
                </>
              )}
              <div className="panel-actions">
                <button type="submit" className="submit-btn btn-primary">Save</button>
                <button type="button" className="cancel-btn btn-secondary" onClick={() => {
                  setObjectConfirmData(null);
                  // Signal backend to return to AMBIENT mode
                  if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
                    wsRef.current.send(JSON.stringify({ type: 'COMMAND', command: 'OBJECT_CONFIRM_DONE' }));
                  }
                }}>Cancel</button>
              </div>
            </form>
          </div>
        )}

        {trackedFaces.map((face, i) => {
          // Unique per tracked face (from the backend tracker) — duplicate keys leave stale tags on screen
          const stableKey = face.track_id ?? `${face.name}-${i}`;
          const isUnknown = face.name === "Unknown Person";
          return (
            <div
              key={stableKey}
              className="face-anchor"
              style={{
                top: `${face.y}%`,
                left: `${face.x}%`,
                width: `${face.w}%`,
                height: `${face.h}%`,
                opacity: 1
              }}
            >
              <div className="anchor-dot"></div>
              <svg preserveAspectRatio="none" className="connection-svg" viewBox="0 0 100 100">
                <line x1="0" y1="100" x2="100" y2="0" stroke="rgba(216, 243, 220, 0.6)" strokeWidth="2" vectorEffect="non-scaling-stroke" />
              </svg>
              <div className={`face-tooltip ${isUnknown ? 'unknown' : ''}`}>
                <div className="face-name">{face.name}</div>
                {face.relation && <div className="face-relation">{face.relation}</div>}
                {/* Nothing is shown until there has been a conversation with this person */}
                {face.topics && face.topics.length > 0 ? (
                  <div className="face-topics-list">
                    {face.topics.map((t, i) => (
                      <span key={i} className="face-topic-tag">{t}</span>
                    ))}
                  </div>
                ) : face.lastConversation ? (
                  <div className="face-topic">Last Topic: {face.lastConversation}</div>
                ) : null}
                {isUnknown && (
                  <button
                    className="tag-add-btn"
                    onClick={() => handleAddPerson(face)}
                    title="Register this face"
                  >
                    + Add
                  </button>
                )}
              </div>
            </div>
          );
        })}

        {recognizedObjects.map((obj, i) => (
          <div
            key={`obj-${obj.name}-${i}`}
            className="object-anchor"
            style={{
              top: `${obj.y}%`,
              left: `${obj.x}%`,
              width: `${obj.w}%`,
              height: `${obj.h}%`,
            }}
          >
            <div className="object-tooltip">
              <div className="object-name">{obj.name}</div>
              {obj.notes && <div className="object-notes">{obj.notes}</div>}
              {obj.dosage && <div className="object-meta">Dosage: {obj.dosage}</div>}
              {obj.targeted && captureStep === null && (
                <button
                  className="tag-add-btn"
                  onClick={(e) => {
                    e.stopPropagation();
                    captureObject({ x: obj.x, y: obj.y, w: obj.w, h: obj.h }, { detector_class: obj.name });
                  }}
                >
                  + Save
                </button>
              )}
            </div>
          </div>
        ))}
      </div>

      <CommandBar
        isMuted={isMuted}
        isAssisting={isAssisting}
        toggleMute={() => setIsMuted(!isMuted)}
        commandModeActive={commandMode}
        onActivateCommand={() => {
          setCommandMode(true);
          if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify({ type: 'COMMAND', command: 'ACTIVATE_COMMAND' }));
          }
        }}
        toggleAssisting={() => {
          if (isAssisting) {
            // Stop
            setCommandMode(false);
            setTrackedFaces([]);
            if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
              wsRef.current.send(JSON.stringify({ type: 'COMMAND', command: 'STOP' }));
            }
          } else {
            // Start — backend ambient loop restarts automatically via STOP handler
            if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
              wsRef.current.send(JSON.stringify({ type: 'COMMAND', command: 'START' }));
            }
          }
          setIsAssisting(!isAssisting);
        }}
      />
    </div>
  );
}

export default App;
