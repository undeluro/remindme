import React, { useState, useEffect, useRef } from 'react';
import Webcam from 'react-webcam';
import { CommandBar, TopBar } from './components/CommandBar';
import DatabasePanel from './components/DatabasePanel';
import './index.css';

const CAMERA_CONSTRAINTS = {
  facingMode: 'user',
  width: { ideal: 1280 },
  height: { ideal: 720 },
  aspectRatio: 16 / 9,
  frameRate: { ideal: 30, max: 30 }
};

const STREAM_FRAME_SIZE = { width: 960, height: 540 };
const REGISTRATION_FRAME_SIZE = { width: 1280, height: 720 };

function App() {
  const [isMuted, setIsMuted] = useState(true);
  const [isAssisting, setIsAssisting] = useState(true);
  const [commandMode, setCommandMode] = useState(false);
  const [trackedFaces, setTrackedFaces] = useState([]);
  const [chatLogs, setChatLogs] = useState([]);
  const [yoloDebug, setYoloDebug] = useState(false);

  // Right Panels State
  const [isAddingPerson, setIsAddingPerson] = useState(false);
  const [addPersonSnapshot, setAddPersonSnapshot] = useState(null);
  const [addPersonFaceBox, setAddPersonFaceBox] = useState(null);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isDatabaseOpen, setIsDatabaseOpen] = useState(false);

  // Object recognition state
  const [recognizedObjects, setRecognizedObjects] = useState([]);
  const [objectConfirmData, setObjectConfirmData] = useState(null);

  const webcamRef = useRef(null);
  const wsRef = useRef(null);
  const commandRecorderRef = useRef(null);
  const commandChunksRef = useRef([]);
  const cancelCommandRef = useRef(false);
  const objectTimeoutRef = useRef(null);  // Staleness timer for object overlays

  useEffect(() => {
    // Basic WebSocket setup to talk to the FastAPI backend
    connectWebSocket();
    return () => {
      if (wsRef.current) wsRef.current.close();
      if (commandRecorderRef.current && commandRecorderRef.current.state !== 'inactive') {
        commandRecorderRef.current.stop();
      }
    };
  }, []);

  const connectWebSocket = () => {
    const wsBase = import.meta.env.VITE_WS_URL || 'ws://localhost:8000';
    wsRef.current = new WebSocket(`${wsBase}/ws/stream`);

    wsRef.current.onopen = () => {
      console.log('Connected to companion backend');
    };

    wsRef.current.onerror = (e) => {
      console.error('WebSocket error', e);
    };

    wsRef.current.onmessage = (event) => {
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
          setChatLogs(prev => [...prev.slice(-9), { role: data.role, content: data.content }]);
        } else if (data.type === 'OBJECTS') {
          // Clear any previous staleness timer
          if (objectTimeoutRef.current) clearTimeout(objectTimeoutRef.current);
          setRecognizedObjects(data.objects || []);
          // Auto-clear overlays if no update arrives within 6 seconds
          objectTimeoutRef.current = setTimeout(() => setRecognizedObjects([]), 6000);
        } else if (data.type === 'OBJECT_CONFIRM') {
          setObjectConfirmData(data);
        }
      } catch (e) {
        console.error('Failed to parse WS message', e);
      }
    };

    wsRef.current.onclose = () => {
      console.log('WS closed, retrying in 3s');
      setTimeout(connectWebSocket, 3000);
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
    }, 1500); // 2 FPS for smoother face tracking

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
            return; // Skip the ambient 2.5s chunk logic
          } else {
            if (commandRecorderRef.current && commandRecorderRef.current.state !== 'inactive') {
              commandRecorderRef.current.stop();
            }
          }

          // AMBIENT MODE: 2.5s chunking for hotword detection
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
          }, 2500);
        };

        recordingInterval = setInterval(startRecordingChunk, 2500);
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
      <div className="app-title">RemindMe</div>
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

      {commandMode && <div className="click-to-send-text">Click screen to send</div>}

      <div
        className="webcam-wrapper"
        style={{ cursor: commandMode ? 'pointer' : 'default' }}
        onClick={() => {
          if (commandMode) {
            setCommandMode(false);
            if (commandRecorderRef.current && commandRecorderRef.current.state !== 'inactive') {
              commandRecorderRef.current.stop();
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

        {/* Chat / Logs Window Overlay */}
        <div className="chat-window">
          {chatLogs.map((log, i) => (
            <div key={i} className={`chat-bubble ${log.role}`}>
              {log.content}
            </div>
          ))}
        </div>

        {/* Right Panels Overlay */}
        {isSettingsOpen && (
          <div className="right-panel">
            <h3>Settings</h3>
            <p>Configure application preferences, UI tokens, and API endpoints.</p>
            <div style={{ marginTop: '20px', marginBottom: '20px' }}>
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
                  setChatLogs(prev => [...prev.slice(-9), { role: "system", content: data.message }]);
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

        {objectConfirmData && (
          <div className="right-panel">
            <h3>Confirm Object</h3>
            <p>Review the detected object before saving.</p>
            {objectConfirmData.crop_preview_b64 && (
              <div className="snapshot-preview" style={{ backgroundImage: `url(data:image/jpeg;base64,${objectConfirmData.crop_preview_b64})` }}></div>
            )}
            <form onSubmit={async (e) => {
              e.preventDefault();
              const name = e.target.obj_name.value;
              const notes = e.target.obj_notes.value;
              const category = e.target.obj_category.value;
              const dosage = e.target.obj_dosage.value;
              const schedule = e.target.obj_schedule.value;
              if (!name) return;

              try {
                const apiBase = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000';
                const response = await fetch(`${apiBase}/api/objects`, {
                  method: "POST",
                  headers: { "Content-Type": "application/json" },
                  body: JSON.stringify({
                    name,
                    notes,
                    category,
                    dosage,
                    schedule,
                    frame_b64: objectConfirmData.frame_b64,
                    object_box: objectConfirmData.object_box,
                  })
                });
                const data = await response.json();
                if (data.success) {
                  setChatLogs(prev => [...prev.slice(-9), { role: "system", content: `Zapamiętałem: ${name} — ${notes}` }]);
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
              <label>Name</label>
              <input type="text" name="obj_name" required placeholder="e.g. Amoksiklav" defaultValue={objectConfirmData.suggested_name || ''} />
              <label>Note</label>
              <input type="text" name="obj_notes" placeholder="e.g. Take 3 times a day" defaultValue={objectConfirmData.note_text || ''} />
              <label>Category</label>
              <select name="obj_category" defaultValue="">
                <option value="">— Select —</option>
                <option value="medicine">Medicine</option>
                <option value="supplement">Supplement</option>
                <option value="device">Device</option>
                <option value="food">Food</option>
                <option value="other">Other</option>
              </select>
              <label>Dosage</label>
              <input type="text" name="obj_dosage" placeholder="e.g. 500mg" />
              <label>Schedule</label>
              <input type="text" name="obj_schedule" placeholder="e.g. 3x daily after meals" />
              <div className="panel-actions">
                <button type="submit" className="submit-btn btn-primary">Zatwierdź</button>
                <button type="button" className="cancel-btn btn-secondary" onClick={() => {
                  setObjectConfirmData(null);
                  // Signal backend to return to AMBIENT mode
                  if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
                    wsRef.current.send(JSON.stringify({ type: 'COMMAND', command: 'OBJECT_CONFIRM_DONE' }));
                  }
                }}>Anuluj</button>
              </div>
            </form>
          </div>
        )}

        {trackedFaces.map((face) => {
          // Stable key: use name + spatial quadrant so React doesn't swap DOM nodes
          const quadrant = `${Math.round(face.x / 25)}-${Math.round(face.y / 25)}`;
          const stableKey = `${face.name}-${quadrant}`;
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
                <div className="face-relation">{face.relation}</div>
                {face.topics && face.topics.length > 0 ? (
                  <div className="face-topics-list">
                    {face.topics.map((t, i) => (
                      <span key={i} className="face-topic-tag">{t}</span>
                    ))}
                  </div>
                ) : (
                  <div className="face-topic">Last Topic: {face.lastConversation}</div>
                )}
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
