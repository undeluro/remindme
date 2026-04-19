import React, { useState, useEffect } from 'react';

const API = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000';

export default function DatabasePanel({ onClose }) {
  const [tab, setTab] = useState('people'); // 'people' | 'notes' | 'objects' | 'topics' | 'routines' | 'reminders'
  const [faces, setFaces] = useState([]);
  const [notes, setNotes] = useState([]);
  const [objects, setObjects] = useState([]);
  const [allTopics, setAllTopics] = useState({});
  const [routines, setRoutines] = useState([]);
  const [reminders, setReminders] = useState([]);
  const [newNote, setNewNote] = useState('');
  const [newRoutine, setNewRoutine] = useState('');
  const [newReminderTime, setNewReminderTime] = useState('');
  const [newReminderTask, setNewReminderTask] = useState('');
  const [newTopicInputs, setNewTopicInputs] = useState({});
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchData();
  }, [tab]);

  const fetchData = async () => {
    setLoading(true);
    try {
      if (tab === 'people') {
        const res = await fetch(`${API}/api/faces`, { cache: 'no-store' });
        const data = await res.json();
        setFaces(data.faces || []);
      } else if (tab === 'notes') {
        const res = await fetch(`${API}/api/notes`, { cache: 'no-store' });
        const data = await res.json();
        setNotes(data.notes || []);
      } else if (tab === 'objects') {
        const res = await fetch(`${API}/api/objects`, { cache: 'no-store' });
        const data = await res.json();
        setObjects(data.objects || []);
      } else if (tab === 'topics') {
        const [topicsRes, facesRes] = await Promise.all([
          fetch(`${API}/api/topics`, { cache: 'no-store' }),
          fetch(`${API}/api/faces`, { cache: 'no-store' })
        ]);
        const topicsData = await topicsRes.json();
        const facesData = await facesRes.json();
        setAllTopics(topicsData.topics || {});
        setFaces(facesData.faces || []);
      } else if (tab === 'routines') {
        const res = await fetch(`${API}/api/routines`, { cache: 'no-store' });
        const data = await res.json();
        setRoutines(data.routines || []);
      } else if (tab === 'reminders') {
        const res = await fetch(`${API}/api/reminders`, { cache: 'no-store' });
        const data = await res.json();
        setReminders(data.reminders || []);
      }
    } catch (e) {
      console.error('Failed to fetch:', e);
    }
    setLoading(false);
  };

  const deleteFace = async (name) => {
    try {
      await fetch(`${API}/api/faces/${encodeURIComponent(name)}`, { method: 'DELETE' });
      setFaces(prev => prev.filter(f => f.name !== name));
    } catch (e) {
      console.error('Delete face error:', e);
    }
  };

  const addNote = async () => {
    if (!newNote.trim()) return;
    try {
      await fetch(`${API}/api/notes`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: newNote.trim() })
      });
      setNewNote('');
      fetchData();
    } catch (e) {
      console.error('Add note error:', e);
    }
  };

  const deleteNote = async (index) => {
    try {
      await fetch(`${API}/api/notes/${index}`, { method: 'DELETE' });
      fetchData();
    } catch (e) {
      console.error('Delete note error:', e);
    }
  };

  const deleteObject = async (name) => {
    try {
      await fetch(`${API}/api/objects/${encodeURIComponent(name)}`, { method: 'DELETE' });
      setObjects(prev => prev.filter(o => o.name !== name));
    } catch (e) {
      console.error('Delete object error:', e);
    }
  };

  const deleteTopic = async (personName, topicIndex) => {
    try {
      await fetch(`${API}/api/faces/${encodeURIComponent(personName)}/topics/${topicIndex}`, { method: 'DELETE' });
      const res = await fetch(`${API}/api/topics`, { cache: 'no-store' });
      const data = await res.json();
      setAllTopics(data.topics || {});
    } catch (e) {
      console.error('Delete topic error:', e);
    }
  };

  const addTopic = async (personName) => {
    const topic = (newTopicInputs[personName] || '').trim();
    if (!topic) return;
    try {
      await fetch(`${API}/api/faces/${encodeURIComponent(personName)}/topics`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ topic })
      });
      setNewTopicInputs(prev => ({ ...prev, [personName]: '' }));
      fetchData();
    } catch (e) {
      console.error('Add topic error:', e);
    }
  };


  const addRoutine = async () => {
    if (!newRoutine.trim()) return;
    try {
      await fetch(`${API}/api/routines`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: newRoutine.trim() })
      });
      setNewRoutine('');
      fetchData();
    } catch (e) {
      console.error('Add routine error:', e);
    }
  };

  const deleteRoutine = async (index) => {
    try {
      await fetch(`${API}/api/routines/${index}`, { method: 'DELETE' });
      fetchData();
    } catch (e) {
      console.error('Delete routine error:', e);
    }
  };

  const addReminder = async () => {
    if (!newReminderTime.trim() || !newReminderTask.trim()) return;
    try {
      await fetch(`${API}/api/reminders`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ time: newReminderTime.trim(), task: newReminderTask.trim() })
      });
      setNewReminderTime('');
      setNewReminderTask('');
      fetchData();
    } catch (e) {
      console.error('Add reminder error:', e);
    }
  };

  const deleteReminder = async (index) => {
    try {
      await fetch(`${API}/api/reminders/${index}`, { method: 'DELETE' });
      fetchData();
    } catch (e) {
      console.error('Delete reminder error:', e);
    }
  };

  return (
    <div className="right-panel">
      <h3>Memory Database</h3>

      <div className="db-tabs">
        <button
          className={`db-tab ${tab === 'people' ? 'active' : ''}`}
          onClick={() => setTab('people')}
        >
          People
        </button>
        <button
          className={`db-tab ${tab === 'topics' ? 'active' : ''}`}
          onClick={() => setTab('topics')}
        >
          Topics
        </button>
        <button
          className={`db-tab ${tab === 'notes' ? 'active' : ''}`}
          onClick={() => setTab('notes')}
        >
          Notes
        </button>
        <button
          className={`db-tab ${tab === 'objects' ? 'active' : ''}`}
          onClick={() => setTab('objects')}
        >
          Objects
        </button>
        <button
          className={`db-tab ${tab === 'routines' ? 'active' : ''}`}
          onClick={() => setTab('routines')}
        >
          Routines
        </button>
        <button
          className={`db-tab ${tab === 'reminders' ? 'active' : ''}`}
          onClick={() => setTab('reminders')}
        >
          Reminders
        </button>
      </div>

      <div className="db-content">
        {loading && <p className="db-empty">Loading...</p>}

        {!loading && tab === 'people' && (
          <>
            {faces.length === 0 ? (
              <p className="db-empty">No registered people yet.</p>
            ) : (
              <ul className="db-list">
                {faces.map((face) => (
                  <li key={face.name} className="db-item">
                    <div className="db-item-info">
                      <span className="db-item-name">
                        {face.name}
                      </span>
                      <span className="db-item-meta">{face.relation} · {face.vectors || 1} vectors</span>
                      {face.topics && face.topics.length > 0 && (
                        <div className="db-item-topics">
                          {face.topics.slice(-3).map((t, i) => (
                            <span key={i} className="topic-chip">{t}</span>
                          ))}
                        </div>
                      )}
                    </div>
                    <div className="db-item-actions">
                      <button
                        className="db-delete-btn"
                        onClick={() => deleteFace(face.name)}
                        title="Remove person"
                      >
                        ✕
                      </button>
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </>
        )}

        {!loading && tab === 'notes' && (
          <>
            <div className="db-add-note">
              <input
                type="text"
                value={newNote}
                onChange={(e) => setNewNote(e.target.value)}
                placeholder="Add a note..."
                onKeyDown={(e) => e.key === 'Enter' && addNote()}
              />
              <button className="db-add-btn" onClick={addNote}>+</button>
            </div>
            {notes.length === 0 ? (
              <p className="db-empty">No notes yet.</p>
            ) : (
              <ul className="db-list">
                {notes.map((note, i) => (
                  <li key={i} className="db-item">
                    <div className="db-item-info">
                      <span className="db-item-note">{note}</span>
                    </div>
                    <button
                      className="db-delete-btn"
                      onClick={() => deleteNote(i)}
                      title="Delete note"
                    >
                      ✕
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </>
        )}

        {!loading && tab === 'objects' && (
          <>
            {objects.length === 0 ? (
              <p className="db-empty">No registered objects yet.</p>
            ) : (
              <ul className="db-list">
                {objects.map((obj) => (
                  <li key={obj.name} className="db-item">
                    <div className="db-item-info">
                      <span className="db-item-name">{obj.name}</span>
                      <span className="db-item-meta">
                        {obj.category && `${obj.category} · `}{obj.vectors || 0} vectors
                      </span>
                      {obj.notes && <span className="db-item-note">{obj.notes}</span>}
                      {obj.dosage && <span className="db-item-meta">Dosage: {obj.dosage}</span>}
                    </div>
                    <button
                      className="db-delete-btn"
                      onClick={() => deleteObject(obj.name)}
                      title="Remove object"
                    >
                      ✕
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </>
        )}

        {!loading && tab === 'topics' && (
          <>
            {faces.length === 0 ? (
              <p className="db-empty">Register people first to add topics.</p>
            ) : (
              <ul className="db-list">
                {faces.map((face) => {
                  const data = allTopics[face.name] || {};
                  return (
                    <li key={face.name} className="db-item db-item-topics-section">
                      <div className="db-item-info" style={{ width: '100%' }}>
                        <span className="db-item-name">{face.name}</span>
                        {data.topics && data.topics.length > 0 ? (
                          <div className="db-topics-grid">
                            {data.topics.map((topic, i) => (
                              <div key={i} className="topic-chip-deletable">
                                <span>{topic}</span>
                                <button
                                  className="topic-delete-btn"
                                  onClick={() => deleteTopic(face.name, i)}
                                  title="Delete topic"
                                >
                                  ✕
                                </button>
                              </div>
                            ))}
                          </div>
                        ) : (
                          <span className="db-item-meta">No topics yet</span>
                        )}
                        <div className="db-add-note" style={{ marginTop: '6px' }}>
                          <input
                            type="text"
                            value={newTopicInputs[face.name] || ''}
                            onChange={(e) => setNewTopicInputs(prev => ({ ...prev, [face.name]: e.target.value }))}
                            placeholder="Add a topic..."
                            onKeyDown={(e) => e.key === 'Enter' && addTopic(face.name)}
                          />
                          <button className="db-add-btn" onClick={() => addTopic(face.name)}>+</button>
                        </div>
                      </div>
                    </li>
                  );
                })}
              </ul>
            )}
          </>
        )}

        {!loading && tab === 'routines' && (
          <>
            <div className="db-add-note">
              <input
                type="text"
                value={newRoutine}
                onChange={(e) => setNewRoutine(e.target.value)}
                placeholder="Add a routine..."
                onKeyDown={(e) => e.key === 'Enter' && addRoutine()}
              />
              <button className="db-add-btn" onClick={addRoutine}>+</button>
            </div>
            {routines.length === 0 ? (
              <p className="db-empty">No routines yet.</p>
            ) : (
              <ul className="db-list">
                {routines.map((routine, i) => (
                  <li key={i} className="db-item">
                    <div className="db-item-info">
                      <span className="db-item-note">{routine}</span>
                    </div>
                    <button
                      className="db-delete-btn"
                      onClick={() => deleteRoutine(i)}
                      title="Delete routine"
                    >
                      ✕
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </>
        )}

        {!loading && tab === 'reminders' && (
          <>
            <div className="db-add-note">
              <input
                type="text"
                value={newReminderTime}
                onChange={(e) => setNewReminderTime(e.target.value)}
                placeholder="Time (e.g. 10:00 AM)"
                style={{ width: '100px', marginRight: '5px' }}
                onKeyDown={(e) => e.key === 'Enter' && addReminder()}
              />
              <input
                type="text"
                value={newReminderTask}
                onChange={(e) => setNewReminderTask(e.target.value)}
                placeholder="Task description..."
                style={{ flex: 1 }}
                onKeyDown={(e) => e.key === 'Enter' && addReminder()}
              />
              <button className="db-add-btn" onClick={addReminder}>+</button>
            </div>
            {reminders.length === 0 ? (
              <p className="db-empty">No reminders yet.</p>
            ) : (
              <ul className="db-list">
                {reminders.map((reminder, i) => (
                  <li key={i} className="db-item">
                    <div className="db-item-info">
                      <span className="db-item-name">{reminder.task}</span>
                      <span className="db-item-meta">{reminder.time}</span>
                    </div>
                    <button
                      className="db-delete-btn"
                      onClick={() => deleteReminder(i)}
                      title="Delete reminder"
                    >
                      ✕
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </>
        )}
      </div>

      <button className="panel-close-btn" onClick={onClose}>Close</button>
    </div>
  );
}
