import React from 'react';
import { Mic, MicOff, StopCircle, PlayCircle, Settings, Database } from 'lucide-react';

export const CommandBar = ({
  isMuted,
  isAssisting,
  toggleMute,
  commandModeActive,
  toggleAssisting,
  onActivateCommand,
}) => {
  return (
    <div className="command-bar">
      <div className="command-bar-group">
        {isAssisting && !commandModeActive && (
          <button
            className="icon-btn"
            onClick={onActivateCommand}
            title="Activate Command Mode"
          >
            <svg width="24" height="24" viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg">
              <rect x="8" y="14" width="4" height="16" rx="2" fill="currentColor"/>
              <rect x="16" y="6" width="4" height="24" rx="2" fill="currentColor"/>
              <rect x="24" y="10" width="4" height="20" rx="2" fill="currentColor"/>
              <path d="M28 4 L29 7 L32 8 L29 9 L28 12 L27 9 L24 8 L27 7 Z" fill="currentColor"/>
            </svg>
          </button>
        )}

        <button 
          className="icon-text-btn"
          onClick={toggleAssisting}
          style={{ color: isAssisting ? '#d32f2f' : '#2e7d32', padding: '0 16px' }}
        >
          {isAssisting ? <StopCircle size={20} /> : <PlayCircle size={20} />}
          {isAssisting ? 'Stop Assisting' : 'Start Assisting'}
        </button>
        
        <button 
          className={`icon-btn ${isMuted ? 'muted' : 'active-mic'}`}
          onClick={toggleMute}
          title={isMuted ? "Unmute Mic" : "Mute Mic"}
        >
          {isMuted ? <MicOff size={24} /> : <Mic size={24} color="#f8f9fa" />}
        </button>
      </div>
    </div>
  );
};

export const TopBar = ({ onOpenSettings, onOpenDatabase }) => {
  return (
    <div className="top-bar">
      <button className="icon-btn" title="Memory/Logs" onClick={onOpenDatabase}>
        <Database size={20} />
      </button>
      <button className="icon-btn" title="Settings" onClick={onOpenSettings}>
        <Settings size={20} />
      </button>
    </div>
  );
};
