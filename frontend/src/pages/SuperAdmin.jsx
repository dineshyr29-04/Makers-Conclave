import { useState, useEffect, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import CityMap from '../components/CityMap'
import { EmergencyAlertCard, ViolationAlertCard } from '../components/AlertCard'
import { useWebSocket } from '../hooks/useWebSocket'
import api from '../services/api'
import './Dashboard.css'
import '../components/AlertCard.css'

export default function SuperAdminDashboard() {
  const navigate = useNavigate()
  const fullName = localStorage.getItem('full_name') || 'Admin'

  const [junctions, setJunctions] = useState([])
  const [emergencyAlerts, setEmergencyAlerts] = useState([])
  const [violationAlerts, setViolationAlerts] = useState([])
  const [summary, setSummary] = useState({ total_emergency_events: 0, total_violations: 0, pending_violations: 0, active_cameras: 0 })
  const [activeTab, setActiveTab] = useState('emergency')
  const [selectedCamera, setSelectedCamera] = useState('CAM_001')
  const [wsConnected, setWsConnected] = useState(false)

  // Load initial data
  useEffect(() => {
    api.get('/emergency/signals').then(r => setJunctions(r.data.junctions)).catch(() => {})
    api.get('/dashboard/summary').then(r => setSummary(r.data)).catch(() => {})
  }, [])

  // WebSocket handler
  const handleWsMessage = useCallback((msg) => {
    if (msg.type === 'emergency_alert') {
      setEmergencyAlerts(prev => [msg, ...prev].slice(0, 20))
      setSummary(s => ({ ...s, total_emergency_events: s.total_emergency_events + 1 }))
    } else if (msg.type === 'violation_alert') {
      setViolationAlerts(prev => [msg, ...prev].slice(0, 20))
      setSummary(s => ({ ...s, total_violations: s.total_violations + 1, pending_violations: s.pending_violations + 1 }))
    } else if (msg.type === 'signal_update') {
      setJunctions(msg.junctions || [])
    } else if (msg.type === 'history') {
      const emergency = msg.events?.filter(e => e.type === 'emergency_alert') || []
      const violations = msg.events?.filter(e => e.type === 'violation_alert') || []
      if (emergency.length) setEmergencyAlerts(emergency)
      if (violations.length) setViolationAlerts(violations)
    }
  }, [])

  const { connected } = useWebSocket('/api/emergency/ws/super_admin', handleWsMessage)
  useWebSocket('/api/violations/ws/super_admin', handleWsMessage)

  const handleLogout = () => {
    localStorage.clear()
    navigate('/login')
  }

  const preemptedCount = junctions.filter(j => j.is_preempted).length

  return (
    <div className="dashboard">
      {/* ── Sidebar ─────────────────────────────────────────── */}
      <aside className="sidebar">
        <div className="sidebar__logo">
          <svg width="28" height="28" viewBox="0 0 36 36" fill="none">
            <circle cx="18" cy="18" r="17" stroke="#3b82f6" strokeWidth="2" />
            <path d="M18 8 L26 14 L26 22 L18 28 L10 22 L10 14 Z" fill="rgba(59,130,246,0.2)" stroke="#3b82f6" strokeWidth="1.5" />
            <circle cx="18" cy="18" r="4" fill="#3b82f6" />
          </svg>
          <span className="sidebar__brand">CityAI</span>
        </div>

        <nav className="sidebar__nav">
          <div className="sidebar__section-label">Modules</div>
          <button className={`sidebar__item ${activeTab === 'emergency' ? 'active' : ''}`} onClick={() => setActiveTab('emergency')}>
            <span className="sidebar__icon">🚨</span>
            <span>Emergency</span>
            {emergencyAlerts.length > 0 && <span className="sidebar__badge">{emergencyAlerts.length}</span>}
          </button>
          <button className={`sidebar__item ${activeTab === 'violations' ? 'active' : ''}`} onClick={() => setActiveTab('violations')}>
            <span className="sidebar__icon">⚠️</span>
            <span>Violations</span>
            {violationAlerts.length > 0 && <span className="sidebar__badge sidebar__badge--warn">{violationAlerts.length}</span>}
          </button>
          <button className={`sidebar__item ${activeTab === 'cameras' ? 'active' : ''}`} onClick={() => setActiveTab('cameras')}>
            <span className="sidebar__icon">📹</span>
            <span>Live Feeds</span>
          </button>
          <button className={`sidebar__item ${activeTab === 'signals' ? 'active' : ''}`} onClick={() => setActiveTab('signals')}>
            <span className="sidebar__icon">🚦</span>
            <span>Signals</span>
          </button>
        </nav>

        <div className="sidebar__footer">
          <div className="sidebar__user">
            <div className="sidebar__avatar">
              {fullName.charAt(0).toUpperCase()}
            </div>
            <div>
              <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)' }}>{fullName}</div>
              <div style={{ fontSize: 11, color: 'var(--color-brand)' }}>Super Admin</div>
            </div>
          </div>
          <button className="btn btn-ghost" style={{ width: '100%', justifyContent: 'center', marginTop: 8 }} onClick={handleLogout}>
            Logout
          </button>
        </div>
      </aside>

      {/* ── Main Content ─────────────────────────────────────── */}
      <main className="dashboard__main">
        {/* Top bar */}
        <header className="dashboard__topbar">
          <div>
            <h1 className="dashboard__title">Command Center</h1>
            <p className="dashboard__subtitle">Bengaluru City · All Modules</p>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <div className="topbar__connection">
              <div className={`status-dot ${connected ? 'online' : 'alert'}`} />
              <span>{connected ? 'Live' : 'Reconnecting...'}</span>
            </div>
            <div className="topbar__time">{new Date().toLocaleTimeString()}</div>
          </div>
        </header>

        {/* Stats Row */}
        <div className="stats-row">
          <StatCard icon="🚨" label="Emergency Events" value={summary.total_emergency_events} color="var(--color-emergency)" />
          <StatCard icon="⚠️" label="Total Violations" value={summary.total_violations} color="var(--color-violation)" />
          <StatCard icon="⏳" label="Pending Review" value={summary.pending_violations} color="#a78bfa" />
          <StatCard icon="📹" label="Active Cameras" value={summary.active_cameras} color="var(--color-success)" />
          <StatCard icon="🚦" label="Preempted Signals" value={preemptedCount} color="var(--color-preempted)" />
        </div>

        {/* Map + Panel row */}
        <div className="dashboard__split">
          {/* Map */}
          <div className="dashboard__map card">
            <CityMap junctions={junctions} />
          </div>

          {/* Right Panel */}
          <aside className="dashboard__panel">
            <div className="panel__header">
              <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
                {activeTab === 'emergency' && '🚨 Emergency Alerts'}
                {activeTab === 'violations' && '⚠️ Civic Violations'}
                {activeTab === 'cameras' && '📹 Live Feeds'}
                {activeTab === 'signals' && '🚦 Signal Status'}
              </span>
              <span className="text-sm" style={{ color: 'var(--text-muted)' }}>
                {activeTab === 'emergency' && `${emergencyAlerts.length} alerts`}
                {activeTab === 'violations' && `${violationAlerts.length} violations`}
              </span>
            </div>

            <div className="panel__content overflow-y-auto">
              {activeTab === 'emergency' && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                  {emergencyAlerts.length === 0
                    ? <EmptyState icon="🚨" text="No emergency events. System is monitoring." />
                    : emergencyAlerts.map((e, i) => (
                        <EmergencyAlertCard key={i} event={e} />
                      ))}
                </div>
              )}

              {activeTab === 'violations' && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                  {violationAlerts.length === 0
                    ? <EmptyState icon="⚠️" text="No violations detected. System is monitoring." />
                    : violationAlerts.map((e, i) => (
                        <ViolationAlertCard key={i} event={e} />
                      ))}
                </div>
              )}

              {activeTab === 'cameras' && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                  {['CAM_001', 'CAM_002', 'CAM_WEBCAM'].map(camId => (
                    <div key={camId} className="card" style={{ padding: 10 }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                        <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)' }}>{camId}</span>
                        <span className="badge badge-success">Live</span>
                      </div>
                      <img
                        src={`/api/cameras/${camId}/stream`}
                        alt={`Camera ${camId}`}
                        style={{ width: '100%', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border)' }}
                        onError={(e) => { e.target.style.display = 'none' }}
                      />
                    </div>
                  ))}
                </div>
              )}

              {activeTab === 'signals' && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                  {junctions.map(j => (
                    <div key={j.id} className="card" style={{ padding: '12px 14px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <div>
                        <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)' }}>{j.name}</div>
                        <div style={{ fontSize: 11, color: 'var(--text-muted)', fontFamily: 'monospace' }}>{j.id}</div>
                      </div>
                      <SignalBadge state={j.state} isPreempted={j.is_preempted} />
                    </div>
                  ))}
                  {junctions.length === 0 && <EmptyState icon="🚦" text="Loading signal data..." />}
                </div>
              )}
            </div>
          </aside>
        </div>
      </main>
    </div>
  )
}

function StatCard({ icon, label, value, color }) {
  return (
    <div className="stat-card card">
      <div className="stat-card__icon" style={{ color }}>{icon}</div>
      <div className="stat-card__value" style={{ color }}>{value}</div>
      <div className="stat-card__label">{label}</div>
    </div>
  )
}

function SignalBadge({ state, isPreempted }) {
  const colors = {
    GREEN: 'var(--color-success)',
    RED: 'var(--color-emergency)',
    YELLOW: 'var(--color-violation)',
    PREEMPTED: 'var(--color-preempted)',
  }
  const color = colors[state] || 'var(--text-muted)'
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
      {isPreempted && <span style={{ fontSize: 11, color: 'var(--color-preempted)' }}>🚑 Cleared</span>}
      <div style={{
        width: 14, height: 14, borderRadius: '50%', background: color,
        boxShadow: `0 0 8px ${color}`,
      }} />
      <span style={{ fontSize: 12, fontWeight: 600, color }}>{state}</span>
    </div>
  )
}

function EmptyState({ icon, text }) {
  return (
    <div style={{ textAlign: 'center', padding: '40px 20px', color: 'var(--text-muted)' }}>
      <div style={{ fontSize: 36, marginBottom: 12 }}>{icon}</div>
      <div style={{ fontSize: 13 }}>{text}</div>
    </div>
  )
}
