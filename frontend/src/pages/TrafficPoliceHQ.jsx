import { useState, useEffect, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import CityMap from '../components/CityMap'
import { EmergencyAlertCard } from '../components/AlertCard'
import { useWebSocket } from '../hooks/useWebSocket'
import api from '../services/api'
import './Dashboard.css'
import '../components/AlertCard.css'

export default function TrafficPoliceDashboard() {
  const navigate = useNavigate()
  const fullName = localStorage.getItem('full_name') || 'Traffic HQ'
  const [junctions, setJunctions] = useState([])
  const [alerts, setAlerts] = useState([])
  const [activeTab, setActiveTab] = useState('alerts')

  useEffect(() => {
    api.get('/emergency/signals').then(r => setJunctions(r.data.junctions)).catch(() => {})
  }, [])

  const handleWsMessage = useCallback((msg) => {
    if (msg.type === 'emergency_alert') {
      setAlerts(prev => [msg, ...prev].slice(0, 30))
    } else if (msg.type === 'signal_update') {
      setJunctions(msg.junctions || [])
    } else if (msg.type === 'history') {
      setAlerts(msg.events?.filter(e => e.type === 'emergency_alert') || [])
    }
  }, [])

  const { connected } = useWebSocket('/api/emergency/ws/traffic_police_hq', handleWsMessage)

  return (
    <div className="dashboard">
      <aside className="sidebar">
        <div className="sidebar__logo">
          <span style={{ fontSize: 24 }}>🚔</span>
          <span className="sidebar__brand">Traffic HQ</span>
        </div>
        <nav className="sidebar__nav">
          <div className="sidebar__section-label">Modules</div>
          <button className={`sidebar__item ${activeTab === 'alerts' ? 'active' : ''}`} onClick={() => setActiveTab('alerts')}>
            <span className="sidebar__icon">🚨</span>
            <span>Emergency Alerts</span>
            {alerts.length > 0 && <span className="sidebar__badge">{alerts.length}</span>}
          </button>
          <button className={`sidebar__item ${activeTab === 'signals' ? 'active' : ''}`} onClick={() => setActiveTab('signals')}>
            <span className="sidebar__icon">🚦</span>
            <span>Signal Control</span>
          </button>
          <button className={`sidebar__item ${activeTab === 'cameras' ? 'active' : ''}`} onClick={() => setActiveTab('cameras')}>
            <span className="sidebar__icon">📹</span>
            <span>Live Feeds</span>
          </button>
        </nav>
        <div className="sidebar__footer">
          <div className="sidebar__user">
            <div className="sidebar__avatar">{fullName.charAt(0)}</div>
            <div>
              <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-primary)' }}>{fullName}</div>
              <div style={{ fontSize: 11, color: '#60a5fa' }}>Traffic Police HQ</div>
            </div>
          </div>
          <button className="btn btn-ghost" style={{ width: '100%', justifyContent: 'center', marginTop: 8 }} onClick={() => { localStorage.clear(); navigate('/login') }}>
            Logout
          </button>
        </div>
      </aside>

      <main className="dashboard__main">
        <header className="dashboard__topbar">
          <div>
            <h1 className="dashboard__title">Traffic Police Command</h1>
            <p className="dashboard__subtitle">Bengaluru City Traffic Monitoring</p>
          </div>
          <div className="topbar__connection">
            <div className={`status-dot ${connected ? 'online' : 'alert'}`} />
            <span>{connected ? 'Live' : 'Reconnecting...'}</span>
          </div>
        </header>

        <div className="stats-row">
          <div className="stat-card card">
            <div className="stat-card__icon">🚨</div>
            <div className="stat-card__value" style={{ color: 'var(--color-emergency)' }}>{alerts.length}</div>
            <div className="stat-card__label">Emergency Events</div>
          </div>
          <div className="stat-card card">
            <div className="stat-card__icon">🚦</div>
            <div className="stat-card__value" style={{ color: 'var(--color-preempted)' }}>{junctions.filter(j => j.is_preempted).length}</div>
            <div className="stat-card__label">Preempted Signals</div>
          </div>
          <div className="stat-card card">
            <div className="stat-card__icon">✅</div>
            <div className="stat-card__value" style={{ color: 'var(--color-success)' }}>{junctions.filter(j => !j.is_preempted).length}</div>
            <div className="stat-card__label">Normal Signals</div>
          </div>
        </div>

        <div className="dashboard__split">
          <div className="dashboard__map card">
            <CityMap junctions={junctions} />
          </div>

          <aside className="dashboard__panel">
            <div className="panel__header">
              <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
                {activeTab === 'alerts' ? '🚨 Emergency Alerts' : activeTab === 'signals' ? '🚦 Signal Control' : '📹 Live Feeds'}
              </span>
            </div>
            <div className="panel__content overflow-y-auto">
              {activeTab === 'alerts' && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                  {alerts.length === 0
                    ? <div style={{ textAlign: 'center', padding: '40px 20px', color: 'var(--text-muted)' }}>
                        <div style={{ fontSize: 36 }}>🚨</div>
                        <div style={{ fontSize: 13, marginTop: 12 }}>No emergency events. Monitoring active.</div>
                      </div>
                    : alerts.map((e, i) => <EmergencyAlertCard key={i} event={e} />)}
                </div>
              )}
              {activeTab === 'signals' && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                  {junctions.map(j => (
                    <div key={j.id} className="card" style={{ padding: '12px 14px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <div>
                        <div style={{ fontSize: 13, fontWeight: 600 }}>{j.name}</div>
                        <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>{j.id}</div>
                      </div>
                      <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                        {j.is_preempted && <span style={{ fontSize: 11, color: 'var(--color-preempted)' }}>🚑 Preempted</span>}
                        <SignalDot state={j.state} />
                      </div>
                    </div>
                  ))}
                </div>
              )}
              {activeTab === 'cameras' && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                  {['CAM_001', 'CAM_WEBCAM'].map(camId => (
                    <div key={camId} className="card" style={{ padding: 10 }}>
                      <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 8 }}>{camId}</div>
                      <img src={`/api/cameras/${camId}/stream`} alt={camId}
                        style={{ width: '100%', borderRadius: 'var(--radius-sm)' }}
                        onError={(e) => { e.target.style.display = 'none' }} />
                    </div>
                  ))}
                </div>
              )}
            </div>
          </aside>
        </div>
      </main>
    </div>
  )
}

function SignalDot({ state }) {
  const colors = { GREEN: '#10b981', RED: '#ef4444', YELLOW: '#f59e0b', PREEMPTED: '#22d3ee' }
  const c = colors[state] || '#8b949e'
  return <div style={{ width: 12, height: 12, borderRadius: '50%', background: c, boxShadow: `0 0 6px ${c}` }} />
}
