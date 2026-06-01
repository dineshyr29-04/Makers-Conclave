import { useState, useEffect, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { ViolationAlertCard } from '../components/AlertCard'
import { useWebSocket } from '../hooks/useWebSocket'
import api from '../services/api'
import './Dashboard.css'
import '../components/AlertCard.css'

export default function MunicipalHQDashboard() {
  const navigate = useNavigate()
  const fullName = localStorage.getItem('full_name') || 'Municipal HQ'
  const [violations, setViolations] = useState([])
  const [stats, setStats] = useState({ total: 0, pending: 0, reviewed: 0 })

  useEffect(() => {
    api.get('/violations/').then(r => {
      const v = r.data.violations || []
      setViolations(v)
      setStats({
        total: v.length,
        pending: v.filter(x => x.status === 'PENDING').length,
        reviewed: v.filter(x => x.status === 'REVIEWED').length,
      })
    }).catch(() => {})
  }, [])

  const handleWsMessage = useCallback((msg) => {
    if (msg.type === 'violation_alert') {
      setViolations(prev => [msg, ...prev])
      setStats(s => ({ ...s, total: s.total + 1, pending: s.pending + 1 }))
    } else if (msg.type === 'history') {
      const v = msg.events?.filter(e => e.type === 'violation_alert') || []
      setViolations(v)
    }
  }, [])

  const { connected } = useWebSocket('/api/violations/ws/municipal_hq', handleWsMessage)

  const handleReview = async (event) => {
    try {
      await api.patch(`/violations/${event.event_id}/status`, null, { params: { status: 'REVIEWED' } })
      setViolations(prev => prev.map(v => v.event_id === event.event_id ? { ...v, status: 'REVIEWED' } : v))
      setStats(s => ({ ...s, pending: Math.max(0, s.pending - 1), reviewed: s.reviewed + 1 }))
    } catch { }
  }

  return (
    <div className="dashboard">
      <aside className="sidebar">
        <div className="sidebar__logo">
          <span style={{ fontSize: 24 }}>🏛️</span>
          <span className="sidebar__brand">Municipal</span>
        </div>
        <nav className="sidebar__nav">
          <div className="sidebar__section-label">Modules</div>
          <button className="sidebar__item active">
            <span className="sidebar__icon">⚠️</span>
            <span>Violations</span>
            {stats.pending > 0 && <span className="sidebar__badge sidebar__badge--warn">{stats.pending}</span>}
          </button>
        </nav>
        <div className="sidebar__footer">
          <div className="sidebar__user">
            <div className="sidebar__avatar" style={{ background: 'linear-gradient(135deg, #f59e0b, #ef4444)' }}>{fullName.charAt(0)}</div>
            <div>
              <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-primary)' }}>{fullName}</div>
              <div style={{ fontSize: 11, color: 'var(--color-violation)' }}>Municipal Corp HQ</div>
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
            <h1 className="dashboard__title">Civic Violations Dashboard</h1>
            <p className="dashboard__subtitle">Bengaluru Municipal Corporation</p>
          </div>
          <div className="topbar__connection">
            <div className={`status-dot ${connected ? 'online' : 'alert'}`} />
            <span>{connected ? 'Live' : 'Reconnecting...'}</span>
          </div>
        </header>

        <div className="stats-row">
          <div className="stat-card card">
            <div className="stat-card__icon">📋</div>
            <div className="stat-card__value" style={{ color: 'var(--text-primary)' }}>{stats.total}</div>
            <div className="stat-card__label">Total Violations</div>
          </div>
          <div className="stat-card card">
            <div className="stat-card__icon">⏳</div>
            <div className="stat-card__value" style={{ color: 'var(--color-violation)' }}>{stats.pending}</div>
            <div className="stat-card__label">Pending Review</div>
          </div>
          <div className="stat-card card">
            <div className="stat-card__icon">✅</div>
            <div className="stat-card__value" style={{ color: 'var(--color-success)' }}>{stats.reviewed}</div>
            <div className="stat-card__label">Reviewed</div>
          </div>
        </div>

        {/* Full-width violations list */}
        <div style={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
          <div style={{
            display: 'flex', justifyContent: 'space-between', alignItems: 'center',
            marginBottom: 12, paddingBottom: 12, borderBottom: '1px solid var(--border)'
          }}>
            <span style={{ fontWeight: 600, fontSize: 15 }}>⚠️ Violation Reports</span>
            <span className="text-sm" style={{ color: 'var(--text-muted)' }}>Showing {violations.length} records</span>
          </div>
          <div style={{ flex: 1, overflowY: 'auto', display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', gap: 12, alignContent: 'start' }}>
            {violations.length === 0
              ? <div style={{ gridColumn: '1/-1', textAlign: 'center', padding: '60px 20px', color: 'var(--text-muted)' }}>
                  <div style={{ fontSize: 48 }}>✅</div>
                  <div style={{ marginTop: 16, fontSize: 14 }}>No violations detected. System monitoring is active.</div>
                </div>
              : violations.map((v, i) => <ViolationAlertCard key={i} event={v} onReview={handleReview} />)}
          </div>
        </div>
      </main>
    </div>
  )
}
