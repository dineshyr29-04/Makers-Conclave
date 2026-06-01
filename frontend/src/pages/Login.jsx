import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import api from '../services/api'
import './Login.css'

const ROLE_ROUTES = {
  super_admin: '/admin',
  traffic_police_hq: '/traffic',
  municipal_hq: '/municipal',
  junction_operator: '/operator',
}

export default function Login() {
  const navigate = useNavigate()
  const [form, setForm] = useState({ username: '', password: '' })
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError('')
    try {
      const body = new URLSearchParams({ username: form.username, password: form.password })
      const { data } = await api.post('/auth/token', body, {
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      })
      localStorage.setItem('token', data.access_token)
      localStorage.setItem('role', data.role)
      localStorage.setItem('username', data.username)
      localStorage.setItem('full_name', data.full_name)
      navigate(ROLE_ROUTES[data.role] || '/admin')
    } catch {
      setError('Invalid credentials. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  const fillDemo = (role) => {
    const creds = {
      super_admin: ['admin', 'admin123'],
      traffic_police_hq: ['traffic_hq', 'traffic123'],
      municipal_hq: ['municipal_hq', 'municipal123'],
      junction_operator: ['operator1', 'op123'],
    }
    const [u, p] = creds[role]
    setForm({ username: u, password: p })
  }

  return (
    <div className="login-bg">
      {/* Animated grid */}
      <div className="login-grid" />

      {/* Glow orbs */}
      <div className="orb orb-blue" />
      <div className="orb orb-cyan" />

      <div className="login-container animate-fade-in">
        {/* Logo / Header */}
        <div className="login-header">
          <div className="login-logo">
            <svg width="36" height="36" viewBox="0 0 36 36" fill="none">
              <circle cx="18" cy="18" r="17" stroke="#3b82f6" strokeWidth="2" />
              <path d="M18 8 L26 14 L26 22 L18 28 L10 22 L10 14 Z" fill="rgba(59,130,246,0.2)" stroke="#3b82f6" strokeWidth="1.5" />
              <circle cx="18" cy="18" r="4" fill="#3b82f6" />
            </svg>
          </div>
          <h1 className="login-title">City Intelligence Platform</h1>
          <p className="login-subtitle">Unified AI-powered city monitoring system</p>
        </div>

        {/* Form */}
        <form className="login-form" onSubmit={handleSubmit}>
          <div className="form-group">
            <label htmlFor="username">Username</label>
            <input
              id="username"
              type="text"
              placeholder="Enter your username"
              value={form.username}
              onChange={(e) => setForm({ ...form, username: e.target.value })}
              autoComplete="username"
              required
            />
          </div>
          <div className="form-group">
            <label htmlFor="password">Password</label>
            <input
              id="password"
              type="password"
              placeholder="Enter your password"
              value={form.password}
              onChange={(e) => setForm({ ...form, password: e.target.value })}
              autoComplete="current-password"
              required
            />
          </div>

          {error && <div className="login-error">{error}</div>}

          <button type="submit" className="btn btn-primary login-btn" disabled={loading}>
            {loading ? <span className="spinner" /> : null}
            {loading ? 'Authenticating...' : 'Access System'}
          </button>
        </form>

        {/* Demo Quick Access */}
        <div className="demo-access">
          <span className="demo-label">Quick Demo Access</span>
          <div className="demo-btns">
            {['super_admin', 'traffic_police_hq', 'municipal_hq', 'junction_operator'].map((role) => (
              <button key={role} className="demo-btn" onClick={() => fillDemo(role)}>
                {role.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
              </button>
            ))}
          </div>
        </div>

        <p className="login-footer">
          Makers Conclave Demo · All data is simulated
        </p>
      </div>
    </div>
  )
}
