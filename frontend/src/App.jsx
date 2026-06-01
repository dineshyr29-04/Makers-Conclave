import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import Login from './pages/Login'
import SuperAdmin from './pages/SuperAdmin'
import TrafficPoliceHQ from './pages/TrafficPoliceHQ'
import MunicipalHQ from './pages/MunicipalHQ'
import './index.css'

function ProtectedRoute({ children, allowedRole }) {
  const token = localStorage.getItem('token')
  const role = localStorage.getItem('role')
  if (!token) return <Navigate to="/login" replace />
  if (allowedRole && role !== allowedRole && role !== 'super_admin') {
    return <Navigate to={`/${role === 'traffic_police_hq' ? 'traffic' : role === 'municipal_hq' ? 'municipal' : 'login'}`} replace />
  }
  return children
}

export default function App() {
  const role = localStorage.getItem('role')

  return (
    <BrowserRouter>
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route path="/admin" element={
          <ProtectedRoute allowedRole="super_admin"><SuperAdmin /></ProtectedRoute>
        } />
        <Route path="/traffic" element={
          <ProtectedRoute allowedRole="traffic_police_hq"><TrafficPoliceHQ /></ProtectedRoute>
        } />
        <Route path="/municipal" element={
          <ProtectedRoute allowedRole="municipal_hq"><MunicipalHQ /></ProtectedRoute>
        } />
        <Route path="/" element={
          role ? <Navigate to={role === 'super_admin' ? '/admin' : role === 'traffic_police_hq' ? '/traffic' : '/municipal'} />
               : <Navigate to="/login" />
        } />
        <Route path="*" element={<Navigate to="/" />} />
      </Routes>
    </BrowserRouter>
  )
}
