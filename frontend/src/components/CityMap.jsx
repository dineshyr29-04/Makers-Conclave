import { useEffect, useRef } from 'react'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'

const SIGNAL_COLORS = {
  GREEN: '#10b981',
  RED: '#ef4444',
  YELLOW: '#f59e0b',
  PREEMPTED: '#22d3ee',
}

// Fix Leaflet default marker icons
delete L.Icon.Default.prototype._getIconUrl
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
})

export default function CityMap({ junctions = [], onJunctionClick }) {
  const mapContainer = useRef(null)
  const mapRef = useRef(null)
  const markersRef = useRef({})

  // Initialize Leaflet map once
  useEffect(() => {
    if (mapRef.current) return

    mapRef.current = L.map(mapContainer.current, {
      center: [12.9716, 77.5946],
      zoom: 14,
      zoomControl: true,
    })

    // Dark tile layer (CartoDB Dark Matter — free, no token)
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
      attribution: '&copy; <a href="https://carto.com/">CARTO</a>',
      subdomains: 'abcd',
      maxZoom: 20,
    }).addTo(mapRef.current)

    return () => {
      mapRef.current?.remove()
      mapRef.current = null
      markersRef.current = {}
    }
  }, [])

  // Update markers when junctions change
  useEffect(() => {
    if (!mapRef.current || junctions.length === 0) return

    junctions.forEach((junction) => {
      const { id, lat, lon, name, state, is_preempted } = junction
      const color = SIGNAL_COLORS[state] || SIGNAL_COLORS.RED

      // Remove old marker if it exists
      if (markersRef.current[id]) {
        mapRef.current.removeLayer(markersRef.current[id])
      }

      // Create a custom DivIcon
      const icon = L.divIcon({
        html: `
          <div style="
            width:20px; height:20px; border-radius:50%;
            background:${color};
            border: 2px solid rgba(255,255,255,0.4);
            box-shadow: 0 0 0 3px ${color}40, 0 0 14px ${color}80;
            ${is_preempted ? 'animation: pulse-map 1.2s infinite;' : ''}
          "></div>
        `,
        className: '',
        iconSize: [20, 20],
        iconAnchor: [10, 10],
      })

      const marker = L.marker([lat, lon], { icon })
        .addTo(mapRef.current)
        .bindPopup(`
          <div style="font-family:Inter,sans-serif; min-width:160px">
            <div style="font-weight:700; font-size:13px; margin-bottom:4px">${name}</div>
            <div style="font-size:11px; color:#666; margin-bottom:6px">${id}</div>
            <div style="display:flex; align-items:center; gap:6px">
              <div style="width:10px;height:10px;border-radius:50%;background:${color}"></div>
              <span style="font-weight:600; color:${color}">${state}</span>
              ${is_preempted ? '<span style="font-size:11px;color:#22d3ee">🚑 Preempted</span>' : ''}
            </div>
          </div>
        `)

      if (onJunctionClick) {
        marker.on('click', () => onJunctionClick(junction))
      }

      markersRef.current[id] = marker
    })
  }, [junctions, onJunctionClick])

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%', borderRadius: 'var(--radius-md)', overflow: 'hidden' }}>
      <style>{`
        @keyframes pulse-map {
          0%, 100% { box-shadow: 0 0 0 3px rgba(34,211,238,0.4), 0 0 14px rgba(34,211,238,0.6); }
          50% { box-shadow: 0 0 0 8px rgba(34,211,238,0.1), 0 0 20px rgba(34,211,238,0.8); }
        }
        .leaflet-container { background: #0d1117; }
        .leaflet-popup-content-wrapper {
          background: #0d1117; border: 1px solid rgba(255,255,255,0.1);
          color: #f0f6ff; border-radius: 10px;
        }
        .leaflet-popup-tip { background: #0d1117; }
        .leaflet-control-zoom a {
          background: #0d1117 !important; color: #f0f6ff !important;
          border-color: rgba(255,255,255,0.1) !important;
        }
        .leaflet-control-attribution { display: none; }
      `}</style>

      <div ref={mapContainer} style={{ width: '100%', height: '100%' }} />

      {/* Signal legend */}
      <div style={{
        position: 'absolute', bottom: 16, left: 16, zIndex: 1000,
        background: 'rgba(13,17,23,0.9)', border: '1px solid rgba(255,255,255,0.1)',
        borderRadius: '10px', padding: '10px 14px',
        backdropFilter: 'blur(12px)',
      }}>
        <div style={{ fontSize: 10, fontWeight: 600, color: '#484f58', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: 8 }}>
          Signal Status
        </div>
        {Object.entries(SIGNAL_COLORS).map(([state, color]) => (
          <div key={state} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
            <div style={{ width: 10, height: 10, borderRadius: '50%', background: color, boxShadow: `0 0 6px ${color}` }} />
            <span style={{ fontSize: 11, color: '#8b949e' }}>{state}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
