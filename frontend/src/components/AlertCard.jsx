import { formatDistanceToNow } from 'date-fns'

export function EmergencyAlertCard({ event, onDismiss }) {
  const timeAgo = formatDistanceToNow(new Date(event.timestamp), { addSuffix: true })

  return (
    <div className="alert-card alert-card--emergency animate-slide-in">
      <div className="alert-card__header">
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <div className="status-dot alert" style={{ width: 10, height: 10 }} />
          <span className="badge badge-emergency">🚨 Emergency</span>
        </div>
        <span className="alert-card__time">{timeAgo}</span>
      </div>

      <div className="alert-card__title">
        {event.vehicle_type} Detected
      </div>

      <div className="alert-card__grid">
        <div className="alert-card__field">
          <span className="alert-card__label">Plate</span>
          <span className="alert-card__value font-mono">{event.plate_number || 'Unread'}</span>
        </div>
        <div className="alert-card__field">
          <span className="alert-card__label">Camera</span>
          <span className="alert-card__value font-mono">{event.camera_id}</span>
        </div>
        <div className="alert-card__field">
          <span className="alert-card__label">Junction</span>
          <span className="alert-card__value">{event.junction_id}</span>
        </div>
        <div className="alert-card__field">
          <span className="alert-card__label">Confidence</span>
          <span className="alert-card__value">{Math.round(event.confidence * 100)}%</span>
        </div>
      </div>

      {event.junctions_cleared?.length > 0 && (
        <div className="alert-card__cleared">
          <span style={{ color: 'var(--text-muted)', fontSize: 11 }}>Route cleared: </span>
          {event.junctions_cleared.map(j => (
            <span key={j} className="badge badge-info" style={{ fontSize: 10 }}>{j}</span>
          ))}
        </div>
      )}

      {event.snapshot_url && (
        <img src={event.snapshot_url} alt="Vehicle snapshot" className="alert-card__snapshot" />
      )}

      {onDismiss && (
        <button className="alert-card__dismiss" onClick={() => onDismiss(event.event_id)}>
          Acknowledge
        </button>
      )}
    </div>
  )
}

export function ViolationAlertCard({ event, onReview }) {
  const timeAgo = formatDistanceToNow(new Date(event.timestamp), { addSuffix: true })

  return (
    <div className="alert-card alert-card--violation animate-slide-in">
      <div className="alert-card__header">
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <div className="status-dot warn" style={{ width: 10, height: 10 }} />
          <span className="badge badge-violation">⚠ Violation</span>
        </div>
        <span className="alert-card__time">{timeAgo}</span>
      </div>

      <div className="alert-card__title">
        {event.violation_type} — {event.object_class}
      </div>

      <div className="alert-card__grid">
        <div className="alert-card__field">
          <span className="alert-card__label">Plate</span>
          <span className="alert-card__value font-mono">{event.plate_number || 'Not captured'}</span>
        </div>
        <div className="alert-card__field">
          <span className="alert-card__label">Camera</span>
          <span className="alert-card__value font-mono">{event.camera_id}</span>
        </div>
        <div className="alert-card__field">
          <span className="alert-card__label">Confidence</span>
          <span className="alert-card__value">{Math.round(event.confidence * 100)}%</span>
        </div>
      </div>

      {/* Evidence images */}
      {(event.face_image_url || event.body_image_url) && (
        <div className="alert-card__evidence">
          {event.face_image_url && (
            <div className="evidence-item">
              <span className="evidence-label">Face</span>
              <img src={event.face_image_url} alt="Face capture" className="evidence-img" />
            </div>
          )}
          {event.body_image_url && (
            <div className="evidence-item">
              <span className="evidence-label">Body</span>
              <img src={event.body_image_url} alt="Body capture" className="evidence-img" />
            </div>
          )}
        </div>
      )}

      {onReview && (
        <button className="alert-card__dismiss alert-card__dismiss--warn" onClick={() => onReview(event)}>
          Review & Action
        </button>
      )}
    </div>
  )
}
