import { useState, useEffect, useRef, useCallback } from 'react'

/**
 * useWebSocket — connects to the City AI WebSocket server
 * and dispatches received messages to a provided handler.
 *
 * @param {string} endpoint - e.g. "/api/emergency/ws/super_admin"
 * @param {function} onMessage - callback for each parsed message
 */
export function useWebSocket(endpoint, onMessage) {
  const wsRef = useRef(null)
  const [connected, setConnected] = useState(false)
  const reconnectTimeout = useRef(null)
  const onMessageRef = useRef(onMessage)
  onMessageRef.current = onMessage

  const connect = useCallback(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws'
    const url = `${protocol}://${window.location.host}${endpoint}`
    const ws = new WebSocket(url)

    ws.onopen = () => {
      setConnected(true)
      console.log(`[WS] Connected to ${endpoint}`)
    }

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        onMessageRef.current(data)
      } catch (e) {
        console.warn('[WS] Failed to parse message', e)
      }
    }

    ws.onclose = () => {
      setConnected(false)
      console.log(`[WS] Disconnected from ${endpoint}, reconnecting in 3s...`)
      reconnectTimeout.current = setTimeout(connect, 3000)
    }

    ws.onerror = (err) => {
      console.error('[WS] Error', err)
      ws.close()
    }

    wsRef.current = ws
  }, [endpoint])

  useEffect(() => {
    connect()
    return () => {
      clearTimeout(reconnectTimeout.current)
      wsRef.current?.close()
    }
  }, [connect])

  return { connected }
}
