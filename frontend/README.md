# City Intelligence Platform - Frontend

This directory contains the React + Vite frontend for the City Intelligence Platform. It provides a polished, role-based dashboard for city monitoring, integrating live camera feeds, WebSocket alerts, and a real-time city map.

## Architecture & Tech Stack

- **Framework**: React 19 + Vite for ultra-fast development and optimized builds.
- **Styling**: Vanilla CSS with CSS Variables for a customized, glassmorphism design system. Modern UI with CSS animations.
- **Routing**: `react-router-dom` for role-based protected routes (Super Admin, Traffic HQ, Municipal HQ).
- **Maps**: `leaflet` and `react-leaflet` with free CartoDB Dark Matter tiles. No API tokens required.
- **Networking**: 
  - `axios` for standard REST API requests.
  - Native WebSockets wrapped in a custom `useWebSocket` hook with auto-reconnection logic for real-time alerts.

## Project Structure

- `/src/assets` - Static assets and SVG icons.
- `/src/components` - Reusable UI components (Alert Cards, Map, etc.).
- `/src/hooks` - Custom React hooks (e.g., `useWebSocket`).
- `/src/pages` - Page-level components corresponding to different routes (Login, SuperAdmin, TrafficPoliceHQ, MunicipalHQ).
- `/src/services` - API client configurations and interceptors.
- `index.css` - Global styles, design tokens, and utility classes.

## Development Setup

1. **Prerequisites**: Node.js 18+
2. **Install Dependencies**:
   ```bash
   npm install
   ```
3. **Environment Setup**:
   The frontend proxy (in `vite.config.js`) routes `/api` and `/api/*/ws` to `localhost:8000` (the backend server).
4. **Run the Development Server**:
   ```bash
   npm run dev
   ```

## Contribution Guidelines

- **Styling**: Always use the CSS variables defined in `index.css` for colors, spacing, and typography to ensure consistency across the application.
- **Adding New Pages**: Create the page in `/src/pages`, then add the corresponding protected route in `App.jsx`. Use the layout patterns established in the existing dashboards.
