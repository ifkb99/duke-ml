import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import '@fontsource-variable/cinzel';
import '@fontsource-variable/outfit';
import { App } from './App.js';
import './index.css';
import './analytics.js';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
