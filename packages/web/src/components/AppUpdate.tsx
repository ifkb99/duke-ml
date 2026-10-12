import {useEffect} from 'react';
import {useRegisterSW} from 'virtual:pwa-register/react';
import {CloseIcon} from './Icons.js';

/**
 * Registers the service worker (vite.config.ts) and says when it has something
 * to tell: the app is stored for offline play, or a newer build is waiting.
 * Updating reloads the page; the game in progress comes back (useGame saves it).
 */
export function AppUpdate() {
  const {needRefresh: [needRefresh, setNeedRefresh], offlineReady: [offlineReady, setOfflineReady],
         updateServiceWorker} = useRegisterSW();

  useEffect(() => {
    if (!offlineReady) return;
    const timer = setTimeout(() => setOfflineReady(false), 5000);
    return () => clearTimeout(timer);
  }, [offlineReady, setOfflineReady]);

  if (needRefresh) {
    return (
      <div className="toast" role="status">
        <span>A new version is ready</span>
        <button type="button" className="btn btn-gold" onClick={() => updateServiceWorker()}>Update</button>
        <button type="button" className="icon-btn" onClick={() => setNeedRefresh(false)} aria-label="Later"><CloseIcon /></button>
      </div>
    );
  }
  if (offlineReady) return <div className="toast" role="status"><span>Ready to play offline</span></div>;
  return null;
}
