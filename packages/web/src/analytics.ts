// Cloudflare Web Analytics. Cloudflare adds this tag to the pages it serves, but
// return visits load index.html from the service worker's copy, which was fetched
// without it: add it here then. Only on the real site, so local builds and the
// phone test server don't report.
const SITE = 'duke.nyaaa.fun';
const TOKEN = 'a02f14fc000c41f98d2b884dee4f5abb';   // public: it is in the tag Cloudflare adds

if (location.hostname === SITE && !document.querySelector('script[data-cf-beacon]')) {
  const script = document.createElement('script');
  script.src = 'https://static.cloudflareinsights.com/beacon.min.js';
  script.crossOrigin = 'anonymous';
  script.dataset.cfBeacon = JSON.stringify({token: TOKEN});
  document.head.append(script);
}
