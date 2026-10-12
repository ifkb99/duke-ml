// Cloudflare Web Analytics. Cloudflare adds this tag to the pages it serves, but
// return visits load index.html from the service worker's copy, which was fetched
// without it: add it here then. Only on the real site, so local builds and the
// phone test server don't report.
const SITE = 'duke.nyaaa.fun';
// The settings of the tag Cloudflare adds, copied from the live page. With a
// `version` the script reports to /cdn-cgi/rum on this site, as Cloudflare's own
// tag does; without one it reports to cloudflareinsights.com, which refuses this
// site's token. The token is public.
const SETTINGS = {version: '2024.11.0', token: 'a02f14fc000c41f98d2b884dee4f5abb', r: 1, spa: 2};

if (location.hostname === SITE && !document.querySelector('script[data-cf-beacon]')) {
  const script = document.createElement('script');
  script.src = 'https://static.cloudflareinsights.com/beacon.min.js';
  script.crossOrigin = 'anonymous';
  script.dataset.cfBeacon = JSON.stringify(SETTINGS);
  document.head.append(script);
}
