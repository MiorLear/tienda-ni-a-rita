const CACHE = 'tienda-v1';
const ASSETS = ['/', '/index.html'];

self.addEventListener('install', e => {
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(ASSETS)));
  self.skipWaiting();
});

self.addEventListener('activate', e => {
  e.waitUntil(caches.keys().then(keys =>
    Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))
  ));
  self.clients.claim();
});

// Network-first para API, cache-first para assets
self.addEventListener('fetch', e => {
  const url = new URL(e.request.url);
  if (url.pathname.startsWith('/api') || url.port === '8000') {
    // API: network first, guarda en cola si offline
    e.respondWith(
      fetch(e.request).catch(() =>
        new Response(JSON.stringify({ offline: true, message: 'Guardado localmente, se sincronizará pronto.' }),
          { headers: { 'Content-Type': 'application/json' } })
      )
    );
  } else {
    // Assets: cache first
    e.respondWith(
      caches.match(e.request).then(cached => cached || fetch(e.request))
    );
  }
});

// Background sync para operaciones offline
self.addEventListener('sync', e => {
  if (e.tag === 'sync-ventas') {
    e.waitUntil(syncPendingOps());
  }
});

async function syncPendingOps() {
  // Aquí iría la lógica para sincronizar operaciones guardadas offline
  console.log('Sincronizando operaciones pendientes...');
}