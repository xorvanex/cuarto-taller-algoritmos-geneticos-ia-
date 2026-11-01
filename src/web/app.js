document.addEventListener('DOMContentLoaded', () => {
  // Chart.js debe estar cargado (CDN en index.html)
  if (typeof Chart === 'undefined') {
    console.error('[NexusOpt] Chart.js no encontrado. Verificar CDN en index.html.');
    return;
  }

  // Registrar plugins de Chart.js necesarios
  Chart.defaults.color = '#94A3B8';
  Chart.defaults.font.family = "'Fira Code', monospace";

  // Iniciar controlador
  window._dashboard = new DashboardController();
  window.app = window._dashboard;

  console.log('[NexusOpt AI] Dashboard inicializado. Motor SGA listo.');
  console.log('[NexusOpt AI] Instancia VRP: 15 clientes, 3 camiones, cap=120kg.');
  console.log('[NexusOpt AI] Caso base cátedra: N=1000, e=2, Pc=90%, Pm=2%, maxGens=150.');
});
