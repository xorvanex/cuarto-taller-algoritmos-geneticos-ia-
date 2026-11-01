document.addEventListener('DOMContentLoaded', () => {
  // Chart.js debe estar cargado (CDN en index.html)
  if (typeof Chart === 'undefined') {
    console.error('[Taller AG] Chart.js no encontrado. Verificar CDN en index.html.');
    return;
  }

  // Registrar plugins de Chart.js necesarios
  Chart.defaults.color = '#94A3B8';
  Chart.defaults.font.family = "'Fira Code', monospace";

  // Iniciar controlador
  window._dashboard = new DashboardController();
  window.app = window._dashboard;

  console.log('[Taller AG] Panel inicializado. Motor SGA listo.');
  console.log('[Taller AG] Instancia VRP: 15 clientes, 3 camiones, capacidad de 120 kg.');
  console.log('[Taller AG] Caso base: N=1000, elitismo=2, Pc=90%, Pm=2%, Gmax=150.');
});
