class ConvergenceChart {
  /**
   * @param {string|HTMLCanvasElement} canvasOrId
   */
  constructor(canvasOrId) {
    const el     = typeof canvasOrId === 'string' ? document.getElementById(canvasOrId) : canvasOrId;
    this.ctx     = el ? el.getContext('2d') : null;
    this.chart   = null;
    if (this.ctx) this._init();
  }

  _init() {
    // Línea de umbral de convergencia
    const thresholdPlugin = {
      id: 'thresholdLine',
      afterDraw(chart) {
        if (!chart.chartArea) return;
        const { ctx, chartArea: { left, right }, scales: { y } } = chart;
        const th = chart.options.plugins.thresholdLine?.value;
        if (th == null || !y) return;
        const yPx = y.getPixelForValue(th);
        if (yPx == null || isNaN(yPx)) return;
        ctx.save();
        ctx.strokeStyle = '#EF4444';
        ctx.lineWidth   = 1.5;
        ctx.setLineDash([6, 4]);
        ctx.globalAlpha = 0.7;
        ctx.beginPath();
        ctx.moveTo(left, yPx);
        ctx.lineTo(right, yPx);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.font        = '10px Fira Code, monospace';
        ctx.fillStyle   = '#EF4444';
        ctx.globalAlpha = 0.85;
        ctx.fillText('Umbral 1e-2', right - 80, yPx - 6);
        ctx.restore();
      },
    };

    this.chart = new Chart(this.ctx, {
      type: 'line',
      plugins: [thresholdPlugin],
      data: {
        labels: [],
        datasets: [
          {
            label: 'Mejor',
            data:  [],
            borderColor:     '#10B981',
            backgroundColor: 'rgba(16,185,129,0.08)',
            borderWidth:     2,
            pointRadius:     0,
            tension:         0.4,
            fill:            false,
          },
          {
            label: 'Promedio',
            data:  [],
            borderColor:     '#3B82F6',
            backgroundColor: 'rgba(59,130,246,0.08)',
            borderWidth:     1.5,
            pointRadius:     0,
            tension:         0.4,
            fill:            false,
            borderDash:      [5, 3],
          },
          {
            label: 'Peor',
            data:  [],
            borderColor:     '#EF4444',
            backgroundColor: 'rgba(239,68,68,0.05)',
            borderWidth:     1,
            pointRadius:     0,
            tension:         0.4,
            fill:            false,
            borderDash:      [3, 3],
          },
        ],
      },
      options: {
        responsive:          true,
        maintainAspectRatio: false,
        animation:           { duration: 0 },
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: {
            labels: {
              color:     '#94A3B8',
              font:      { family: 'Fira Code', size: 11 },
              boxWidth:  20,
              padding:   12,
            },
          },
          tooltip: {
            backgroundColor: 'rgba(27,35,54,0.95)',
            borderColor:     'rgba(255,255,255,0.09)',
            borderWidth:     1,
            titleColor:      '#F8FAFC',
            bodyColor:       '#94A3B8',
            titleFont:       { family: 'Fira Code', size: 11 },
            bodyFont:        { family: 'Fira Code', size: 10 },
            callbacks: {
              label(ctx) {
                return ` ${ctx.dataset.label}: ${ctx.raw.toFixed(5)}`;
              },
            },
          },
          thresholdLine: { value: 1e-2 },
        },
        scales: {
          x: {
            type: 'linear',
            min: 0,
            suggestedMax: 150,
            title: {
              display: true,
              text:    'Generación',
              color:   '#64748B',
              font:    { family: 'Fira Code', size: 10 },
            },
            grid:  { color: 'rgba(255,255,255,0.04)' },
            ticks: {
              color: '#64748B',
              font:  { family: 'Fira Code', size: 10 },
              precision: 0,
              callback: (v) => Number.isInteger(v) && v >= 0 ? v : null,
            },
          },
          y: {
            type:  'logarithmic',
            title: {
              display: true,
              text:    'Fitness (escala log)',
              color:   '#64748B',
              font:    { family: 'Fira Code', size: 10 },
            },
            grid:  { color: 'rgba(255,255,255,0.04)' },
            ticks: {
              color: '#64748B',
              font:  { family: 'Fira Code', size: 10 },
              callback(v) {
                if (v >= 1000) return (v / 1000).toFixed(0) + 'k';
                if (v >= 100)  return Number(v.toFixed(0)).toLocaleString();
                if (v >= 10)   return Number(v.toFixed(0)).toString();
                if (v >= 1)    return Number(v.toFixed(1)).toString();
                if (v >= 0.1)  return Number(v.toFixed(2)).toString();
                if (v >= 0.01) return Number(v.toFixed(2)).toString();
                return v.toString();
              },
            },
          },
        },
      },
    });
  }

  /**
   * Ajusta dinámicamente el límite superior sugerido del eje horizontal.
   * @param {number} maxGens
   */
  setMaxGens(maxGens) {
    if (this.chart && this.chart.options && this.chart.options.scales && this.chart.options.scales.x) {
      this.chart.options.scales.x.suggestedMax = maxGens;
      this.chart.update('none');
    }
  }

  /**
   * Actualiza la gráfica con el historial completo o rebanado del GA.
   * @param {GenerationStats[]} history
   */
  update(history) {
    const chart = this.chart;
    if (!history || history.length === 0) {
      this.clear();
      return;
    }
    const labels = history.map(s => s.gen !== undefined ? s.gen : (s.generation !== undefined ? s.generation : 0));
    chart.data.labels = labels;
    chart.data.datasets[0].pointRadius = history.length <= 1 ? 4 : (ctx => ctx.dataIndex === history.length - 1 ? 3 : 0);
    chart.data.datasets[1].pointRadius = history.length <= 1 ? 3 : 0;
    chart.data.datasets[2].pointRadius = history.length <= 1 ? 3 : 0;
    chart.data.datasets[0].data = history.map(s => {
      const v = s.best !== undefined ? s.best : s.best_fitness;
      return typeof v === 'number' && !isNaN(v) ? Math.max(v, 1e-7) : null;
    });
    chart.data.datasets[1].data = history.map(s => {
      const v = s.avg !== undefined ? s.avg : s.avg_fitness;
      return typeof v === 'number' && !isNaN(v) ? Math.max(v, 1e-7) : null;
    });
    chart.data.datasets[2].data = history.map(s => {
      const v = s.worst !== undefined ? s.worst : s.worst_fitness;
      return typeof v === 'number' && !isNaN(v) ? Math.max(v, 1e-7) : null;
    });
    chart.update('none');
  }

  /** Resetea la gráfica a su estado vacío manteniendo la escala. */
  clear() {
    this.chart.data.labels = [];
    this.chart.data.datasets.forEach(d => (d.data = []));
    this.chart.update('none');
  }
}

/* ════════════════════════════════════════════════════════════
   9. CONTROLADOR DE UI — Orquesta todo
   ════════════════════════════════════════════════════════════ */

/* ════════════════════════════════════════════════════════════
   11. PROBLEMA 2: PCB LAYOUT — Datos, Renderer y Controlador
   ════════════════════════════════════════════════════════════ */

const PCB_PY_DATA = typeof window !== 'undefined' && window.PCB_SIMULATION_DATA ? window.PCB_SIMULATION_DATA : null;

const PCB_INSTANCE = PCB_PY_DATA && PCB_PY_DATA.instance ? PCB_PY_DATA.instance : {
  board_max_width: 100.0,
  board_max_height: 80.0,
  clearance: 2.0,
  components: [
    { id: 0,  ref: "U1",  name: "MCU ARM-Cortex",     width: 16.0, height: 16.0, category: "ic",        color: "#10B981" },
    { id: 1,  ref: "U2",  name: "Regulador LDO 3.3V", width: 10.0, height: 8.0,  category: "power",     color: "#F59E0B" },
    { id: 2,  ref: "Y1",  name: "Cristal 16MHz",      width: 8.0,  height: 5.0,  category: "clock",     color: "#8B5CF6" },
    { id: 3,  ref: "U3",  name: "Memoria SPI Flash",  width: 8.0,  height: 8.0,  category: "ic",        color: "#06B6D4" },
    { id: 4,  ref: "J1",  name: "USB-C Power/Data",   width: 10.0, height: 8.0,  category: "connector", color: "#3B82F6" },
    { id: 5,  ref: "J2",  name: "Header GPIO 2x5",    width: 20.0, height: 6.0,  category: "connector", color: "#6366F1" },
    { id: 6,  ref: "C1",  name: "Cap Desacoplo 100n", width: 4.0,  height: 4.0,  category: "passive",   color: "#EC4899" },
    { id: 7,  ref: "C2",  name: "Cap Desacoplo 100n", width: 4.0,  height: 4.0,  category: "passive",   color: "#EC4899" },
    { id: 8,  ref: "C3",  name: "Cap Entrada 10uF",   width: 6.0,  height: 6.0,  category: "power",     color: "#F97316" },
    { id: 9,  ref: "C4",  name: "Cap Salida 10uF",    width: 6.0,  height: 6.0,  category: "power",     color: "#F97316" },
    { id: 10, ref: "R1",  name: "Res Reset Pull-up",  width: 4.0,  height: 3.0,  category: "passive",   color: "#A855F7" },
    { id: 11, ref: "R2",  name: "Res SPI Term",       width: 4.0,  height: 3.0,  category: "passive",   color: "#A855F7" },
    { id: 12, ref: "R3",  name: "Res LED Estado",     width: 4.0,  height: 3.0,  category: "passive",   color: "#A855F7" },
    { id: 13, ref: "D1",  name: "Diodo ESD USB",      width: 5.0,  height: 4.0,  category: "power",     color: "#EF4444" }
  ],
  nets: []
};

/**
 * Renderizador de alta resolución para el plano técnico del PCB en HTML5 Canvas.
 */
