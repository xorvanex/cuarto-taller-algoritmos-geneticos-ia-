class MarkowitzFrontierChart {
  constructor(canvasOrId) {
    const el = typeof canvasOrId === 'string' ? document.getElementById(canvasOrId) : canvasOrId;
    this.ctx = el ? el.getContext('2d') : null;
    this.chart = null;
    if (this.ctx) this._init();
  }

  _init() {
    this.chart = new Chart(this.ctx, {
      type: 'scatter',
      data: {
        datasets: [
          {
            label: 'Portafolios Factibles (Monte Carlo)',
            data: [],
            backgroundColor: 'rgba(148, 163, 184, 0.35)',
            pointRadius: 2.5,
            pointHoverRadius: 4,
            borderColor: 'transparent',
            order: 3
          },
          {
            label: 'Línea Asignación Capital (CAL)',
            type: 'line',
            data: [],
            borderColor: '#38BDF8',
            borderWidth: 1.8,
            borderDash: [5, 5],
            pointRadius: 0,
            fill: false,
            order: 2
          },
          {
            label: 'Activos Individuales',
            data: [],
            backgroundColor: (ctx) => ctx.raw?.color || '#EF4444',
            borderColor: '#FFFFFF',
            borderWidth: 1.2,
            pointRadius: 6,
            pointHoverRadius: 8,
            order: 1
          },
          {
            label: 'Portafolio Óptimo SGA',
            data: [],
            backgroundColor: '#10B981',
            borderColor: '#FFFFFF',
            borderWidth: 2,
            pointRadius: 10,
            pointHoverRadius: 12,
            pointStyle: 'star',
            order: 0
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 150 },
        scales: {
          x: {
            title: { display: true, text: 'Riesgo / Volatilidad sigma_p (%)', color: '#94A3B8', font: { size: 10 } },
            min: 5,
            max: 25,
            grid: { color: 'rgba(51, 65, 85, 0.4)' },
            ticks: { color: '#94A3B8', callback: v => v + '%' }
          },
          y: {
            title: { display: true, text: 'Rendimiento Esperado E[Rp] (%)', color: '#94A3B8', font: { size: 10 } },
            min: 3,
            max: 21,
            grid: { color: 'rgba(51, 65, 85, 0.4)' },
            ticks: { color: '#94A3B8', callback: v => v + '%' }
          }
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (ctx) => {
                const r = ctx.raw;
                if (ctx.datasetIndex === 0) {
                  return `Portafolio: Retorno ${r.y}% | Vol ${r.x}% | SR: ${r.sharpe}`;
                } else if (ctx.datasetIndex === 2) {
                  return `${r.ticker}: Retorno ${r.y}% | Vol ${r.x}%`;
                } else if (ctx.datasetIndex === 3) {
                  return `Óptimo SGA: Retorno ${r.y}% | Vol ${r.x}% | SR: ${r.sharpe}`;
                }
                return `CAL`;
              }
            }
          }
        }
      }
    });
  }

  setData(cloud, assets, currentPortfolio, rf = 3.5) {
    if (!this.chart) return;
    this.chart.data.datasets[0].data = (cloud || []).map(p => ({ x: p.vol, y: p.ret, sharpe: p.sharpe }));
    if (currentPortfolio && currentPortfolio.volatility > 0) {
      const slope = (currentPortfolio.expected_return - rf) / currentPortfolio.volatility;
      this.chart.data.datasets[1].data = [
        { x: 0, y: rf },
        { x: 25, y: +(rf + slope * 25).toFixed(2) }
      ];
    }
    this.chart.data.datasets[2].data = (assets || []).map(a => ({
      x: +(a.volatility * 100).toFixed(2),
      y: +(a.expected_return * 100).toFixed(2),
      ticker: a.ticker,
      color: a.color
    }));
    if (currentPortfolio) {
      this.chart.data.datasets[3].data = [{
        x: currentPortfolio.volatility,
        y: currentPortfolio.expected_return,
        sharpe: currentPortfolio.sharpe_ratio
      }];
    }
    this.chart.update('none');
  }

  updateCurrentPoint(portfolio, rf = 3.5) {
    if (!this.chart || !portfolio) return;
    this.chart.data.datasets[3].data = [{
      x: portfolio.volatility,
      y: portfolio.expected_return,
      sharpe: portfolio.sharpe_ratio
    }];
    if (portfolio.volatility > 0) {
      const slope = (portfolio.expected_return - rf) / portfolio.volatility;
      this.chart.data.datasets[1].data = [
        { x: 0, y: rf },
        { x: 25, y: +(rf + slope * 25).toFixed(2) }
      ];
    }
    this.chart.update('none');
  }
}

class MarkowitzAllocationChart {
  constructor(canvasOrId) {
    const el = typeof canvasOrId === 'string' ? document.getElementById(canvasOrId) : canvasOrId;
    this.ctx = el ? el.getContext('2d') : null;
    this.chart = null;
    if (this.ctx) this._init();
  }

  _init() {
    this.chart = new Chart(this.ctx, {
      type: 'bar',
      data: {
        labels: [],
        datasets: [{
          label: 'Asignación (%)',
          data: [],
          backgroundColor: [],
          borderRadius: 4,
          borderWidth: 1,
          borderColor: '#1E293B'
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            title: { display: true, text: 'Ponderación Patrimonial (%)', color: '#94A3B8', font: { size: 10 } },
            min: 0,
            max: 38,
            grid: { color: 'rgba(51, 65, 85, 0.4)' },
            ticks: { color: '#94A3B8', callback: v => v + '%' }
          },
          y: {
            grid: { display: false },
            ticks: { color: '#F1F5F9', font: { weight: 'bold', size: 10 } }
          }
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (ctx) => `Peso: ${ctx.parsed.x.toFixed(2)}%`
            }
          }
        }
      }
    });
  }

  updateAllocations(allocations) {
    if (!this.chart || !allocations) return;
    const sorted = [...allocations].sort((a, b) => b.pct - a.pct);
    this.chart.data.labels = sorted.map(a => a.ticker);
    this.chart.data.datasets[0].data = sorted.map(a => a.pct);
    this.chart.data.datasets[0].backgroundColor = sorted.map(a => a.color || '#3B82F6');
    this.chart.update('none');
  }
}

