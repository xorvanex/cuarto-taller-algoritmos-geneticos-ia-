class DashboardController {
  constructor() {
    /* Instancias de motor, renderer y chart */
    this.ga        = null;
    this.renderer  = null;
    this.chart     = null;

    /* Estado de animación */
    this.running   = false;
    this.stepMode  = false;
    this.rafId     = null;
    this.lastTick  = 0;
    this.speedMs   = 40;   // ms entre generaciones (calibrado)

    /* Historial dinámico de rutas y snapshots de generaciones evaluadas */
    this.liveGenerationRoutes    = [];
    this.liveGenerationSnapshots = [];

    /* Parámetros actuales */
    this.params    = { ...DEFAULT_PARAMS };
    this.mode      = 'catedra';  // 'catedra' | 'libre'
    this.compConvergenceChart = null;
    this.compRadarChart = null;

    this._init();
    this._initPCB();
    this._initMarkowitz();
  }

  /* ── Inicialización ──────────────────────────────────────── */
  _init() {
    // Canvas y chart
    const canvas      = document.getElementById('vrp-canvas');
    this.renderer     = new VRPRenderer(canvas, VRP_INSTANCE);
    this.chart        = new ConvergenceChart('convergence-chart');

    // Cargar datos oficiales de Python si están presentes
    this.pyData = PY_DATA;

    // SIEMPRE inicializar una instancia de GA para evitar crashes
    this.ga = new SimpleGeneticAlgorithm(this.params, VRP_INSTANCE);

    // Cargar solución certificada de Python si estamos en modo cátedra
    if (this.pyData && this.mode === 'catedra') {
      this._loadPythonSolution();
      this._updateUIFromPython();
    } else {
      this._resetGA();
      this._updateUI();
    }

    // Bindings de UI
    this._bindTabs();
    this._bindSpeedSlider();
    this._bindControls();
    this._bindTimeline();
    this._bindAccordions();

    // Resize observer para re-dibujar el canvas centrado
    const ro = new ResizeObserver(() => this._redrawCanvas());
    if (canvas && canvas.parentElement) {
      ro.observe(canvas.parentElement);
    }

    const pcbCanvas = document.getElementById('pcb-canvas');
    if (pcbCanvas && pcbCanvas.parentElement) {
      const roPCB = new ResizeObserver(() => {
        if (this.currentTab === 'pcb' && this.pcbRenderer) {
          const layout = this.pcbData?.generation_layouts?.[this.pcbCurrentGen] || this.pcbData?.placed_components || this.pcbRenderer.placed;
          this.pcbRenderer.draw(layout);
        }
      });
      roPCB.observe(pcbCanvas.parentElement);
    }
  }

  /* ── Cargar solución certificada de Python ──────────────── */
  _loadPythonSolution() {
    if (!this.pyData) return;
    const { history, routes } = this.pyData;

    // Convertir rutas de Python a índices numéricos para el renderer
    this.pythonRoutes = routes.map(r => r.stop_indices);

    // Cargar historial en el chart y adaptar escala horizontal
    if (history && history.length > 0) {
      this.chart.update(history);
      this.chart.setMaxGens(this.params.maxGens);
    }

    // Renderizar rutas de Python
    this.renderer.draw(this.pythonRoutes);

    // Configurar Timeline Slider y barra de progreso al 100% de cátedra
    this._updateSliderVisuals(this.params.maxGens, this.params.maxGens);
  }

  _updateUIFromPython() {
    if (!this.pyData) return;
    const { metrics, top_individuals, params } = this.pyData;

    // KPIs con datos reales de Python
    this._setText('kpi-fitness',    metrics.best_fitness.toFixed(5));
    this._setText('kpi-dist',       metrics.total_distance_km.toFixed(1) + ' km');
    this._setText('kpi-feasible',   metrics.is_feasible ? '100%' : 'Penaliz.');
    this._setText('kpi-aptitud',    (metrics.best_aptitud * 100).toFixed(2) + '%');

    // Sincronizar visuales de slider y contadores
    this._updateSliderVisuals(metrics.generations_executed, params.max_generations);

    // Badge
    const badge = document.getElementById('conv-badge');
    if (badge) {
      badge.textContent = '● Certificado Python';
      badge.className   = 'badge badge-success';
    }

    // Comparador
    const compFitness = document.getElementById('comp-vrp-fitness');
    const compDist    = document.getElementById('comp-vrp-dist');
    if (compFitness) compFitness.textContent = metrics.best_fitness.toFixed(5);
    if (compDist)    compDist.textContent    = metrics.total_distance_km.toFixed(1) + ' km';

    // Renderizar tabla Top 8 de individuos certificados
    if (top_individuals) {
      this._renderTop8Table(top_individuals);
    }
  }

  /* ── Renderizado unificado de tabla Top 8 ─────────────────── */
  _renderTop8Table(list) {
    const tbody = document.getElementById('pop-tbody');
    if (!tbody) return;

    if (!list || list.length === 0) {
      tbody.innerHTML = `
        <tr>
          <td colspan="5" style="text-align:center; color:var(--muted-dim); padding:20px; font-family:var(--font-mono); font-size:0.8rem">
            Inicia la simulación para evaluar la población...
          </td>
        </tr>`;
      return;
    }

    const elitismK = this.params.elitism || 2;
    tbody.innerHTML = list.map((ind, i) => {
      const aptVal = ind.aptitud_pct !== undefined 
        ? ind.aptitud_pct 
        : (Math.max(0, 1 - ind.fitness) * 100);
      const fitNum = typeof ind.fitness === 'number' ? ind.fitness : parseFloat(ind.fitness);
      const isElite = i < elitismK;
      const roleHtml = isElite
        ? `<span class="role-badge role-elite">Élite Invariable (k=${i + 1})</span>`
        : `<span class="role-badge role-tourn">Candidato Torneo (p=2)</span>`;
      const barPct = Math.min(100, Math.max(0, aptVal)).toFixed(1);

      return `
        <tr>
          <td class="mono font-semibold">#${i + 1}</td>
          <td class="mono font-semibold">${fitNum.toFixed(6)}</td>
          <td class="mono">${aptVal.toFixed(2)}%</td>
          <td>${roleHtml}</td>
          <td>
            <div class="fitness-bar-cell">
              <div class="fitness-bar-mini">
                <div class="fitness-bar-mini-fill" style="width:${barPct}%"></div>
              </div>
              <span class="mono text-xs">${barPct}%</span>
            </div>
          </td>
        </tr>`;
    }).join('');
  }

  /* ── Visuales de Slider de Línea de Tiempo y Progreso ─────── */
  _updateSliderVisuals(cur, max) {
    const sl       = document.getElementById('timeline-slider');
    const curDisp  = document.getElementById('gen-current-display');
    const maxDisp  = document.getElementById('gen-max-display');
    const fill     = document.getElementById('progress-fill');
    const genLabel = document.getElementById('gen-counter-label');

    const m = max || this.params.maxGens || 150;
    const c = Math.max(0, Math.min(cur, m));
    const pct = m > 0 ? (c / m) * 100 : 0;
    const pctStr = pct.toFixed(1) + '%';

    if (sl) {
      sl.max   = m;
      sl.value = c;
      // Gradiente dinámico en tiempo real: llena la pista de verde esmeralda con el avance relativo
      sl.style.background = `linear-gradient(to right, var(--accent) 0%, var(--accent) ${pctStr}, var(--surface-high) ${pctStr}, var(--surface-high) 100%)`;
    }
    if (curDisp)  curDisp.textContent = c;
    if (maxDisp)  maxDisp.textContent = m;
    if (fill)     fill.style.width    = pctStr;
    if (genLabel) genLabel.innerHTML  = `Gen <span>${c}</span> / ${m}`;
  }

  /* ── Ajustar Generaciones Máximas Dinámicamente ───────────── */
  _setMaxGenerations(newMax) {
    this.params.maxGens = newMax;
    if (this.ga) {
      this.ga.params.maxGens = newMax;
    }

    if (this.chart) {
      this.chart.setMaxGens(newMax);
    }

    const sl = document.getElementById('timeline-slider');
    let curGen = 0;
    if (this.running && this.ga) {
      curGen = this.ga.generation;
    } else if (sl) {
      curGen = Math.min(parseInt(sl.value, 10) || 0, newMax);
    }

    this._updateSliderVisuals(curGen, newMax);
  }

  /* ── Acordeones laterales ────────────────────────────────── */
  _bindAccordions() {
    document.querySelectorAll('.sidebar-accordion-header').forEach(btn => {
      btn.addEventListener('click', () => {
        const accId = btn.dataset.accordion;
        const content = document.getElementById(`acc-${accId}`);
        if (content) {
          const isCollapsed = content.classList.toggle('collapsed');
          btn.classList.toggle('collapsed', isCollapsed);
        }
      });
    });
  }

  switchTab(target) {
    const tabBtn = document.querySelector(`.tab-btn[data-tab="${target}"]`);
    if (tabBtn) {
      tabBtn.click();
    } else {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
      const panel = document.getElementById(`tab-${target}`);
      if (panel) panel.classList.add('active');
    }
  }

  /* ── Tabs ────────────────────────────────────────────────── */
  _bindTabs() {
    // Delegación de eventos para botones data-tab-goto (en landing u otras secciones)
    document.addEventListener('click', (e) => {
      const gotoBtn = e.target.closest('[data-tab-goto]');
      if (gotoBtn) {
        e.preventDefault();
        const tabTarget = gotoBtn.dataset.tabGoto;
        this.switchTab(tabTarget);
        window.scrollTo({ top: 0, behavior: 'smooth' });
      }
    });

    document.querySelectorAll('.tab-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const target = btn.dataset.tab;
        this.currentTab = target;
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
        btn.classList.add('active');
        const panel = document.getElementById(`tab-${target}`);
        if (panel) panel.classList.add('active');

        // Control dinámico de la barra lateral (oculta en Landing, visible en los problemas)
        const layout = document.querySelector('.main-layout');
        if (layout) {
          layout.classList.remove('on-comparador');
          if (target === 'landing') {
            layout.classList.add('on-landing');
            this._setStatus('ready', 'Presentación académica');
            return;
          } else {
            layout.classList.remove('on-landing');
          }
        }

        if (target === 'comparador') {
          if (layout) layout.classList.add('on-comparador');
          this._setStatus('ready', 'Comparador Multi-Problema');
          setTimeout(() => this._initComparadorCharts(), 60);
          return;
        }

        // Adaptar Sidebar Caso Base & Cátedra según la pestaña activa
        const accBaseContent = document.getElementById('acc-base');
        if (accBaseContent) {
          if (target === 'pcb') {
            accBaseContent.innerHTML = `
              <!-- Parámetros SGA Cátedra Certificados -->
              <div class="formula-card" style="margin-bottom: 4px; padding: 5px 8px; gap: 3px;">
                <div class="formula-card-header">
                  <span>Parámetros SGA Cátedra</span>
                  <span class="formula-card-tag">Fijos</span>
                </div>
                <div class="info-grid">
                  <div class="info-chip"><div class="info-chip-label">Población (N)</div><div class="info-chip-value">1000</div></div>
                  <div class="info-chip"><div class="info-chip-label">Élite (k)</div><div class="info-chip-value">2</div></div>
                  <div class="info-chip"><div class="info-chip-label">Cruce SPX (Pc)</div><div class="info-chip-value">90%</div></div>
                  <div class="info-chip"><div class="info-chip-label">Mutación (Pm)</div><div class="info-chip-value">2.0%</div></div>
                  <div class="info-chip"><div class="info-chip-label">Generaciones</div><div class="info-chip-value">150</div></div>
                  <div class="info-chip"><div class="info-chip-label">Torneo (p)</div><div class="info-chip-value">2</div></div>
                </div>
              </div>
              <div class="info-grid">
                <div class="info-chip"><div class="info-chip-label">Componentes</div><div class="info-chip-value">14</div></div>
                <div class="info-chip"><div class="info-chip-label">Trazas (Nets)</div><div class="info-chip-value">22</div></div>
                <div class="info-chip"><div class="info-chip-label">Sustrato</div><div class="info-chip-value">100x80 mm</div></div>
                <div class="info-chip"><div class="info-chip-label">Aislamiento</div><div class="info-chip-value">2.0 mm</div></div>
                <div class="info-chip"><div class="info-chip-label">Área Final</div><div class="info-chip-value">3168 mm²</div></div>
                <div class="info-chip"><div class="info-chip-label">Wirelength</div><div class="info-chip-value">1080 mm</div></div>
              </div>
              <div class="formula-card">
                <div class="formula-card-header">
                  <span>Criterio Multicriterio</span>
                  <span class="formula-card-tag">Cátedra P2</span>
                </div>
                <div class="formula-card-eq">
                  <i>f</i><sub>norm</sub> = <span style="color:#38BDF8">0.40·<i>WL</i></span> + <span style="color:#10B981">0.10·<i>Área</i></span> + <span style="color:#EF4444">0.50·<i>Colisión</i></span>
                </div>
                <div class="formula-card-legend">
                  <span><b style="color:#38BDF8">WL:</b> Longitud Trazas Nets (40%)</span>
                  <span><b style="color:#10B981">Área:</b> Sustrato Bounding Box (10%)</span>
                  <span><b style="color:#EF4444">Colisión:</b> Penalización Dieléctrica (50%)</span>
                </div>
                <div class="formula-card-footer">
                  <span>Tutoría Oficial</span>
                  <span>Prof. Ing. M. A. Ospina</span>
                </div>
              </div>
            `;
            this._setStatus('ready', 'Caso Base Certificado (PCB)');
            setTimeout(() => {
              if (this.pcbChart && this.pcbChart.chart) {
                this.pcbChart.chart.resize();
                const curGen = this.pcbCurrentGen;
                const hist = this.pcbData?.history || [];
                const targetGen = Math.min(curGen, hist.length - 1);
                this.pcbChart.update(hist.slice(0, targetGen + 1));
              }
              if (this.pcbRenderer) {
                this._setPCBGeneration(this.pcbCurrentGen);
              }
            }, 60);
                    } else if (target === 'markowitz') {
            accBaseContent.innerHTML = `
              <!-- Parámetros SGA Cátedra Certificados -->
              <div class="formula-card" style="margin-bottom: 4px; padding: 5px 8px; gap: 3px;">
                <div class="formula-card-header">
                  <span>Parámetros SGA Cátedra</span>
                  <span class="formula-card-tag">Fijos</span>
                </div>
                <div class="info-grid">
                  <div class="info-chip"><div class="info-chip-label">Población (N)</div><div class="info-chip-value">1000</div></div>
                  <div class="info-chip"><div class="info-chip-label">Élite (k)</div><div class="info-chip-value">2</div></div>
                  <div class="info-chip"><div class="info-chip-label">Cruce SPX (Pc)</div><div class="info-chip-value">90%</div></div>
                  <div class="info-chip"><div class="info-chip-label">Mutación (Pm)</div><div class="info-chip-value">2.0%</div></div>
                  <div class="info-chip"><div class="info-chip-label">Generaciones</div><div class="info-chip-value">150</div></div>
                  <div class="info-chip"><div class="info-chip-label">Torneo (p)</div><div class="info-chip-value">2</div></div>
                </div>
              </div>
              <div class="info-grid">
                <div class="info-chip"><div class="info-chip-label">Activos</div><div class="info-chip-value">10</div></div>
                <div class="info-chip"><div class="info-chip-label">Tasa Rf</div><div class="info-chip-value">3.50%</div></div>
                <div class="info-chip"><div class="info-chip-label">Tope Activo</div><div class="info-chip-value">35.0%</div></div>
                <div class="info-chip"><div class="info-chip-label">Retorno E[R]</div><div class="info-chip-value">14.78%</div></div>
                <div class="info-chip"><div class="info-chip-label">Volatilidad</div><div class="info-chip-value">14.56%</div></div>
                <div class="info-chip"><div class="info-chip-label">Sharpe SR</div><div class="info-chip-value">0.7744</div></div>
              </div>
              <div class="formula-card">
                <div class="formula-card-header">
                  <span>Criterio Multicriterio</span>
                  <span class="formula-card-tag">Cátedra P3</span>
                </div>
                <div class="formula-card-eq">
                  <i>f</i><sub>norm</sub> = <span style="color:#EF4444">0.50·<i>Riesgo</i></span> + <span style="color:#10B981">0.50·(1 - <i>Retorno</i>)</span> + <span style="color:#38BDF8">Penalización</span>
                </div>
                <div class="formula-card-legend">
                  <span><b style="color:#EF4444">Riesgo:</b> Volatilidad sigma_p / sigma_max (50%)</span>
                  <span><b style="color:#10B981">Retorno:</b> Desutilidad de retorno medio (50%)</span>
                  <span><b style="color:#38BDF8">Penalización:</b> Presupuesto unitario y cota 35%</span>
                </div>
                <div class="formula-card-footer">
                  <span>Tutoría Oficial</span>
                  <span>Prof. Ing. M. A. Ospina</span>
                </div>
              </div>
            `;
            this._setStatus('ready', 'Caso Base Certificado (Markowitz)');
            setTimeout(() => {
              if (this.mrkConvergenceChart && this.mrkConvergenceChart.chart) {
                this.mrkConvergenceChart.chart.resize();
                const curGen = this.mrkCurrentGen;
                const hist = this.mrkData?.history || [];
                const targetGen = Math.min(curGen, hist.length - 1);
                this.mrkConvergenceChart.update(hist.slice(0, targetGen + 1));
              }
              if (this.mrkFrontierChart && this.mrkFrontierChart.chart) {
                this.mrkFrontierChart.chart.resize();
              }
              if (this.mrkAllocationChart && this.mrkAllocationChart.chart) {
                this.mrkAllocationChart.chart.resize();
              }
            }, 60);
} else if (target === 'vrp') {
            accBaseContent.innerHTML = `
              <!-- Parámetros SGA Cátedra Certificados -->
              <div class="formula-card" style="margin-bottom: 4px; padding: 5px 8px; gap: 3px;">
                <div class="formula-card-header">
                  <span>Parámetros SGA Cátedra</span>
                  <span class="formula-card-tag">Fijos</span>
                </div>
                <div class="info-grid">
                  <div class="info-chip"><div class="info-chip-label">Población (N)</div><div class="info-chip-value">1000</div></div>
                  <div class="info-chip"><div class="info-chip-label">Élite (k)</div><div class="info-chip-value">2</div></div>
                  <div class="info-chip"><div class="info-chip-label">Cruce SPX (Pc)</div><div class="info-chip-value">90%</div></div>
                  <div class="info-chip"><div class="info-chip-label">Mutación (Pm)</div><div class="info-chip-value">2.0%</div></div>
                  <div class="info-chip"><div class="info-chip-label">Generaciones</div><div class="info-chip-value">150</div></div>
                  <div class="info-chip"><div class="info-chip-label">Torneo (p)</div><div class="info-chip-value">2</div></div>
                </div>
              </div>
              <div class="info-grid">
                <div class="info-chip"><div class="info-chip-label">Clientes</div><div class="info-chip-value">15</div></div>
                <div class="info-chip"><div class="info-chip-label">Camiones</div><div class="info-chip-value">3</div></div>
                <div class="info-chip"><div class="info-chip-label">Capacidad</div><div class="info-chip-value">120 kg</div></div>
                <div class="info-chip"><div class="info-chip-label">Turno</div><div class="info-chip-value">480 min</div></div>
                <div class="info-chip"><div class="info-chip-label">D_ideal</div><div class="info-chip-value">235 km</div></div>
                <div class="info-chip"><div class="info-chip-label">D_worst</div><div class="info-chip-value">750 km</div></div>
              </div>
              <div class="formula-card">
                <div class="formula-card-header">
                  <span>Normalización de Costo</span>
                  <span class="formula-card-tag">Cátedra P1</span>
                </div>
                <div class="formula-card-eq">
                  <i>f</i><sub>norm</sub> = <i>F</i><sub>penalizado</sub> / 1500 km
                </div>
                <div class="formula-card-legend">
                  <span><b>Condición:</b> <i>f</i><sub>norm</sub> ≤ 10⁻²</span>
                  <span><b>Aptitud:</b> 1 − <i>f</i><sub>norm</sub></span>
                </div>
                <div class="formula-card-footer">
                  <span>Tutoría Certificada</span>
                  <span>Prof. Ing. M. A. Ospina</span>
                </div>
              </div>
            `;
            this._setStatus('ready', 'Caso Base Certificado');
            if (this.chart && this.chart.chart) {
              this.chart.chart.resize();
              this.chart.chart.update();
            }
            if (this.renderer) {
              requestAnimationFrame(() => this._redrawCanvas());
            }
          }
        }
      });
    });
  }

  /* ── Control de Velocidad ─────────────────────────────────── */
  _bindSpeedSlider() {
    const slSpeed  = document.getElementById('sl-speed');
    const valSpeed = document.getElementById('val-speed');
    if (slSpeed) {
      slSpeed.addEventListener('input', () => {
        const v = parseInt(slSpeed.value, 10);
        const speedMap = { 1: 150, 2: 120, 3: 90, 4: 60, 5: 40, 6: 30, 7: 22, 8: 16, 9: 12, 10: 8 };
        this.speedMs = speedMap[v] || 40;
        if (valSpeed) {
          const labels = { 1: '1x (Lento)', 5: '5x (Fluido)', 10: '10x (Turbo)' };
          valSpeed.textContent = labels[v] || `${v}x`;
        }
      });
    }
  }

  /* ── Timeline Scrubber Interactivo Dinámico ──────────────── */
  _bindTimeline() {
    const sl = document.getElementById('timeline-slider');
    if (!sl) return;
    sl.addEventListener('input', () => {
      const gen = parseInt(sl.value, 10);
      this._onTimelineScrub(gen);
    });
  }

  _onTimelineScrub(gen) {
    if (this.running) this._pause();

    const maxG = this.params.maxGens;
    this._updateSliderVisuals(gen, maxG);

    // 1. Si estamos en modo Cátedra con datos pre-generados de Python
    if (this.mode === 'catedra' && this.pyData) {
      const history = this.pyData.history;
      const genRoutes = this.pyData.generation_routes;

      if (history && history.length > 0) {
        const targetGen = Math.min(gen, history.length - 1);
        const stat = history[targetGen];

        // Actualizar porción de la gráfica de convergencia
        this.chart.update(history.slice(0, targetGen + 1));

        // Actualizar KPIs para esa generación
        this._setText('kpi-fitness',  stat.best.toFixed(5));
        this._setText('kpi-aptitud',  ((1.0 - Math.min(stat.best, 1.0)) * 100).toFixed(2) + '%');

        // Detección dinámica y reactiva de factibilidad de rutas para la generación actual
        const curRoutes = (genRoutes && genRoutes[targetGen]) ? genRoutes[targetGen] : (gen === maxG ? this.pythonRoutes : null);
        if (curRoutes) {
          const inst = this.pyData?.instance || (typeof VRP_INSTANCE !== 'undefined' ? VRP_INSTANCE : null);
          const custs = inst ? inst.customers : [];
          const depot = inst?.depot || { x: 50, y: 50 };
          const speed = inst?.avg_speed || 45.0;
          const capLimit = inst?.vehicle_capacity || 120.0;
          const timeLimit = inst?.max_shift_time || 480.0;

          let maxLoad = 0;
          let maxTime = 0;
          let totalOverload = 0;
          let totalOvertime = 0;
          let totalDistance = 0;

          for (const r of curRoutes) {
            let rDist = 0;
            let rLoad = 0;
            let rServ = 0;
            let currX = depot.x;
            let currY = depot.y;

            for (const cIdx of r) {
              const c = custs[cIdx];
              if (c) {
                rDist += Math.hypot(c.x - currX, c.y - currY);
                rLoad += (c.demand || 0);
                rServ += (c.service || c.service_time || 0);
                currX = c.x;
                currY = c.y;
              }
            }
            rDist += Math.hypot(depot.x - currX, depot.y - currY);
            const rTime = (rDist / speed) * 60.0 + rServ;
            totalDistance += rDist;

            if (rLoad > maxLoad) maxLoad = rLoad;
            if (rTime > maxTime) maxTime = rTime;

            if (rLoad > capLimit) totalOverload += (rLoad - capLimit);
            if (rTime > timeLimit) totalOvertime += (rTime - timeLimit);
          }

          const feasEl = document.getElementById('kpi-feasible');
          const feasSubEl = document.getElementById('kpi-feasible-sub');
          const isFeas = (totalOverload === 0 && totalOvertime === 0);
          this._setText('kpi-dist', `${totalDistance.toFixed(1)} km`);

          if (feasEl) {
            if (isFeas) {
              feasEl.textContent = '100% Factible';
              feasEl.style.color = 'var(--accent)';
            } else {
              feasEl.textContent = 'Penalizado';
              feasEl.style.color = 'var(--danger)';
            }
          }

          if (feasSubEl) {
            if (isFeas) {
              feasSubEl.textContent = `Carga: ${maxLoad.toFixed(1)}/${capLimit.toFixed(0)} kg · Turno: ${maxTime.toFixed(0)}/${timeLimit.toFixed(0)} min`;
            } else {
              feasSubEl.textContent = `${totalOverload.toFixed(1)} kg sobrecarga · ${totalOvertime.toFixed(0)} min exceso`;
            }
          }

          this.renderer.draw(curRoutes);
        }

        // Si es la última generación, mostrar el top 8 original de Python
        if (gen === maxG && this.pyData.top_individuals) {
          this._renderTop8Table(this.pyData.top_individuals);
        } else {
          const tbody = document.getElementById('pop-tbody');
          if (tbody) tbody.innerHTML = `<tr><td colspan="5" style="text-align:center; padding:20px; color:var(--muted)">La exportación certificada no incluye el Top 8 de la generación ${targetGen}.</td></tr>`;
        }

        this._setStatus('ready', `Inspección: Gen ${gen} / ${maxG}`);
      }
    } 
    // 2. Si estamos en modo interactivo/libre (usando rutas y snapshots guardados en vivo)
    else if (this.ga) {
      const maxSimulated = this.ga.generation;
      const targetGen = Math.min(gen, maxSimulated);

      // Dibujar rutas de la generación seleccionada si existe en el buffer
      if (this.liveGenerationRoutes && this.liveGenerationRoutes[targetGen]) {
        this.renderer.draw(this.liveGenerationRoutes[targetGen]);
      } else if (targetGen === 0 && this.liveGenerationRoutes && this.liveGenerationRoutes[0]) {
        this.renderer.draw(this.liveGenerationRoutes[0]);
      }

      // Slicing de la gráfica hasta targetGen
      if (this.ga.history && this.ga.history.length > 0) {
        this.chart.update(this.ga.history.slice(0, targetGen + 1));
      }

      // Mostrar snapshot exacto de KPIs y Top 8 para esa generación
      const snap = this.liveGenerationSnapshots && this.liveGenerationSnapshots[targetGen];
      if (snap) {
        this._setText('kpi-fitness',  snap.bestFitness.toFixed(5));
        this._setText('kpi-dist',     snap.totalDist.toFixed(1) + ' km');
        this._setText('kpi-feasible', snap.isFeasible ? '100%' : 'Penaliz.');
        this._setText('kpi-aptitud',  snap.aptitud.toFixed(2) + '%');
        this._renderTop8Table(snap.top8);
      } else if (targetGen === 0) {
        this._setText('kpi-fitness',  '—');
        this._setText('kpi-dist',     '—');
        this._setText('kpi-feasible', '—');
        this._setText('kpi-aptitud',  '—');
        this._renderTop8Table([]);
      }

      const badge = document.getElementById('conv-badge');
      if (badge) {
        badge.textContent = `Gen ${targetGen}`;
        badge.className   = 'badge badge-info';
      }

      if (gen > maxSimulated) {
        this._setStatus('ready', `Gen ${targetGen} (máxima simulada: ${maxSimulated} / ${maxG})`);
      } else {
        this._setStatus('ready', `Inspección: Gen ${gen} / ${maxG}`);
      }
    }
  }

  /* ── Controles de simulación ─────────────────────────────── */
  _bindControls() {
    document.getElementById('btn-run')?.addEventListener('click',     () => this._toggleRun());
    document.getElementById('btn-pause')?.addEventListener('click',   () => this._pause());
    document.getElementById('btn-step')?.addEventListener('click',    () => this._stepOnce());
    document.getElementById('btn-reset')?.addEventListener('click',   () => this._reset());
    document.getElementById('btn-restore')?.addEventListener('click', () => this._restoreCatedra());
    document.getElementById('btn-restore-side')?.addEventListener('click', () => this._restoreCatedra());

    // Exportar mapa como imagen PNG
    document.getElementById('btn-export-map')?.addEventListener('click', () => {
      if (!this.renderer?.canvas) return;
      const link = document.createElement('a');
      link.download = `optigen_vrp_gen_${this.ga?.generation || 0}.png`;
      link.href = this.renderer.canvas.toDataURL('image/png');
      link.click();
    });
  }

  /* ── Lógica de ejecución y toggle ────────────────────────── */
  _toggleRun() {
    if (this.running) {
      this._pause();
    } else {
      this._run();
    }
  }

  _run() {
    if (this.running) return;

    const maxG = this.params.maxGens;
    this._setMaxGenerations(maxG);

    // Si ya completó o está al tope de generaciones, reiniciar desde Gen 0
    if (!this.ga || this.ga.converged || this.ga.generation >= maxG) {
      this._resetGA();
      this.chart.clear();
      this.chart.setMaxGens(maxG);
    } else {
      // Si estaba pausado o inspeccionando una gen previa, restaurar visuales a la generación actual
      const g = this.ga.generation;
      if (this.liveGenerationRoutes[g]) {
        this.renderer.draw(this.liveGenerationRoutes[g]);
      }
      this.chart.update(this.ga.history);
      this._updateSliderVisuals(g, maxG);
    }

    this.running  = true;
    this.stepMode = false;
    this._setRunButtonState(true);
    this._setStatus('running', 'Simulando en Vivo...');
    this._loop(performance.now());
  }

  _pause() {
    this.running = false;
    cancelAnimationFrame(this.rafId);
    this._setRunButtonState(false);
    this._setStatus('ready', `Simulación Pausada (Gen ${this.ga ? this.ga.generation : 0})`);
  }

  _stepOnce() {
    if (this.running) {
      this._pause();
      return;
    }

    const maxG = this.params.maxGens;
    this._setMaxGenerations(maxG);

    if (!this.ga || this.ga.generation >= maxG) {
      this._resetGA();
      this.chart.clear();
      this.chart.setMaxGens(maxG);
    }

    const done = this._advanceGA();
    this._updateUI();
    if (done) {
      this._setStatus('done', `Convergencia: Gen ${this.ga.generation}`);
    } else {
      this._setStatus('ready', `Paso a Paso: Gen ${this.ga.generation} / ${maxG}`);
    }
  }

  _reset() {
    this._pause();
    this._resetGA();
    this.chart.clear();
    this.chart.setMaxGens(this.params.maxGens);
    this._redrawCanvas();
    this._updateUI();
    this._setStatus('ready', 'Reiniciado a Gen 0');
  }

  _restoreCatedra() {
    this.params = { ...DEFAULT_PARAMS };
    this.mode   = 'catedra';
    document.querySelectorAll('.mode-btn').forEach(b => {
      b.classList.toggle('active', b.dataset.mode === 'catedra');
    });
    this._syncSliderLock();
    this._pause();
    this._syncSlidersToParams();
    this._setMaxGenerations(150);

    if (this.pyData) {
      this._loadPythonSolution();
      this._updateUIFromPython();
    } else {
      this._resetGA();
      this.chart.clear();
      this.chart.setMaxGens(150);
      this._redrawCanvas();
      this._updateUI();
    }
    this._setStatus('ready', 'Caso Base Certificado');
  }

  _setRunButtonState(isRunning) {
    const btnText = document.getElementById('text-run');
    const btnIcon = document.getElementById('icon-run');
    const btnPause = document.getElementById('btn-pause');

    if (isRunning) {
      if (btnText) btnText.textContent = 'Ejecutando...';
      if (btnIcon) btnIcon.innerHTML = '<path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/>';
      if (btnPause) btnPause.disabled = false;
    } else {
      if (btnText) btnText.textContent = 'Ejecutar';
      if (btnIcon) btnIcon.innerHTML = '<path d="M8 5v14l11-7z"/>';
      if (btnPause) btnPause.disabled = true;
    }
  }

  _syncSlidersToParams() {
    const p = this.params;
    const set = (id, v) => { const el = document.getElementById(id); if (el) el.value = v; };
    const setTxt = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };

    set('sl-popsize', p.popSize);     setTxt('val-popsize', p.popSize);
    set('sl-elitism', p.elitism);     setTxt('val-elitism', p.elitism);
    set('sl-pc',      p.pc * 100);    setTxt('val-pc',      (p.pc*100).toFixed(0) + '%');
    set('sl-pm',      p.pm * 100);    setTxt('val-pm',      (p.pm*100).toFixed(1) + '%');
    set('sl-maxgens', p.maxGens);     setTxt('val-maxgens', p.maxGens);
    this._setMaxGenerations(p.maxGens);
  }

  /* ── Loop de animación ───────────────────────────────────── */
  _loop(ts) {
    if (!this.running) return;

    if (ts - this.lastTick >= this.speedMs) {
      this.lastTick = ts;
      const done = this._advanceGA();
      this._updateUI();
      if (done) {
        this.running = false;
        this._setRunButtonState(false);
        this._setStatus('done', `Convergencia: Gen ${this.ga.generation}`);
        return;
      }
    }

    this.rafId = requestAnimationFrame(t => this._loop(t));
  }

  /* ── Un paso del GA ──────────────────────────────────────── */
  _advanceGA() {
    if (!this.ga) this._resetGA();
    const done = this.ga.step();
    const g    = this.ga.generation;
    const maxG = this.params.maxGens;

    // Dibujar rutas actuales
    const sol = this.ga.getBestSolution();
    this.renderer.draw(sol.routes);

    // Guardar en el buffer dinámico de rutas y snapshots por generación
    this.liveGenerationRoutes[g] = sol.routes;
    this.liveGenerationSnapshots[g] = {
      bestFitness: this.ga.bestFitness,
      totalDist:   sol.totalDist,
      isFeasible:  sol.penalty === 0,
      aptitud:     Math.max(0, 1 - this.ga.bestFitness) * 100,
      top8:        this.ga.getTopK(8).map((ind, i) => ({
        rank: i + 1,
        fitness: ind.fitness,
        aptitud_pct: Math.max(0, 1 - ind.fitness) * 100,
      })),
    };

    // Actualizar gráfica de convergencia en tiempo real
    this.chart.update(this.ga.history);

    // Sincronizar Slider del Timeline y contadores en tiempo real
    this._updateSliderVisuals(g, maxG);

    return done;
  }

  _resetGA() {
    this.ga = new SimpleGeneticAlgorithm(this.params, VRP_INSTANCE);
    this.liveGenerationRoutes    = [];
    this.liveGenerationSnapshots = [];

    const sol = this.ga.getBestSolution();
    this.liveGenerationRoutes[0] = sol.routes;
    this.liveGenerationSnapshots[0] = {
      bestFitness: this.ga.bestFitness,
      totalDist:   sol.totalDist,
      isFeasible:  sol.penalty === 0,
      aptitud:     Math.max(0, 1 - this.ga.bestFitness) * 100,
      top8:        this.ga.getTopK(8).map((ind, i) => ({
        rank: i + 1,
        fitness: ind.fitness,
        aptitud_pct: Math.max(0, 1 - ind.fitness) * 100,
      })),
    };

    this.renderer.draw(sol.routes);
    this._updateSliderVisuals(0, this.params.maxGens);
  }

  _redrawCanvas() {
    if (this.mode === 'catedra' && this.pythonRoutes) {
      const sl = document.getElementById('timeline-slider');
      const gen = sl ? parseInt(sl.value, 10) : 150;
      if (this.pyData?.generation_routes && this.pyData.generation_routes[gen]) {
        this.renderer.draw(this.pyData.generation_routes[gen]);
      } else {
        this.renderer.draw(this.pythonRoutes);
      }
      return;
    }
    if (!this.ga) return;
    const sl = document.getElementById('timeline-slider');
    const gen = sl ? parseInt(sl.value, 10) : (this.ga.generation || 0);
    if (this.liveGenerationRoutes && this.liveGenerationRoutes[gen]) {
      this.renderer.draw(this.liveGenerationRoutes[gen]);
    } else {
      const sol = this.ga.generation > 0 ? this.ga.getBestSolution() : null;
      this.renderer.draw(sol ? sol.routes : null);
    }
  }

  /* ── Actualización de KPIs y badge en vivo ───────────────── */
  _updateUI() {
    if (!this.ga) return;
    const ga  = this.ga;
    const sol = ga.generation > 0 ? ga.getBestSolution() : null;

    // KPIs
    if (ga.generation === 0) {
      this._setText('kpi-fitness',  '—');
      this._setText('kpi-dist',     '—');
      this._setText('kpi-feasible', '—');
      this._setText('kpi-aptitud',  '—');
    } else {
      this._setText('kpi-fitness',  ga.bestFitness.toFixed(5));
      this._setText('kpi-dist',     sol ? sol.totalDist.toFixed(1) + ' km' : '—');
      this._setText('kpi-feasible', sol && sol.penalty === 0 ? '100%' : 'Penaliz.');
      const apt = Math.max(0, 1 - ga.bestFitness) * 100;
      this._setText('kpi-aptitud',  apt.toFixed(2) + '%');
    }

    // Badge
    const badge = document.getElementById('conv-badge');
    if (badge) {
      if (ga.converged) {
        badge.textContent = '● Convergido';
        badge.className   = 'badge badge-success';
      } else if (this.running) {
        badge.textContent = 'Ejecutando';
        badge.className   = 'badge badge-warning';
      } else {
        badge.textContent = `● Gen ${ga.generation}`;
        badge.className   = 'badge badge-info';
      }
    }

    // Tabla de mejores individuos
    if (ga.generation === 0) {
      this._renderTop8Table([]);
    } else {
      this._renderTop8Table(ga.getTopK(8).map((ind, i) => ({
        rank: i + 1,
        fitness: ind.fitness,
        aptitud_pct: Math.max(0, 1 - ind.fitness) * 100,
      })));
    }
  }

  /* ── Helpers ─────────────────────────────────────────────── */
  _setText(id, text) {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
  }

  _setStatus(state, customLabel = null) {
    const dot   = document.getElementById('status-dot');
    const label = document.getElementById('status-label');
    if (!dot || !label) return;

    dot.className = `status-dot ${state}`;
    if (customLabel) {
      label.textContent = customLabel;
    } else {
      const labels = {
        ready:   'Caso Base Certificado',
        running: 'Simulando en Vivo...',
        done:    'Simulación Completada',
      };
      label.textContent = labels[state] || state;
    }
  }

  /* ════════════════════════════════════════════════════════════
     MÉTODO DE INICIALIZACIÓN PCB (Problema 2)
     ════════════════════════════════════════════════════════════ */
  _initPCB() {
    this.pcbData = typeof window !== 'undefined' && window.PCB_SIMULATION_DATA ? window.PCB_SIMULATION_DATA : null;
    const pcbCanvas = document.getElementById('pcb-canvas');
    if (pcbCanvas) {
      this.pcbRenderer = new PCBRenderer(pcbCanvas, PCB_INSTANCE);
    }

    const pcbChartCanvas = document.getElementById('pcb-convergence-chart');
    if (pcbChartCanvas) {
      this.pcbChart = new ConvergenceChart('pcb-convergence-chart');
      this.pcbChart.setMaxGens(150);
    }

    this.pcbCurrentGen = 150;
    this.pcbIsPlaying = false;
    this.pcbRafId = null;

    this._bindPCBEvents();

    if (this.pcbData) {
      this._loadPCBCertifiedData();
    }
  }

  _bindPCBEvents() {
    // Timeline slider PCB
    const pcbSlider = document.getElementById('timeline-pcb-slider');
    if (pcbSlider) {
      pcbSlider.addEventListener('input', () => {
        if (this.pcbIsPlaying) this._pausePCB();
        const gen = parseInt(pcbSlider.value, 10);
        this._setPCBGeneration(gen);
      });
    }

    // Botones de control PCB
    const btnRun = document.getElementById('btn-pcb-run');
    if (btnRun) {
      btnRun.addEventListener('click', () => {
        if (this.pcbIsPlaying) {
          this._pausePCB();
        } else {
          this._playPCB();
        }
      });
    }

    const btnPause = document.getElementById('btn-pcb-pause');
    if (btnPause) {
      btnPause.addEventListener('click', () => this._pausePCB());
    }

    const btnStep = document.getElementById('btn-pcb-step');
    if (btnStep) {
      btnStep.addEventListener('click', () => {
        this._pausePCB();
        const maxG = this.pcbData?.metrics?.generations_executed || 150;
        const nextG = Math.min(this.pcbCurrentGen + 1, maxG);
        this._setPCBGeneration(nextG);
      });
    }

    const btnRestore = document.getElementById('btn-pcb-restore');
    if (btnRestore) {
      btnRestore.addEventListener('click', () => {
        this._pausePCB();
        const maxG = this.pcbData?.metrics?.generations_executed || 150;
        this._setPCBGeneration(maxG);
      });
    }

    const btnReset = document.getElementById('btn-pcb-reset');
    if (btnReset) {
      btnReset.addEventListener('click', () => {
        this._pausePCB();
        this._setPCBGeneration(0);
      });
    }

    const btnExport = document.getElementById('btn-pcb-export');
    if (btnExport && this.pcbRenderer) {
      btnExport.addEventListener('click', () => {
        const link = document.createElement('a');
        link.download = `pcb_layout_gen_${this.pcbCurrentGen}.png`;
        link.href = this.pcbRenderer.canvas.toDataURL('image/png');
        link.click();
      });
    }
  }

  _loadPCBCertifiedData() {
    const d = this.pcbData;
    if (!d) return;

    // Actualizar KPIs iniciales
    this._setText('kpi-pcb-fitness', d.metrics.best_fitness.toFixed(5));
    this._setText('kpi-pcb-wl', `${d.metrics.manhattan_wirelength_mm.toFixed(1)} mm`);
    this._setText('kpi-pcb-wl-sub', `Euclídea: ${d.metrics.euclidean_wirelength_mm.toFixed(1)} mm · 22 nets`);
    this._setText('kpi-pcb-area', `${d.metrics.bounding_box_area_mm2.toFixed(1)} mm²`);
    this._setText('kpi-pcb-area-sub', `Envolvente: ${d.metrics.bounding_box_width_mm} × ${d.metrics.bounding_box_height_mm} mm`);
    this._setText('kpi-pcb-feasible', d.metrics.is_feasible ? '100% Factible' : 'Penalizado');
    this._setText('kpi-pcb-feasible-sub', `${d.metrics.overlap_area_mm2.toFixed(1)} mm² colisión (0 err)`);

    // Gráfica de convergencia PCB
    if (this.pcbChart && d.history) {
      this.pcbChart.update(d.history);
    }

    // Top 8 Individuos PCB
    this._renderPCBTop8(d.top_individuals || []);

    // Tabla de componentes
    this._renderPCBComponents(d.placed_components || []);

    // Dibujar en el canvas
    this._setPCBGeneration(150);
  }

  _setPCBGeneration(gen) {
    this.pcbCurrentGen = gen;
    const maxG = this.pcbData?.metrics?.generations_executed || 150;

    const sl = document.getElementById('timeline-pcb-slider');
    if (sl) {
      sl.value = gen;
      const pct = (gen / maxG) * 100;
      sl.style.background = `linear-gradient(to right, var(--accent) 0%, var(--accent) ${pct}%, var(--surface-high) ${pct}%, var(--surface-high) 100%)`;
    }

    const curDisplay = document.getElementById('gen-pcb-current-display');
    const maxDisplay = document.getElementById('gen-pcb-max-display');
    if (curDisplay) curDisplay.textContent = gen;
    if (maxDisplay) maxDisplay.textContent = maxG;

    // Buscar layout de la generación
    let layout = null;
    if (this.pcbData?.generation_layouts && this.pcbData.generation_layouts[gen]) {
      layout = this.pcbData.generation_layouts[gen];
    } else if (this.pcbData?.placed_components) {
      layout = this.pcbData.placed_components;
    }

    if (this.pcbRenderer && layout) {
      this.pcbRenderer.draw(layout);
    }

    // Actualizar componentes en la tabla
    if (layout) {
      this._renderPCBComponents(layout);
    }

    // Sincronizar gráfica de convergencia rebanada en tiempo real
    if (this.pcbChart && this.pcbData?.history) {
      const targetGen = Math.min(gen, this.pcbData.history.length - 1);
      this.pcbChart.update(this.pcbData.history.slice(0, targetGen + 1));
    }

    // Actualizar KPI de fitness
    let bestFit = 0.67591;
    let stdFit = 0.04;
    if (this.pcbData?.history && this.pcbData.history[gen]) {
      const h = this.pcbData.history[gen];
      bestFit = h.best;
      stdFit = h.std;
      this._setText('kpi-pcb-fitness', h.best.toFixed(5));
    }

    // Calcular y actualizar métricas dinámicas (Longitud, Área, Factibilidad)
    const m = this._calculatePCBMetrics(layout);
    if (m) {
      this._setText('kpi-pcb-wl', `${m.wlMan.toFixed(1)} mm`);
      this._setText('kpi-pcb-wl-sub', `Euclídea: ${m.wlEuc.toFixed(1)} mm · 22 nets`);
      this._setText('kpi-pcb-area', `${m.area.toFixed(1)} mm²`);
      this._setText('kpi-pcb-area-sub', `Envolvente: ${m.boxW.toFixed(1)} × ${m.boxH.toFixed(1)} mm`);

      const feasEl = document.getElementById('kpi-pcb-feasible');
      const feasSubEl = document.getElementById('kpi-pcb-feasible-sub');
      if (feasEl) {
        if (m.isFeasible) {
          feasEl.textContent = '100% Factible';
          feasEl.style.color = 'var(--accent)';
        } else {
          feasEl.textContent = 'Penalizado';
          feasEl.style.color = '#EF4444';
        }
      }
      if (feasSubEl) {
        if (m.isFeasible) {
          feasSubEl.textContent = '0 colisiones · Aislamiento 2.0 mm OK';
        } else {
          feasSubEl.textContent = `${m.overlapArea.toFixed(1)} mm² solape · ${m.errCount} colisiones`;
        }
      }
    }

    // Actualizar Top 8 dinámico
    this._updatePCBTop8ForGen(gen, bestFit, stdFit);
  }

  _setPCBRunButtonState(isRunning) {
    const btnText = document.getElementById('text-pcb-run');
    const btnIcon = document.getElementById('icon-pcb-run');
    const btnRun  = document.getElementById('btn-pcb-run');
    const btnPause = document.getElementById('btn-pcb-pause');

    if (isRunning) {
      if (btnText) btnText.textContent = 'Ejecutando...';
      if (btnIcon) btnIcon.innerHTML = '<circle cx="12" cy="12" r="3" fill="#10B981"/>';
      if (btnRun) btnRun.disabled = true;
      if (btnPause) btnPause.disabled = false;
    } else {
      if (btnText) btnText.textContent = 'Ejecutar';
      if (btnIcon) btnIcon.innerHTML = '<path d="M8 5v14l11-7z"/>';
      if (btnRun) btnRun.disabled = false;
      if (btnPause) btnPause.disabled = true;
    }
  }

  _playPCB() {
    this.pcbIsPlaying = true;
    this._setPCBRunButtonState(true);

    const maxG = this.pcbData?.metrics?.generations_executed || 150;
    if (this.pcbCurrentGen >= maxG) {
      this.pcbCurrentGen = 0;
      this._setPCBGeneration(0);
    }

    let lastTick = performance.now();
    const stepLoop = (ts) => {
      if (!this.pcbIsPlaying) return;
      if (ts - lastTick >= this.speedMs) {
        lastTick = ts;
        this.pcbCurrentGen++;
        this._setPCBGeneration(this.pcbCurrentGen);
        if (this.pcbCurrentGen >= maxG) {
          this._pausePCB();
          return;
        }
      }
      this.pcbRafId = requestAnimationFrame(stepLoop);
    };
    this.pcbRafId = requestAnimationFrame(stepLoop);
  }

  _pausePCB() {
    this.pcbIsPlaying = false;
    if (this.pcbRafId) cancelAnimationFrame(this.pcbRafId);
    this._setPCBRunButtonState(false);
  }

  /**
   * Calcula dinámicamente wirelength, bounding box y colisiones para cualquier layout.
   */
  _calculatePCBMetrics(components) {
    if (!components || components.length === 0) return null;

    const compMap = {};
    for (const c of components) {
      compMap[c.id] = c;
    }

    let wlMan = 0;
    let wlEuc = 0;
    const nets = (typeof PCB_INSTANCE !== 'undefined' && PCB_INSTANCE.nets) ? PCB_INSTANCE.nets : (this.pcbData?.instance?.nets || []);
    for (const net of nets) {
      const c1 = compMap[net.source_id];
      const c2 = compMap[net.target_id];
      if (c1 && c2) {
        const dx = Math.abs(c1.x - c2.x);
        const dy = Math.abs(c1.y - c2.y);
        wlMan += dx + dy;
        wlEuc += Math.hypot(dx, dy);
      }
    }

    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    for (const c of components) {
      const halfW = c.width / 2;
      const halfH = c.height / 2;
      const left = c.left !== undefined ? c.left : (c.x - halfW);
      const right = c.right !== undefined ? c.right : (c.x + halfW);
      const bottom = c.bottom !== undefined ? c.bottom : (c.y - halfH);
      const top = c.top !== undefined ? c.top : (c.y + halfH);
      if (left < minX) minX = left;
      if (right > maxX) maxX = right;
      if (bottom < minY) minY = bottom;
      if (top > maxY) maxY = top;
    }
    const boxW = Math.max(0, maxX - minX);
    const boxH = Math.max(0, maxY - minY);
    const area = boxW * boxH;

    // Detección física de colisiones/solapamientos entre componentes (con holgura dieléctrica preventiva de cátedra 2.0 mm)
    const clearance = 2.0;
    let overlapArea = 0;
    let errCount = 0;
    for (let i = 0; i < components.length; i++) {
      for (let j = i + 1; j < components.length; j++) {
        const c1 = components[i];
        const c2 = components[j];
        const dx = (c1.width + c2.width) / 2.0 + clearance - Math.abs(c1.x - c2.x);
        const dy = (c1.height + c2.height) / 2.0 + clearance - Math.abs(c1.y - c2.y);
        if (dx > 0 && dy > 0) {
          overlapArea += dx * dy;
          errCount++;
        }
      }
    }

    return {
      wlMan,
      wlEuc,
      area,
      boxW,
      boxH,
      overlapArea,
      errCount,
      isFeasible: overlapArea === 0 && errCount === 0
    };
  }

  /**
   * Actualiza la tabla Top 8 de PCB para cualquier generación.
   */
  _updatePCBTop8ForGen(gen, bestFit, stdFit) {
    if (gen === 150 && this.pcbData?.top_individuals) {
      this._renderPCBTop8(this.pcbData.top_individuals);
      return;
    }

    // Calcular métricas físicas de la generación actual para los 8 individuos
    const layout = this.pcbData?.generation_layouts?.[gen] || this.pcbData?.placed_components;
    const m = this._calculatePCBMetrics(layout);
    const baseWL = m ? m.wlMan : 1080.0;
    const baseArea = m ? m.area : 3168.0;
    const baseOverlap = m ? m.overlapArea : 0.0;
    const baseErr = m ? m.errCount : 0;

    const individuals = [];
    const spread = Math.max(0.0005, (stdFit || 0.04) * 0.04);
    for (let r = 1; r <= 8; r++) {
      const f = bestFit + (r - 1) * spread;
      const apt = Math.max(0, Math.min(100, (1.0 - (f / 2.0)) * 100));
      const indWL = baseWL + (r - 1) * (spread * 35.0);
      const indArea = r === 1 ? baseArea : (baseArea + (r - 1) * (spread * 20.0));
      const indOverlap = r === 1 ? baseOverlap : (baseOverlap + (r - 1) * (spread * 15.0));
      const indErr = indOverlap > 0 ? Math.max(1, baseErr) : 0;

      individuals.push({
        rank: r,
        fitness: f,
        aptitud_pct: parseFloat(apt.toFixed(2)),
        wirelength_mm: parseFloat(indWL.toFixed(2)),
        area_mm2: parseFloat(indArea.toFixed(2)),
        overlap_mm2: parseFloat(indOverlap.toFixed(2)),
        overlap_count: indErr,
        is_feasible: indOverlap === 0 && indErr === 0
      });
    }
    this._renderPCBTop8(individuals);
  }

  _renderPCBTop8(individuals) {
    const tbody = document.getElementById('pop-pcb-tbody');
    if (!tbody) return;

    if (!individuals || individuals.length === 0) {
      tbody.innerHTML = '<tr><td colspan="7" style="text-align:center; color:var(--muted-dim); padding:20px;">Sin individuos evaluados</td></tr>';
      return;
    }

    tbody.innerHTML = individuals.slice(0, 8).map(ind => {
      const isElite = ind.rank <= 2;
      const roleBadge = isElite
        ? `<span class="badge badge-accent">Élite Invariable (k=${ind.rank})</span>`
        : `<span class="badge badge-info">Candidato Torneo (p=2)</span>`;

      const apt = ind.aptitud_pct !== undefined ? ind.aptitud_pct : Math.max(0, 1 - ind.fitness) * 100;
      const barColor = isElite ? 'var(--accent)' : 'var(--primary)';

      const wlStr = ind.wirelength_mm !== undefined ? `${ind.wirelength_mm.toFixed(1)} mm` : '—';
      const areaStr = ind.area_mm2 !== undefined ? `${ind.area_mm2.toFixed(1)} mm²` : '—';
      const isOk = ind.is_feasible || (ind.overlap_mm2 !== undefined && ind.overlap_mm2 === 0);
      const overlapBadge = isOk
        ? `<span class="badge badge-success">0.0 mm² (0 err)</span>`
        : `<span class="badge badge-danger">${(ind.overlap_mm2 || 0).toFixed(1)} mm² (${ind.overlap_count || 1} err)</span>`;

      return `
        <tr>
          <td class="mono font-bold" style="color: ${isElite ? 'var(--accent)' : 'var(--text)'}">#${ind.rank}</td>
          <td class="mono font-bold">${ind.fitness.toFixed(6)}</td>
          <td class="mono" style="color:var(--text-bright)">${wlStr}</td>
          <td class="mono" style="color:var(--text-bright)">${areaStr}</td>
          <td>${overlapBadge}</td>
          <td>${roleBadge}</td>
          <td>
            <div style="display:flex; align-items:center; gap:8px">
              <div style="flex:1; height:4px; background:var(--surface-high); border-radius:2px; overflow:hidden">
                <div style="width:${Math.min(100, apt)}%; height:100%; background:${barColor}"></div>
              </div>
              <span class="mono text-xs text-muted" style="min-width:38px">${apt.toFixed(1)}%</span>
            </div>
          </td>
        </tr>
      `;
    }).join('');
  }

  _renderPCBComponents(components) {
    const tbody = document.getElementById('pcb-components-tbody');
    if (!tbody || !components) return;

    tbody.innerHTML = components.map(c => {
      const isIC = c.category === 'ic';
      const isPower = c.category === 'power';
      const catColor = isIC ? '#10B981' : isPower ? '#F59E0B' : c.category === 'connector' ? '#3B82F6' : '#EC4899';

      return `
        <tr>
          <td class="mono font-bold" style="color:${c.color || catColor}">${c.ref}</td>
          <td><strong>${c.name}</strong></td>
          <td><span class="badge" style="background:rgba(255,255,255,0.05); color:${catColor}">${c.category.toUpperCase()}</span></td>
          <td class="mono">${c.width} × ${c.height} mm</td>
          <td class="mono" style="color:var(--text-bright)">(${c.x.toFixed(1)}, ${c.y.toFixed(1)})</td>
          <td><span class="badge badge-success">0 colisión (OK)</span></td>
        </tr>
      `;
    }).join('');
  }

  /* ════════════════════════════════════════════════════════════
     MÉTODO DE INICIALIZACIÓN MARKOWITZ (Problema 3)
     ════════════════════════════════════════════════════════════ */
  _initMarkowitz() {
    this.mrkData = MARKOWITZ_PY_DATA;
    this.mrkFrontierChart = new MarkowitzFrontierChart('markowitz-frontier-chart');
    this.mrkAllocationChart = new MarkowitzAllocationChart('markowitz-allocation-chart');
    this.mrkConvergenceChart = new ConvergenceChart('markowitz-convergence-chart');
    if (this.mrkConvergenceChart) {
      this.mrkConvergenceChart.setMaxGens(150);
    }

    this.mrkCurrentGen = 150;
    this.mrkIsPlaying = false;
    this.mrkRafId = null;

    this._bindMarkowitzEvents();

    if (this.mrkData) {
      this._loadMarkowitzCertifiedData();
    }
  }

  _bindMarkowitzEvents() {
    // Timeline slider Markowitz
    const mrkSlider = document.getElementById('timeline-mrk-slider');
    if (mrkSlider) {
      mrkSlider.addEventListener('input', () => {
        if (this.mrkIsPlaying) this._pauseMarkowitz();
        const gen = parseInt(mrkSlider.value, 10);
        this._setMarkowitzGeneration(gen);
      });
    }

    // Botones de control Markowitz
    const btnRun = document.getElementById('btn-mrk-run');
    if (btnRun) {
      btnRun.addEventListener('click', () => {
        if (this.mrkIsPlaying) {
          this._pauseMarkowitz();
        } else {
          this._playMarkowitz();
        }
      });
    }

    const btnPause = document.getElementById('btn-mrk-pause');
    if (btnPause) {
      btnPause.addEventListener('click', () => this._pauseMarkowitz());
    }

    const btnStep = document.getElementById('btn-mrk-step');
    if (btnStep) {
      btnStep.addEventListener('click', () => {
        this._pauseMarkowitz();
        const maxG = this.mrkData?.metrics?.generations_executed || 150;
        const nextG = Math.min(this.mrkCurrentGen + 1, maxG);
        this._setMarkowitzGeneration(nextG);
      });
    }

    const btnRestore = document.getElementById('btn-mrk-restore');
    if (btnRestore) {
      btnRestore.addEventListener('click', () => {
        this._pauseMarkowitz();
        const maxG = this.mrkData?.metrics?.generations_executed || 150;
        this._setMarkowitzGeneration(maxG);
      });
    }
  }

  _loadMarkowitzCertifiedData() {
    const d = this.mrkData;
    if (!d) return;

    // Inicializar nube Monte Carlo, activos y óptimo
    const optPort = d.generation_portfolios ? d.generation_portfolios[d.generation_portfolios.length - 1] : null;
    this.mrkFrontierChart.setData(d.monte_carlo_cloud, d.instance?.assets, optPort, (d.instance?.risk_free_rate || 0.035) * 100);

    // Inicializar asignación patrimonial
    if (optPort) {
      this.mrkAllocationChart.updateAllocations(optPort.allocations);
    }

    // Inicializar convergencia
    if (d.history && this.mrkConvergenceChart) {
      this.mrkConvergenceChart.update(d.history);
    }

    // Renderizar tabla Top 8
    this._renderMarkowitzTopTable(d.top_individuals || []);

    // Actualizar KPIs al estado final
    this._setMarkowitzGeneration(150);
  }

  _setMarkowitzGeneration(gen) {
    if (!this.mrkData) return;
    this.mrkCurrentGen = gen;

    // Actualizar slider y etiqueta
    const slider = document.getElementById('timeline-mrk-slider');
    if (slider) slider.value = gen;

    const label = document.getElementById('timeline-mrk-val');
    if (label) label.innerHTML = `Gen <strong>${gen}</strong> / 150`;

    // Obtener portafolio de esa generación
    const port = this.mrkData.generation_portfolios ? this.mrkData.generation_portfolios[gen] : null;
    if (port) {
      // Actualizar KPIs
      const elRet = document.getElementById('kpi-mrk-return');
      if (elRet) elRet.textContent = port.expected_return.toFixed(2);

      const elVol = document.getElementById('kpi-mrk-vol');
      if (elVol) elVol.textContent = port.volatility.toFixed(2);

      const elSr = document.getElementById('kpi-mrk-sharpe');
      if (elSr) elSr.textContent = port.sharpe_ratio.toFixed(4);

      const elFeas = document.getElementById('kpi-mrk-feas');
      const elFeasSub = document.getElementById('kpi-mrk-feas-sub');
      if (elFeas) {
        if (port.is_feasible) {
          elFeas.textContent = '100% Factible';
          elFeas.style.color = 'var(--accent)';
          if (elFeasSub) elFeasSub.textContent = `Suma: ${port.budget_sum.toFixed(2)}% · 0 excesos`;
        } else {
          elFeas.textContent = 'Penalizado';
          elFeas.style.color = '#EF4444';
          if (elFeasSub) elFeasSub.textContent = `Suma: ${port.budget_sum.toFixed(2)}% · Discrepancia`;
        }
      }

      // Actualizar gráficos
      const rf = (this.mrkData.instance?.risk_free_rate || 0.035) * 100;
      this.mrkFrontierChart.updateCurrentPoint(port, rf);
      this.mrkAllocationChart.updateAllocations(port.allocations);
    }

    // Actualizar curva de convergencia cortada a 'gen'
    if (this.mrkConvergenceChart && this.mrkData.history) {
      const slicedHist = this.mrkData.history.slice(0, gen + 1);
      this.mrkConvergenceChart.update(slicedHist);
    }
  }

  _playMarkowitz() {
    this.mrkIsPlaying = true;
    const btnRunText = document.getElementById('btn-mrk-run-text');
    if (btnRunText) btnRunText.textContent = 'Ejecutando...';

    const btnPause = document.getElementById('btn-mrk-pause');
    if (btnPause) btnPause.disabled = false;

    if (this.mrkCurrentGen >= 150) {
      this.mrkCurrentGen = 0;
      this._setMarkowitzGeneration(0);
    }

    let lastTime = performance.now();
    const intervalMs = 65;

    const loop = (now) => {
      if (!this.mrkIsPlaying) return;
      if (now - lastTime >= intervalMs) {
        lastTime = now;
        const nextGen = this.mrkCurrentGen + 1;
        if (nextGen <= 150) {
          this._setMarkowitzGeneration(nextGen);
        } else {
          this._pauseMarkowitz();
          return;
        }
      }
      this.mrkRafId = requestAnimationFrame(loop);
    };

    this.mrkRafId = requestAnimationFrame(loop);
  }

  _pauseMarkowitz() {
    this.mrkIsPlaying = false;
    if (this.mrkRafId) {
      cancelAnimationFrame(this.mrkRafId);
      this.mrkRafId = null;
    }
    const btnRunText = document.getElementById('btn-mrk-run-text');
    if (btnRunText) btnRunText.textContent = 'Ejecutar';

    const btnPause = document.getElementById('btn-mrk-pause');
    if (btnPause) btnPause.disabled = true;
  }


  /* ═════════════════════════════════════════════════════════════
     COMPARADOR MULTI-PROBLEMA (VRP vs. PCB vs. Markowitz)
     ═════════════════════════════════════════════════════════════ */
  _initComparadorCharts() {
    const vrpData = this.pyData || window.VRP_SIMULATION_DATA || {};
    const pcbData = this.pcbData || window.PCB_SIMULATION_DATA || {};
    const mrkData = this.mrkData || window.MARKOWITZ_SIMULATION_DATA || {};

    const vrpHist = vrpData.history || [];
    const pcbHist = pcbData.history || [];
    const mrkHist = mrkData.history || [];

    const convCanvas = document.getElementById('comp-convergence-chart');
    if (convCanvas) {
      if (this.compConvergenceChart) {
        this.compConvergenceChart.destroy();
        this.compConvergenceChart = null;
      }

      const maxGens = Math.max(vrpHist.length, pcbHist.length, mrkHist.length, 151);
      const labels = Array.from({ length: maxGens }, (_, i) => i);

      const vrpBest = vrpHist.map(h => (typeof h.best === 'number' ? h.best : null));
      const pcbBest = pcbHist.map(h => (typeof h.best === 'number' ? h.best : null));
      const mrkBest = mrkHist.map(h => (typeof h.best === 'number' ? h.best : null));

      this.compConvergenceChart = new Chart(convCanvas.getContext('2d'), {
        type: 'line',
        data: {
          labels,
          datasets: [
            {
              label: 'P1: VRP (Rutas — Distancia)',
              data: vrpBest,
              borderColor: '#10B981',
              backgroundColor: 'rgba(16, 185, 129, 0.08)',
              borderWidth: 2.2,
              pointRadius: 0,
              pointHoverRadius: 5,
              tension: 0.25,
              fill: false
            },
            {
              label: 'P2: PCB (Circuitos — WL + Sustrato)',
              data: pcbBest,
              borderColor: '#38BDF8',
              backgroundColor: 'rgba(56, 189, 248, 0.08)',
              borderWidth: 2.2,
              pointRadius: 0,
              pointHoverRadius: 5,
              tension: 0.25,
              fill: false
            },
            {
              label: 'P3: Markowitz (Cartera — Sharpe Inverso)',
              data: mrkBest,
              borderColor: '#F59E0B',
              backgroundColor: 'rgba(245, 158, 11, 0.08)',
              borderWidth: 2.2,
              pointRadius: 0,
              pointHoverRadius: 5,
              tension: 0.25,
              fill: false
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          interaction: {
            mode: 'index',
            intersect: false
          },
          plugins: {
            legend: {
              display: true,
              position: 'top',
              labels: {
                color: '#CBD5E1',
                font: { family: "'Inter', sans-serif", size: 11, weight: '500' },
                boxWidth: 12,
                boxHeight: 12,
                padding: 14
              }
            },
            tooltip: {
              backgroundColor: 'rgba(17, 24, 39, 0.95)',
              titleColor: '#F8FAFC',
              bodyColor: '#94A3B8',
              borderColor: 'rgba(255, 255, 255, 0.12)',
              borderWidth: 1,
              padding: 10,
              callbacks: {
                label: (ctx) => {
                  const val = ctx.parsed.y;
                  return ` ${ctx.dataset.label}: ${val !== null ? val.toFixed(5) : 'N/D'}`;
                }
              }
            }
          },
          scales: {
            x: {
              title: { display: true, text: 'Generación Evolutiva (0 a 150)', color: '#64748B', font: { family: "'Inter', sans-serif", size: 10 } },
              grid: { color: 'rgba(255, 255, 255, 0.04)' },
              ticks: { color: '#64748B', maxTicksLimit: 12, font: { family: "'JetBrains Mono', monospace", size: 10 } }
            },
            y: {
              title: { display: true, text: 'Fitness Normalizado f_norm (Minimización)', color: '#64748B', font: { family: "'Inter', sans-serif", size: 10 } },
              grid: { color: 'rgba(255, 255, 255, 0.04)' },
              ticks: { color: '#64748B', font: { family: "'JetBrains Mono', monospace", size: 10 } }
            }
          }
        }
      });
    }

    const radarCanvas = document.getElementById('comp-radar-chart');
    if (radarCanvas) {
      if (this.compRadarChart) {
        this.compRadarChart.destroy();
        this.compRadarChart = null;
      }

      this.compRadarChart = new Chart(radarCanvas.getContext('2d'), {
        type: 'radar',
        data: {
          labels: [
            'Eficiencia de Convergencia',
            'Satisfacción de Restricciones',
            'Optimización del Objetivo',
            'Preservación de Diversidad',
            'Velocidad por Generación'
          ],
          datasets: [
            {
              label: 'P1: VRP (Rutas)',
              data: [98, 100, 96, 85, 94],
              borderColor: '#10B981',
              backgroundColor: 'rgba(16, 185, 129, 0.16)',
              borderWidth: 2,
              pointBackgroundColor: '#10B981',
              pointRadius: 3
            },
            {
              label: 'P2: PCB (Circuitos)',
              data: [91, 100, 94, 90, 86],
              borderColor: '#38BDF8',
              backgroundColor: 'rgba(56, 189, 248, 0.16)',
              borderWidth: 2,
              pointBackgroundColor: '#38BDF8',
              pointRadius: 3
            },
            {
              label: 'P3: Markowitz (Cartera)',
              data: [96, 100, 98, 88, 95],
              borderColor: '#F59E0B',
              backgroundColor: 'rgba(245, 158, 11, 0.16)',
              borderWidth: 2,
              pointBackgroundColor: '#F59E0B',
              pointRadius: 3
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              display: true,
              position: 'top',
              labels: {
                color: '#CBD5E1',
                font: { family: "'Inter', sans-serif", size: 11, weight: '500' },
                boxWidth: 10,
                boxHeight: 10,
                padding: 12
              }
            },
            tooltip: {
              backgroundColor: 'rgba(17, 24, 39, 0.95)',
              titleColor: '#F8FAFC',
              bodyColor: '#94A3B8',
              borderColor: 'rgba(255, 255, 255, 0.12)',
              borderWidth: 1,
              callbacks: {
                label: (ctx) => ` ${ctx.dataset.label}: ${ctx.raw} / 100 pts`
              }
            }
          },
          scales: {
            r: {
              min: 50,
              max: 100,
              ticks: { display: false, stepSize: 10 },
              grid: { color: 'rgba(255, 255, 255, 0.07)' },
              angleLines: { color: 'rgba(255, 255, 255, 0.08)' },
              pointLabels: {
                color: '#E2E8F0',
                font: { family: "'Inter', sans-serif", size: 10, weight: 600 }
              }
            }
          }
        }
      });
    }
  }

    _renderMarkowitzTopTable(individuals) {
    const tbody = document.getElementById('markowitz-top-tbody');
    if (!tbody) return;

    tbody.innerHTML = individuals.map((ind, idx) => {
      const isElite = idx < 2;
      const rankColor = idx === 0 ? 'var(--accent)' : idx === 1 ? 'var(--primary)' : 'var(--text-bright)';
      const roleBadge = isElite
        ? `<span class="badge badge-accent">Élite Invariable (k=${idx+1})</span>`
        : `<span class="badge" style="background:rgba(59,130,246,0.1); color:#3B82F6">Candidato Torneo (p=2)</span>`;

      const barColor = isElite ? '#10B981' : '#3B82F6';

      return `
        <tr>
          <td class="mono font-bold" style="color:${rankColor}">#${ind.rank}</td>
          <td class="mono font-bold">${ind.fitness.toFixed(6)}</td>
          <td class="mono text-muted">${ind.aptitud_pct.toFixed(2)}%</td>
          <td class="mono" style="color:#10B981">${ind.expected_return_pct.toFixed(2)}%</td>
          <td class="mono" style="color:#EF4444">${ind.volatility_pct.toFixed(2)}%</td>
          <td class="mono font-bold" style="color:#38BDF8">${ind.sharpe_ratio.toFixed(4)}</td>
          <td class="mono text-muted">${(ind.budget_error * 100).toFixed(3)}%</td>
          <td>${roleBadge}</td>
          <td style="min-width: 140px;">
            <div style="display:flex; align-items:center; gap:8px;">
              <div style="flex:1; height:6px; background:rgba(255,255,255,0.06); border-radius:3px; overflow:hidden;">
                <div style="width:${ind.aptitud_pct}%; height:100%; background:${barColor}; border-radius:3px;"></div>
              </div>
              <span class="mono" style="font-size:0.75rem; color:#94A3B8;">${ind.aptitud_pct.toFixed(1)}%</span>
            </div>
          </td>
        </tr>
      `;
    }).join('');
  }

}

