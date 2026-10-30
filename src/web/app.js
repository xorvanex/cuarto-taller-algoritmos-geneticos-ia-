/**
 * EvoSuite AI — Motor de Algoritmos Genéticos (JavaScript)
 * =========================================================
 * Implementación fiel al SGA del profesor Ospina Alarcón:
 *   • Selección: Torneo asimétrico (Padre1=torneo, Padre2=aleatorio)
 *   • Cruce:     SPX (Single-Point Crossover)  Pc = 90 %
 *   • Mutación:  Puntual                        Pm = 2 %
 *   • Elitismo:  2 mejores individuos
 *   • Fitness:   Normalizado ∈ [0,1], convergencia fitness ≤ 1e-2
 *
 * Problema 1 — VRP (Vehicle Routing Problem)
 *   • Codificación RKGA (Random Keys GA)
 *   • 15 clientes, 3 camiones, capacidad 120 kg
 *   • D_ideal = 235 km, D_worst = 750 km
 *
 * Autores: Palmera Navarro D.D. · Camargo Padilla J.D.
 * Materia: Inteligencia Artificial — Corte 3, 2026
 * Docente: Ing. Manuel Alejandro Ospina Alarcón
 */

'use strict';

/* ════════════════════════════════════════════════════════════
   1. INSTANCIA VRP — Caso base de cátedra (igual a vrp.py)
   ════════════════════════════════════════════════════════════ */

/** @typedef {{ id: string, x: number, y: number, demand: number, service: number }} Customer */
/** @typedef {{ x: number, y: number }} Depot */

// Instancia base de referencia (se sobrescribe si Python exportó data/vrp_results.js)
const PY_DATA = typeof window !== 'undefined' && window.VRP_SIMULATION_DATA ? window.VRP_SIMULATION_DATA : null;

const VRP_INSTANCE = PY_DATA && PY_DATA.instance ? {
  depot: PY_DATA.instance.depot,
  customers: PY_DATA.instance.customers,
  numVehicles: PY_DATA.instance.num_vehicles,
  capacity: PY_DATA.instance.vehicle_capacity,
  maxShift: PY_DATA.instance.max_shift_time,
  speed: PY_DATA.instance.avg_speed,
  D_ideal: 235.0,
  D_worst: 750.0,
  lambda1: 0.05,
  lambda2: 0.02,
} : {
  depot: { x: 50, y: 50 },

  /** 15 clientes de cátedra */
  customers: [
    { id: 'C1',  x: 20, y: 30, demand: 15, service: 20 },
    { id: 'C2',  x: 35, y: 15, demand: 18, service: 25 },
    { id: 'C3',  x: 60, y: 10, demand: 12, service: 18 },
    { id: 'C4',  x: 80, y: 25, demand: 20, service: 22 },
    { id: 'C5',  x: 75, y: 55, demand: 14, service: 20 },
    { id: 'C6',  x: 90, y: 70, demand: 16, service: 24 },
    { id: 'C7',  x: 85, y: 85, demand: 22, service: 28 },
    { id: 'C8',  x: 55, y: 80, demand: 13, service: 18 },
    { id: 'C9',  x: 30, y: 85, demand: 17, service: 22 },
    { id: 'C10', x: 15, y: 65, demand: 11, service: 16 },
    { id: 'C11', x: 10, y: 45, demand: 19, service: 26 },
    { id: 'C12', x: 25, y: 20, demand: 14, service: 20 },
    { id: 'C13', x: 45, y: 35, demand: 16, service: 22 },
    { id: 'C14', x: 65, y: 45, demand: 18, service: 24 },
    { id: 'C15', x: 70, y: 75, demand: 15, service: 20 },
  ],

  numVehicles:   3,
  capacity:      120,   // kg por camión
  maxShift:      480,   // minutos (turno de 8h)
  speed:         45,    // km/h

  // Parámetros de normalización (aprobados en tutoría 29-oct-2026)
  D_ideal:  235.0,      // distancia teórica mínima (km)
  D_worst:  750.0,      // distancia peor caso (km)

  // Coeficientes de penalización
  lambda1:  0.05,       // violación de capacidad
  lambda2:  0.02,       // violación de tiempo
};

/* ════════════════════════════════════════════════════════════
   2. PARÁMETROS DEL SGA — Caso base de cátedra
   ════════════════════════════════════════════════════════════ */

const DEFAULT_PARAMS = {
  popSize:   1000,   // N individuos
  elitism:   2,      // número de élites
  pc:        0.90,   // probabilidad de cruce SPX
  pm:        0.02,   // probabilidad de mutación puntual
  maxGens:   150,    // generaciones máximas
  threshold: 1e-2,   // criterio de parada (fitness normalizado)
  tournSize: 2,      // tamaño del torneo (Padre1)
};

/* ════════════════════════════════════════════════════════════
   3. UTILS — Funciones auxiliares
   ════════════════════════════════════════════════════════════ */

/**
 * Distancia euclidiana entre dos puntos.
 * @param {{x:number, y:number}} a
 * @param {{x:number, y:number}} b
 * @returns {number}
 */
function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

/**
 * Retorna el índice del mínimo en un array.
 * @param {number[]} arr
 * @returns {number}
 */
function argmin(arr) {
  return arr.reduce((iMin, v, i, a) => (v < a[iMin] ? i : iMin), 0);
}

/**
 * Genera un array de longitud n con valores aleatorios ∈ [0,1).
 * @param {number} n
 * @returns {number[]}
 */
function randomChromosome(n) {
  return Array.from({ length: n }, () => Math.random());
}

/* ════════════════════════════════════════════════════════════
   4. DECODIFICACIÓN RKGA — Cromosoma → Rutas VRP
   ════════════════════════════════════════════════════════════ */

/**
 * Decodifica un cromosoma RKGA en rutas válidas para el VRP.
 *
 * Algoritmo:
 *   1. Ordenar índices de clientes por clave aleatoria (argsort).
 *   2. Asignar clientes en ese orden al camión activo.
 *   3. Si agregar el cliente viola capacidad o tiempo → abrir nuevo camión.
 *   4. Ignorar clientes sin camión disponible (penalización).
 *
 * @param {number[]} chromosome  - vector de n=15 valores ∈ [0,1)
 * @param {typeof VRP_INSTANCE} inst
 * @returns {{ routes: number[][], totalDist: number, penalty: number }}
 */
function decodeChromosome(chromosome, inst) {
  const n   = inst.customers.length;
  const nV  = inst.numVehicles;

  // Ordenar clientes por clave aleatoria
  const order = Array.from({ length: n }, (_, i) => i)
    .sort((a, b) => chromosome[a] - chromosome[b]);

  /** Estado de cada camión: cargas, tiempo, posición actual */
  const loads   = new Array(nV).fill(0);
  const times   = new Array(nV).fill(0);
  const pos     = new Array(nV).fill(null).map(() => ({ ...inst.depot }));
  const routes  = Array.from({ length: nV }, () => []);

  let penalty = 0;

  for (const ci of order) {
    const cust = inst.customers[ci];
    let assigned = false;

    // Intentar asignar al primer camión que puede aceptarlo
    for (let v = 0; v < nV; v++) {
      const travelTime  = (dist(pos[v], cust) / inst.speed) * 60; // min
      const newLoad     = loads[v] + cust.demand;
      const newTime     = times[v] + travelTime + cust.service;

      if (newLoad <= inst.capacity && newTime <= inst.maxShift) {
        loads[v]  = newLoad;
        times[v]  = newTime;
        pos[v]    = { x: cust.x, y: cust.y };
        routes[v].push(ci);
        assigned  = true;
        break;
      }
    }

    if (!assigned) {
      // No hay camión disponible → penalizar
      penalty += cust.demand * inst.lambda1 + 10 * inst.lambda2;
    }
  }

  // Calcular distancia total (depot → ruta → depot)
  let totalDist = 0;
  for (let v = 0; v < nV; v++) {
    if (routes[v].length === 0) continue;
    let prev = inst.depot;
    for (const ci of routes[v]) {
      totalDist += dist(prev, inst.customers[ci]);
      prev = inst.customers[ci];
    }
    totalDist += dist(prev, inst.depot);  // regreso al depot
  }

  return { routes, totalDist, penalty };
}

/**
 * Calcula el fitness normalizado de un cromosoma.
 *
 * Normalización aprobada por el Ing. Ospina en tutoría (29-oct-2026):
 *   error_dist = (D_total - D_ideal) / (D_worst - D_ideal)
 *   fitness_normalizado = error_dist + penalty
 *
 * Convergencia cuando fitness_normalizado ≤ threshold (1e-2).
 * Aptitud biológica = 1 - fitness_normalizado → tiende a 1.
 *
 * @param {number[]} chromosome
 * @param {typeof VRP_INSTANCE} inst
 * @returns {number}
 */
function computeFitness(chromosome, inst) {
  const { totalDist, penalty } = decodeChromosome(chromosome, inst);
  const errDist = (totalDist - inst.D_ideal) / (inst.D_worst - inst.D_ideal);
  return Math.max(0, errDist) + penalty;
}

/* ════════════════════════════════════════════════════════════
   5. OPERADORES GENÉTICOS — Fiel a los MATLAB del profesor
   ════════════════════════════════════════════════════════════ */

/**
 * Selección por torneo asimétrico.
 *
 * Implementación exacta de seleccion_torneo_2.m:
 *   • Padre1: torneo determinístico (mejor de k individuos aleatorios)
 *   • Padre2: individuo aleatorio uniforme (NO torneo)
 *
 * @param {number[][]} pop        - población actual
 * @param {number[]}   fitness    - fitness de cada individuo
 * @param {number}     tournSize  - tamaño del torneo para Padre1
 * @returns {[number[], number[]]}  [padre1, padre2]
 */
function seleccionTorneo2(pop, fitness, tournSize) {
  const n = pop.length;

  // Padre 1: torneo determinístico
  const candidates1 = Array.from({ length: tournSize }, () =>
    Math.floor(Math.random() * n)
  );
  const idx1 = candidates1.reduce(
    (best, i) => (fitness[i] < fitness[best] ? i : best),
    candidates1[0]
  );
  const padre1 = pop[idx1];

  // Padre 2: aleatorio uniforme (no torneo)
  const idx2   = Math.floor(Math.random() * n);
  const padre2 = pop[idx2];

  return [padre1, padre2];
}

/**
 * Cruce SPX (Single-Point Crossover).
 *
 * Implementación de Cruce.m:
 *   Si r < Pc: elegir punto de corte aleatorio, intercambiar segmentos.
 *   Si no:     retornar copias de los padres sin cambio.
 *
 * @param {number[]} p1  - Padre 1
 * @param {number[]} p2  - Padre 2
 * @param {number}   pc  - probabilidad de cruce
 * @returns {[number[], number[]]}  [hijo1, hijo2]
 */
function cruceSPX(p1, p2, pc) {
  const n = p1.length;
  if (Math.random() < pc) {
    const punto = 1 + Math.floor(Math.random() * (n - 1));
    const h1 = [...p1.slice(0, punto), ...p2.slice(punto)];
    const h2 = [...p2.slice(0, punto), ...p1.slice(punto)];
    return [h1, h2];
  }
  return [[...p1], [...p2]];
}

/**
 * Mutación puntual.
 *
 * Implementación de Mutacion.m:
 *   Por cada gen: si r < Pm → reemplazar con valor aleatorio ∈ [0,1).
 *
 * @param {number[]} chromosome
 * @param {number}   pm  - probabilidad de mutación por gen
 * @returns {number[]}
 */
function mutacionPuntual(chromosome, pm) {
  return chromosome.map(gene =>
    Math.random() < pm ? Math.random() : gene
  );
}

/**
 * Elitismo: copia los 'k' mejores individuos sin modificar.
 *
 * Implementación de Elitismo.m.
 *
 * @param {number[][]} pop      - población actual
 * @param {number[]}   fitness  - fitness de cada individuo
 * @param {number}     k        - número de élites
 * @returns {{ elites: number[][], fitElites: number[] }}
 */
function elitismo(pop, fitness, k) {
  const indices = Array.from({ length: pop.length }, (_, i) => i)
    .sort((a, b) => fitness[a] - fitness[b])
    .slice(0, k);
  return {
    elites:    indices.map(i => [...pop[i]]),
    fitElites: indices.map(i => fitness[i]),
  };
}

/* ════════════════════════════════════════════════════════════
   6. MOTOR SGA — Clase principal
   ════════════════════════════════════════════════════════════ */

/**
 * @typedef {Object} GAParams
 * @property {number} popSize
 * @property {number} elitism
 * @property {number} pc
 * @property {number} pm
 * @property {number} maxGens
 * @property {number} threshold
 * @property {number} tournSize
 */

/**
 * @typedef {Object} GenerationStats
 * @property {number} gen
 * @property {number} best
 * @property {number} avg
 * @property {number} worst
 * @property {number} std
 */

class SimpleGeneticAlgorithm {
  /**
   * @param {GAParams} params
   * @param {typeof VRP_INSTANCE} instance
   */
  constructor(params, instance) {
    this.params   = { ...params };
    this.instance = instance;
    this.reset();
  }

  /** Inicializa o reinicia el estado del algoritmo. */
  reset() {
    const { popSize } = this.params;
    const nGenes = this.instance.customers.length;

    this.generation  = 0;
    this.converged   = false;
    this.history     = [];   // GenerationStats[]
    this.population  = Array.from({ length: popSize }, () => randomChromosome(nGenes));
    this.fitness     = this.population.map(c => computeFitness(c, this.instance));
    this._recordStats();
  }

  /** Registra las estadísticas de la generación actual. */
  _recordStats() {
    const f      = this.fitness;
    const n      = f.length;
    const best   = Math.min(...f);
    const worst  = Math.max(...f);
    const avg    = f.reduce((s, v) => s + v, 0) / n;
    const std    = Math.sqrt(f.reduce((s, v) => s + (v - avg) ** 2, 0) / n);

    this.history.push({ gen: this.generation, best, avg, worst, std });

    // Aptitud biológica (tiende a 1 cuando fitness → 0)
    this.bestFitness     = best;
    this.bestAptitud     = Math.max(0, 1 - best);
    this.bestChromosome  = this.population[this.fitness.indexOf(best)];
  }

  /**
   * Ejecuta una generación del SGA.
   * @returns {boolean}  true si ya convergió o llegó al límite
   */
  step() {
    if (this.converged || this.generation >= this.params.maxGens) {
      this.converged = true;
      return true;
    }

    const { popSize, elitism: kElite, pc, pm, tournSize } = this.params;

    // 1. Elitismo
    const { elites, fitElites } = elitismo(this.population, this.fitness, kElite);

    // 2. Crear nueva población
    const newPop     = [];
    const newFitness = [];

    // Insertar élites
    for (let i = 0; i < elites.length; i++) {
      newPop.push(elites[i]);
      newFitness.push(fitElites[i]);
    }

    // Completar con hijos
    while (newPop.length < popSize) {
      const [p1, p2]  = seleccionTorneo2(this.population, this.fitness, tournSize);
      const [h1, h2]  = cruceSPX(p1, p2, pc);
      const m1        = mutacionPuntual(h1, pm);
      const m2        = mutacionPuntual(h2, pm);

      newPop.push(m1);
      newFitness.push(computeFitness(m1, this.instance));

      if (newPop.length < popSize) {
        newPop.push(m2);
        newFitness.push(computeFitness(m2, this.instance));
      }
    }

    this.population = newPop.slice(0, popSize);
    this.fitness    = newFitness.slice(0, popSize);
    this.generation++;
    this._recordStats();

    // Criterio de convergencia
    if (this.bestFitness <= this.params.threshold) {
      this.converged = true;
      return true;
    }

    return false;
  }

  /**
   * Obtiene la mejor solución decodificada.
   * @returns {{ routes: number[][], totalDist: number, penalty: number }}
   */
  getBestSolution() {
    return decodeChromosome(this.bestChromosome, this.instance);
  }

  /**
   * Top-k individuos de la población actual.
   * @param {number} k
   * @returns {{ chromosome: number[], fitness: number }[]}
   */
  getTopK(k = 10) {
    const idx = Array.from({ length: this.population.length }, (_, i) => i)
      .sort((a, b) => this.fitness[a] - this.fitness[b])
      .slice(0, k);
    return idx.map(i => ({ chromosome: this.population[i], fitness: this.fitness[i] }));
  }
}

/* ════════════════════════════════════════════════════════════
   7. RENDERIZADOR DE CANVAS — Mapa VRP
   ════════════════════════════════════════════════════════════ */

/** Colores de las 3 rutas */
const ROUTE_COLORS = ['#10B981', '#3B82F6', '#F59E0B'];

class VRPRenderer {
  /**
   * @param {HTMLCanvasElement} canvas
   * @param {typeof VRP_INSTANCE} instance
   */
  constructor(canvas, instance) {
    this.canvas   = canvas;
    this.ctx      = canvas.getContext('2d');
    this.instance = instance;
    this.hoveredNode = null;   // índice del cliente bajo el cursor (-1 = depot)

    this._bindEvents();
    this.draw(null, []);
  }

  /** Convierte coordenadas del problema [0-100] → píxeles CSS del canvas. */
  _toCanvas(x, y) {
    const pad = 35;
    const w   = (this.cssWidth  || 400) - 2 * pad;
    const h   = (this.cssHeight || 300) - 2 * pad;
    return {
      px: pad + (x / 100) * w,
      py: pad + (1 - y / 100) * h,  // Y invertido (canvas origin arriba-izq)
    };
  }

  _bindEvents() {
    this.canvas.addEventListener('mousemove', e => this._onMouseMove(e));
    this.canvas.addEventListener('mouseleave', () => {
      this.hoveredNode = null;
      const tooltip = document.getElementById('canvas-tooltip');
      if (tooltip) tooltip.style.display = 'none';
    });
  }

  _onMouseMove(e) {
    const rect = this.canvas.getBoundingClientRect();
    const mx   = e.clientX - rect.left;
    const my   = e.clientY - rect.top;

    let closest = null;
    let minD    = 15;  // radio de detección en px CSS

    // Depot
    const dep = this._toCanvas(this.instance.depot.x, this.instance.depot.y);
    const dDep = Math.hypot(mx - dep.px, my - dep.py);
    if (dDep < minD) { closest = { type: 'depot' }; minD = dDep; }

    // Clientes
    this.instance.customers.forEach((c, i) => {
      const p = this._toCanvas(c.x, c.y);
      const d = Math.hypot(mx - p.px, my - p.py);
      if (d < minD) { closest = { type: 'customer', index: i }; minD = d; }
    });

    const tooltip = document.getElementById('canvas-tooltip');
    if (!tooltip) return;

    if (closest) {
      let label = '';
      if (closest.type === 'depot') {
        label = `🏭 Depósito (${this.instance.depot.x}, ${this.instance.depot.y})`;
      } else {
        const c = this.instance.customers[closest.index];
        label = `${c.id} | Dem: ${c.demand} kg | Serv: ${c.service} min`;
      }
      tooltip.style.display = 'block';
      tooltip.style.left = (e.offsetX + 12) + 'px';
      tooltip.style.top  = (e.offsetY - 24) + 'px';
      tooltip.textContent = label;
    } else {
      tooltip.style.display = 'none';
    }
    this.hoveredNode = closest;
  }

  /**
   * Dibuja el mapa de rutas VRP.
   * @param {number[][]} routes   - rutas actuales (puede ser null)
   * @param {number[]}   [routeColors]
   */
  draw(routes, routeColors = ROUTE_COLORS) {
    const canvas = this.canvas;
    const ctx    = this.ctx;

    const rect = canvas.getBoundingClientRect();
    const dpr  = window.devicePixelRatio || 1;
    const w    = rect.width  || 400;
    const h    = rect.height || 300;

    canvas.width  = Math.floor(w * dpr);
    canvas.height = Math.floor(h * dpr);
    canvas.style.width  = w + 'px';
    canvas.style.height = h + 'px';

    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.scale(dpr, dpr);

    this.cssWidth  = w;
    this.cssHeight = h;

    ctx.clearRect(0, 0, w, h);

    // Fondo oscuro
    ctx.fillStyle = 'rgba(7, 11, 18, 0.5)';
    ctx.fillRect(0, 0, w, h);

    // Grid de referencia
    this._drawGrid(ctx, w, h);

    // Rutas
    if (routes) {
      routes.forEach((route, v) => {
        if (route.length === 0) return;
        this._drawRoute(ctx, route, routeColors[v] || '#fff');
      });
    }

    // Clientes
    this.instance.customers.forEach((c, i) => {
      this._drawCustomer(ctx, c, i, routes);
    });

    // Depósito
    this._drawDepot(ctx);
  }

  _drawGrid(ctx, w, h) {
    ctx.save();
    ctx.strokeStyle = 'rgba(255,255,255,0.04)';
    ctx.lineWidth   = 1;
    const pad = 35;

    for (let x = pad; x <= w - pad; x += (w - 2 * pad) / 10) {
      ctx.beginPath();
      ctx.moveTo(x, pad);
      ctx.lineTo(x, h - pad);
      ctx.stroke();
    }

    for (let y = pad; y <= h - pad; y += (h - 2 * pad) / 10) {
      ctx.beginPath();
      ctx.moveTo(pad, y);
      ctx.lineTo(w - pad, y);
      ctx.stroke();
    }
    ctx.restore();
  }

  _drawRoute(ctx, route, color) {
    const inst = this.instance;
    const points = [
      this._toCanvas(inst.depot.x, inst.depot.y),
      ...route.map(ci => this._toCanvas(inst.customers[ci].x, inst.customers[ci].y)),
      this._toCanvas(inst.depot.x, inst.depot.y),
    ];

    ctx.save();

    // Sombra/glow de la ruta
    ctx.shadowColor  = color;
    ctx.shadowBlur   = 6;
    ctx.strokeStyle  = color + 'BB';
    ctx.lineWidth    = 2;
    ctx.lineJoin     = 'round';
    ctx.lineCap      = 'round';

    ctx.beginPath();
    ctx.moveTo(points[0].px, points[0].py);
    for (let i = 1; i < points.length; i++) {
      ctx.lineTo(points[i].px, points[i].py);
    }
    ctx.stroke();

    // Flechas de dirección en el segmento medio
    for (let i = 0; i < points.length - 1; i++) {
      this._drawArrow(ctx, points[i], points[i+1], color);
    }

    ctx.restore();
  }

  _drawArrow(ctx, from, to, color) {
    const mx   = (from.px + to.px) / 2;
    const my   = (from.py + to.py) / 2;
    const ang  = Math.atan2(to.py - from.py, to.px - from.px);
    const size = 6;

    ctx.save();
    ctx.translate(mx, my);
    ctx.rotate(ang);
    ctx.fillStyle   = color;
    ctx.globalAlpha = 0.7;
    ctx.beginPath();
    ctx.moveTo(size, 0);
    ctx.lineTo(-size * 0.6, -size * 0.4);
    ctx.lineTo(-size * 0.6,  size * 0.4);
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }

  _drawCustomer(ctx, c, idx, routes) {
    const { px, py } = this._toCanvas(c.x, c.y);

    // Determinar a qué ruta pertenece
    let color = '#64748B';
    if (routes) {
      for (let v = 0; v < routes.length; v++) {
        if (routes[v].includes(idx)) {
          color = ROUTE_COLORS[v];
          break;
        }
      }
    }

    const r = 5 + (c.demand / 22) * 3;  // radio proporcional a demanda

    ctx.save();

    // Glow
    ctx.shadowColor = color;
    ctx.shadowBlur  = 8;

    // Círculo exterior
    ctx.beginPath();
    ctx.arc(px, py, r + 2, 0, Math.PI * 2);
    ctx.fillStyle = color + '22';
    ctx.fill();

    // Círculo interior
    ctx.beginPath();
    ctx.arc(px, py, r, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();

    // Borde
    ctx.strokeStyle = 'rgba(255,255,255,0.25)';
    ctx.lineWidth   = 1;
    ctx.stroke();

    // Label
    ctx.shadowBlur  = 0;
    ctx.fillStyle   = '#F8FAFC';
    ctx.font        = `bold ${Math.max(8, r * 1.4)}px 'Fira Code', monospace`;
    ctx.textAlign   = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(c.id.replace('C', ''), px, py);

    ctx.restore();
  }

  _drawDepot(ctx) {
    const { px, py } = this._toCanvas(this.instance.depot.x, this.instance.depot.y);

    ctx.save();
    ctx.shadowColor = '#F8FAFC';
    ctx.shadowBlur  = 12;

    // Depot: cuadrado rotado 45° (diamante)
    const s = 10;
    ctx.beginPath();
    ctx.moveTo(px,     py - s);
    ctx.lineTo(px + s, py    );
    ctx.lineTo(px,     py + s);
    ctx.lineTo(px - s, py    );
    ctx.closePath();

    ctx.fillStyle   = 'rgba(248,250,252,0.9)';
    ctx.fill();
    ctx.strokeStyle = '#10B981';
    ctx.lineWidth   = 2;
    ctx.stroke();

    ctx.shadowBlur  = 0;
    ctx.fillStyle   = '#0B0F19';
    ctx.font        = 'bold 9px Inter, sans-serif';
    ctx.textAlign   = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('D', px, py);
    ctx.restore();
  }
}

/* ════════════════════════════════════════════════════════════
   8. CHART.JS — Gráfica de Convergencia
   ════════════════════════════════════════════════════════════ */

class ConvergenceChart {
  /**
   * @param {string} canvasId
   */
  constructor(canvasId) {
    this.ctx     = document.getElementById(canvasId).getContext('2d');
    this.chart   = null;
    this._init();
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
            title: {
              display: true,
              text:    'Generación',
              color:   '#64748B',
              font:    { family: 'Fira Code', size: 10 },
            },
            grid:  { color: 'rgba(255,255,255,0.04)' },
            ticks: { color: '#64748B', font: { family: 'Fira Code', size: 10 } },
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
              callback(v) { return v.toExponential(1); },
            },
          },
        },
      },
    });
  }

  /**
   * Actualiza la gráfica con el historial completo del GA.
   * @param {GenerationStats[]} history
   */
  update(history) {
    const chart  = this.chart;
    const labels = history.map(s => s.gen);
    chart.data.labels             = labels;
    chart.data.datasets[0].data  = history.map(s => Math.max(s.best,  1e-7));
    chart.data.datasets[1].data  = history.map(s => Math.max(s.avg,   1e-7));
    chart.data.datasets[2].data  = history.map(s => Math.max(s.worst, 1e-7));
    chart.update('none');
  }

  /** Resetea la gráfica a su estado vacío. */
  clear() {
    this.chart.data.labels = [];
    this.chart.data.datasets.forEach(d => (d.data = []));
    this.chart.update('none');
  }
}

/* ════════════════════════════════════════════════════════════
   9. CONTROLADOR DE UI — Orquesta todo
   ════════════════════════════════════════════════════════════ */

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
    this.speedMs   = 50;   // ms entre generaciones (ajustable)

    /* Parámetros actuales */
    this.params    = { ...DEFAULT_PARAMS };
    this.mode      = 'catedra';  // 'catedra' | 'libre'

    this._init();
  }

  /* ── Inicialización ──────────────────────────────────────── */
  _init() {
    // Canvas y chart
    const canvas      = document.getElementById('vrp-canvas');
    this.renderer     = new VRPRenderer(canvas, VRP_INSTANCE);
    this.chart        = new ConvergenceChart('convergence-chart');

    // Cargar datos oficiales de Python si están presentes
    this.pyData = PY_DATA;

    // GA inicial o cargar solución certificada de Python
    if (this.pyData && this.mode === 'catedra') {
      this._loadPythonSolution();
    } else {
      this._resetGA();
    }

    // Bindings de UI
    this._bindTabs();
    this._bindModeToggle();
    this._bindSliders();
    this._bindControls();

    // Resize observer para re-dibujar el canvas
    const ro = new ResizeObserver(() => this._redrawCanvas());
    ro.observe(canvas.parentElement);

    if (this.pyData && this.mode === 'catedra') {
      this._updateUIFromPython();
    } else {
      this._updateUI();
    }
  }

  /* ── Cargar solución certificada de Python ──────────────── */
  _loadPythonSolution() {
    if (!this.pyData) return;
    const { history, routes, metrics } = this.pyData;

    // Convertir rutas de Python a índices numéricos para el renderer
    this.pythonRoutes = routes.map(r => r.stop_indices);

    // Cargar historial en el chart
    if (history && history.length > 0) {
      this.chart.update(history);
    }

    // Renderizar rutas de Python
    this.renderer.draw(this.pythonRoutes);
  }

  _updateUIFromPython() {
    if (!this.pyData) return;
    const { metrics, history, top_individuals, params } = this.pyData;

    // KPIs con datos reales de Python
    this._setText('kpi-fitness',    metrics.best_fitness.toFixed(5));
    this._setText('kpi-dist',       metrics.total_distance_km.toFixed(1) + ' km');
    this._setText('kpi-feasible',   metrics.is_feasible ? '100%' : 'Penaliz.');
    this._setText('kpi-gen',        `${metrics.generations_executed} / ${params.max_generations}`);
    this._setText('kpi-aptitud',    (metrics.best_aptitud * 100).toFixed(2) + '%');

    // Progreso al 100%
    const fill = document.getElementById('progress-fill');
    if (fill) fill.style.width = '100%';

    const genLabel = document.getElementById('gen-counter-label');
    if (genLabel) genLabel.innerHTML = `Gen <span>${metrics.generations_executed}</span> / ${params.max_generations}`;

    const genLabelBar = document.getElementById('gen-counter-label-bar');
    if (genLabelBar) genLabelBar.innerHTML = `Gen <span>${metrics.generations_executed}</span> / ${params.max_generations}`;

    // Badge
    const badge = document.getElementById('conv-badge');
    if (badge) {
      badge.textContent = '🐍 Python Certificado';
      badge.className   = 'badge badge-success';
    }

    // Comparador
    const compFitness = document.getElementById('comp-vrp-fitness');
    const compDist    = document.getElementById('comp-vrp-dist');
    if (compFitness) compFitness.textContent = metrics.best_fitness.toFixed(5);
    if (compDist)    compDist.textContent    = metrics.total_distance_km.toFixed(1) + ' km';

    // Tabla de top individuos generada por Python
    const tbody = document.getElementById('pop-tbody');
    if (tbody && top_individuals) {
      tbody.innerHTML = top_individuals.map(ind => {
        const barPct = (ind.aptitud * 100).toFixed(1);
        return `
          <tr>
            <td>#${ind.rank}</td>
            <td>${ind.fitness.toFixed(6)}</td>
            <td>${(ind.aptitud * 100).toFixed(2)}%</td>
            <td>
              <div class="fitness-bar-cell">
                <div class="fitness-bar-mini">
                  <div class="fitness-bar-mini-fill" style="width:${barPct}%"></div>
                </div>
                <span>${barPct}%</span>
              </div>
            </td>
          </tr>`;
      }).join('');
    }

    this._setStatus('ready');
  }

  /* ── Tabs ────────────────────────────────────────────────── */
  _bindTabs() {
    document.querySelectorAll('.tab-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const tab = btn.dataset.tab;
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
        btn.classList.add('active');
        const panel = document.getElementById(`tab-${tab}`);
        if (panel) panel.classList.add('active');

        // Re-dibujar canvas si volvemos al tab VRP
        if (tab === 'vrp') setTimeout(() => this._redrawCanvas(), 50);
      });
    });
  }

  /* ── Modo Cátedra / Libre ────────────────────────────────── */
  _bindModeToggle() {
    document.querySelectorAll('.mode-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const mode = btn.dataset.mode;
        this.mode  = mode;
        document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        this._syncSliderLock();
      });
    });
  }

  _syncSliderLock() {
    const locked = this.mode === 'catedra';
    document.querySelectorAll('.param-slider').forEach(sl => {
      sl.disabled = locked;
    });
    document.querySelector('.btn-restore')?.classList.toggle('hidden', !locked);
  }

  /* ── Sliders ─────────────────────────────────────────────── */
  _bindSliders() {
    /** @type {[string, string, (v:number)=>any][]} */
    const sliders = [
      ['sl-popsize', 'val-popsize', v => { this.params.popSize  = parseInt(v); return v; }],
      ['sl-elitism', 'val-elitism', v => { this.params.elitism  = parseInt(v); return v; }],
      ['sl-pc',      'val-pc',      v => { this.params.pc        = parseFloat(v) / 100; return parseFloat(v).toFixed(0) + '%'; }],
      ['sl-pm',      'val-pm',      v => { this.params.pm        = parseFloat(v) / 100; return parseFloat(v).toFixed(1) + '%'; }],
      ['sl-maxgens', 'val-maxgens', v => { this.params.maxGens   = parseInt(v); return v; }],
    ];

    sliders.forEach(([slId, valId, fn]) => {
      const sl  = document.getElementById(slId);
      const val = document.getElementById(valId);
      if (!sl || !val) return;
      sl.addEventListener('input', () => {
        val.textContent = fn(sl.value);
      });
    });

    // Speed slider (no afecta al GA, sólo la animación)
    const slSpeed  = document.getElementById('sl-speed');
    const valSpeed = document.getElementById('val-speed');
    if (slSpeed) {
      slSpeed.addEventListener('input', () => {
        // 1 = más rápido (10ms), 10 = más lento (500ms)
        this.speedMs        = Math.round(10 * (10 / parseInt(slSpeed.value)));
        if (valSpeed) valSpeed.textContent = slSpeed.value + 'x';
      });
    }
  }

  /* ── Controles principales ───────────────────────────────── */
  _bindControls() {
    document.getElementById('btn-run')?.addEventListener('click',     () => this._run());
    document.getElementById('btn-pause')?.addEventListener('click',   () => this._pause());
    document.getElementById('btn-step')?.addEventListener('click',    () => this._stepOnce());
    document.getElementById('btn-reset')?.addEventListener('click',   () => this._reset());
    document.getElementById('btn-restore')?.addEventListener('click', () => this._restoreCatedra());
    document.getElementById('btn-restore-side')?.addEventListener('click', () => this._restoreCatedra());

    // Exportar mapa como imagen PNG
    document.getElementById('btn-export-map')?.addEventListener('click', () => {
      if (!this.renderer?.canvas) return;
      const link = document.createElement('a');
      link.download = `vrp_rutas_gen_${this.ga?.generation || 0}.png`;
      link.href = this.renderer.canvas.toDataURL('image/png');
      link.click();
    });
  }

  /* ── Lógica de ejecución ─────────────────────────────────── */
  _run() {
    if (this.running) return;
    if (this.ga.converged || this.ga.generation >= this.params.maxGens || this.ga.generation === 0) {
      this._resetGA();
    }
    this.running  = true;
    this.stepMode = false;
    this._setStatus('running');
    this._loop(performance.now());
  }

  _pause() {
    this.running = false;
    cancelAnimationFrame(this.rafId);
    this._setStatus('ready');
    this._updateUI();
  }

  _stepOnce() {
    if (this.running) return;
    this._advanceGA();
    this._updateUI();
  }

  _reset() {
    this._pause();
    this._resetGA();
    this.chart.clear();
    this._redrawCanvas();
    this._updateUI();
    this._setStatus('ready');
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
    if (this.pyData) {
      this._loadPythonSolution();
      this._updateUIFromPython();
    } else {
      this._resetGA();
      this.chart.clear();
      this._redrawCanvas();
      this._updateUI();
    }
    this._setStatus('ready');
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
        this._setStatus('done');
        return;
      }
    }

    this.rafId = requestAnimationFrame(t => this._loop(t));
  }

  /* ── Un paso del GA ──────────────────────────────────────── */
  _advanceGA() {
    const done = this.ga.step();
    // Dibujar rutas actuales
    const sol = this.ga.getBestSolution();
    this.renderer.draw(sol.routes);
    // Actualizar gráfica
    this.chart.update(this.ga.history);
    return done;
  }

  _resetGA() {
    this.ga = new SimpleGeneticAlgorithm(this.params, VRP_INSTANCE);
    // Dibujar estado inicial (sin rutas)
    this.renderer.draw(null);
  }

  _redrawCanvas() {
    if (this.mode === 'catedra' && this.pythonRoutes) {
      this.renderer.draw(this.pythonRoutes);
      return;
    }
    if (!this.ga) return;
    const sol = this.ga.generation > 0 ? this.ga.getBestSolution() : null;
    this.renderer.draw(sol ? sol.routes : null);
  }

  /* ── Actualización de KPIs y tabla ──────────────────────── */
  _updateUI() {
    if (!this.ga) return;
    const { generation, bestFitness, bestAptitud, history, params } = this.ga;
    const sol = generation > 0 ? this.ga.getBestSolution() : null;

    // KPIs
    this._setText('kpi-fitness',    bestFitness <= 1e-7 ? '—' : bestFitness.toFixed(5));
    this._setText('kpi-dist',       sol ? sol.totalDist.toFixed(1) + ' km' : '—');
    this._setText('kpi-feasible',   sol && sol.penalty === 0 ? '100%' : sol ? 'Penaliz.' : '—');
    this._setText('kpi-gen',        `${generation} / ${this.params.maxGens}`);
    this._setText('kpi-aptitud',    (bestAptitud * 100).toFixed(2) + '%');

    // Progreso
    const pct = Math.min(100, (generation / this.params.maxGens) * 100);
    const fill = document.getElementById('progress-fill');
    if (fill) fill.style.width = pct + '%';

    const genLabel = document.getElementById('gen-counter-label');
    if (genLabel) genLabel.innerHTML = `Gen <span>${generation}</span> / ${this.params.maxGens}`;

    const genLabelBar = document.getElementById('gen-counter-label-bar');
    if (genLabelBar) genLabelBar.innerHTML = `Gen <span>${generation}</span> / ${this.params.maxGens}`;

    // Sincronizar datos en el comparador
    const compFitness = document.getElementById('comp-vrp-fitness');
    const compDist    = document.getElementById('comp-vrp-dist');
    if (compFitness && bestFitness > 1e-7) compFitness.textContent = bestFitness.toFixed(5);
    if (compDist && sol) compDist.textContent = sol.totalDist.toFixed(1) + ' km';

    // Botones
    const btnRun   = document.getElementById('btn-run');
    const btnPause = document.getElementById('btn-pause');
    const btnStep  = document.getElementById('btn-step');
    if (btnRun)   btnRun.disabled   = this.running;
    if (btnPause) btnPause.disabled = !this.running;
    if (btnStep)  btnStep.disabled  = this.running;

    // Tabla de top individuos
    this._updateTable();

    // Badge de convergencia
    const badge = document.getElementById('conv-badge');
    if (badge) {
      if (this.ga.converged && bestFitness <= this.params.threshold) {
        badge.textContent = '✓ Convergió';
        badge.className   = 'badge badge-success';
      } else if (generation >= this.params.maxGens) {
        badge.textContent = '⚠ Máx. gens';
        badge.className   = 'badge badge-warning';
      } else if (generation > 0) {
        badge.textContent = `▶ Gen ${generation}`;
        badge.className   = 'badge badge-info';
      } else {
        badge.textContent = '● Listo';
        badge.className   = 'badge badge-info';
      }
    }
  }

  _updateTable() {
    const tbody = document.getElementById('pop-tbody');
    if (!tbody || !this.ga) return;

    const top = this.ga.getTopK(8);
    tbody.innerHTML = top.map((ind, rank) => {
      const aptitud = Math.max(0, 1 - ind.fitness);
      const barPct  = Math.min(100, aptitud * 100).toFixed(1);
      return `
        <tr>
          <td>#${rank + 1}</td>
          <td>${ind.fitness.toFixed(6)}</td>
          <td>${(aptitud * 100).toFixed(2)}%</td>
          <td>
            <div class="fitness-bar-cell">
              <div class="fitness-bar-mini">
                <div class="fitness-bar-mini-fill" style="width:${barPct}%"></div>
              </div>
              <span>${barPct}%</span>
            </div>
          </td>
        </tr>`;
    }).join('');
  }

  _setText(id, txt) {
    const el = document.getElementById(id);
    if (el) el.textContent = txt;
  }

  _setStatus(state) {
    const dot   = document.getElementById('status-dot');
    const label = document.getElementById('status-label');
    const map   = {
      ready:   ['ready',   'Listo'],
      running: ['running', 'Ejecutando...'],
      done:    ['done',    'Convergido'],
    };
    const [cls, text] = map[state] || ['ready', 'Listo'];
    if (dot)   { dot.className = 'status-dot ' + cls; }
    if (label) { label.textContent = text; }
  }
}

/* ════════════════════════════════════════════════════════════
   10. INICIO — Esperar al DOM y a Chart.js
   ════════════════════════════════════════════════════════════ */

document.addEventListener('DOMContentLoaded', () => {
  // Chart.js debe estar cargado (CDN en index.html)
  if (typeof Chart === 'undefined') {
    console.error('[EvoSuite] Chart.js no encontrado. Verificar CDN en index.html.');
    return;
  }

  // Registrar plugins de Chart.js necesarios
  Chart.defaults.color = '#94A3B8';
  Chart.defaults.font.family = "'Fira Code', monospace";

  // Iniciar controlador
  window._dashboard = new DashboardController();

  console.log('[EvoSuite AI] Dashboard inicializado. Motor SGA listo.');
  console.log('[EvoSuite AI] Instancia VRP: 15 clientes, 3 camiones, cap=120kg.');
  console.log('[EvoSuite AI] Caso base cátedra: N=1000, e=2, Pc=90%, Pm=2%, maxGens=150.');
});
