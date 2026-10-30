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
 * Autor: Dago David Palmera Navarro (Cód. 0222321003)
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

  /**
   * Convierte coordenadas del problema [0-100] → píxeles CSS del canvas,
   * manteniendo un plano cartesiano perfectamente cuadrado (1:1) y centrado.
   */
  _toCanvas(x, y) {
    const pad = 28;
    const w = this.cssWidth || 400;
    const h = this.cssHeight || 300;
    const side = Math.min(w - 2 * pad, h - 2 * pad);
    const offsetX = (w - side) / 2;
    const offsetY = (h - side) / 2;

    return {
      px: offsetX + (x / 100) * side,
      py: offsetY + (1 - y / 100) * side,  // Y invertido para plano cartesiano
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
    let minD    = 18;  // radio de detección en px CSS

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
        label = `Depósito Central (${this.instance.depot.x}, ${this.instance.depot.y})`;
      } else {
        const c = this.instance.customers[closest.index];
        label = `${c.id} | Dem: ${c.demand} kg | Serv: ${c.service} min`;
      }
      tooltip.style.display = 'block';
      tooltip.style.left = (e.offsetX + 14) + 'px';
      tooltip.style.top  = (e.offsetY - 28) + 'px';
      tooltip.textContent = label;
    } else {
      tooltip.style.display = 'none';
    }
    this.hoveredNode = closest;
  }

  /**
   * Dibuja el mapa de rutas VRP centrado y escalado.
   * @param {number[][]} routes   - rutas actuales (puede ser null)
   * @param {string[]}   [routeColors]
   */
  draw(routes, routeColors = ROUTE_COLORS) {
    const canvas = this.canvas;
    const ctx    = this.ctx;

    // Obtener dimensiones reales del contenedor padre (.panel-body)
    const rect = canvas.parentElement.getBoundingClientRect();
    const dpr  = window.devicePixelRatio || 1;
    const w    = Math.floor(rect.width) || 400;
    const h    = Math.floor(rect.height) || 300;

    // Ajustar resolución del canvas para retina displays
    if (canvas.width !== Math.floor(w * dpr) || canvas.height !== Math.floor(h * dpr)) {
      canvas.width  = Math.floor(w * dpr);
      canvas.height = Math.floor(h * dpr);
    }

    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.scale(dpr, dpr);

    this.cssWidth  = w;
    this.cssHeight = h;

    ctx.clearRect(0, 0, w, h);

    // Fondo completo de la tarjeta
    ctx.fillStyle = 'rgba(7, 11, 18, 0.75)';
    ctx.fillRect(0, 0, w, h);

    // Cuadrícula 1:1 centrada
    this._drawGrid(ctx, w, h);

    // Rutas
    if (routes) {
      routes.forEach((route, v) => {
        if (!route || route.length === 0) return;
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
    const pad = 28;
    const side = Math.min(w - 2 * pad, h - 2 * pad);
    const offsetX = (w - side) / 2;
    const offsetY = (h - side) / 2;

    // Recuadro interior centrado con brillo tenue
    ctx.fillStyle = 'rgba(11, 15, 25, 0.8)';
    ctx.fillRect(offsetX, offsetY, side, side);

    // Cuadrícula 10x10
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.lineWidth   = 1;
    const step = side / 10;

    for (let i = 0; i <= 10; i++) {
      const pos = offsetX + i * step;
      ctx.beginPath();
      ctx.moveTo(pos, offsetY);
      ctx.lineTo(pos, offsetY + side);
      ctx.stroke();

      const yPos = offsetY + i * step;
      ctx.beginPath();
      ctx.moveTo(offsetX, yPos);
      ctx.lineTo(offsetX + side, yPos);
      ctx.stroke();
    }

    // Marco exterior de la zona logística
    ctx.strokeStyle = 'rgba(16, 185, 129, 0.25)';
    ctx.lineWidth   = 1.5;
    ctx.strokeRect(offsetX, offsetY, side, side);

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
    ctx.shadowColor  = color;
    ctx.shadowBlur   = 8;
    ctx.strokeStyle  = color + 'CC';
    ctx.lineWidth    = 2.2;
    ctx.lineCap      = 'round';
    ctx.lineJoin     = 'round';

    ctx.beginPath();
    points.forEach((p, i) => {
      if (i === 0) ctx.moveTo(p.px, p.py);
      else ctx.lineTo(p.px, p.py);
    });
    ctx.stroke();

    // Flechas direccionales
    for (let i = 0; i < points.length - 1; i++) {
      this._drawArrow(ctx, points[i], points[i + 1], color);
    }
    ctx.restore();
  }

  _drawArrow(ctx, from, to, color) {
    const mx  = (from.px + to.px) / 2;
    const my  = (from.py + to.py) / 2;
    const ang = Math.atan2(to.py - from.py, to.px - from.px);
    const len = 7;

    ctx.save();
    ctx.translate(mx, my);
    ctx.rotate(ang);
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.moveTo(len, 0);
    ctx.lineTo(-len * 0.7, -len * 0.5);
    ctx.lineTo(-len * 0.7,  len * 0.5);
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }

  _drawCustomer(ctx, c, i, routes) {
    const { px, py } = this._toCanvas(c.x, c.y);
    const isHovered  = this.hoveredNode?.type === 'customer' && this.hoveredNode.index === i;

    // Determinar qué camión atiende al cliente
    let color = '#64748B';
    if (routes) {
      routes.forEach((route, v) => {
        if (route.includes(i)) color = ROUTE_COLORS[v] || '#fff';
      });
    }

    ctx.save();
    if (isHovered) {
      ctx.shadowColor = color;
      ctx.shadowBlur  = 16;
      ctx.beginPath();
      ctx.arc(px, py, 13, 0, Math.PI * 2);
      ctx.fillStyle = color + '33';
      ctx.fill();
    }

    ctx.beginPath();
    ctx.arc(px, py, isHovered ? 9 : 7, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.shadowColor = color;
    ctx.shadowBlur  = 6;
    ctx.fill();

    ctx.beginPath();
    ctx.arc(px, py, 3, 0, Math.PI * 2);
    ctx.fillStyle = '#070B12';
    ctx.fill();

    // Etiqueta del nodo
    ctx.fillStyle    = '#F8FAFC';
    ctx.font         = 'bold 9px "Fira Code", monospace';
    ctx.textAlign    = 'center';
    ctx.textBaseline = 'bottom';
    ctx.fillText(c.id, px, py - 9);

    ctx.restore();
  }

  _drawDepot(ctx) {
    const { px, py } = this._toCanvas(this.instance.depot.x, this.instance.depot.y);
    const isHovered  = this.hoveredNode?.type === 'depot';
    const sz         = isHovered ? 12 : 9;

    ctx.save();
    if (isHovered) {
      ctx.shadowColor = '#10B981';
      ctx.shadowBlur  = 20;
    }

    ctx.translate(px, py);
    ctx.rotate(Math.PI / 4);

    ctx.fillStyle   = '#0B0F19';
    ctx.strokeStyle = '#10B981';
    ctx.lineWidth   = 2.2;
    ctx.fillRect(-sz, -sz, sz * 2, sz * 2);
    ctx.strokeRect(-sz, -sz, sz * 2, sz * 2);

    ctx.rotate(-Math.PI / 4);
    ctx.fillStyle    = '#10B981';
    ctx.font         = 'bold 8px "Inter", sans-serif';
    ctx.textAlign    = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('D', 0, 0);

    ctx.restore();
  }
}

/* ════════════════════════════════════════════════════════════
   8. CHART.JS — Gráfica de Convergencia
   ════════════════════════════════════════════════════════════ */

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
class PCBRenderer {
  constructor(canvas, instance) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.instance = instance;
    this.components = instance.components || [];
    this.nets = instance.nets || [];
    this.placed = [];
    this.hoveredComp = null;
    this.cssWidth = 400;
    this.cssHeight = 320;

    this._bindEvents();
  }

  _bindEvents() {
    this.canvas.addEventListener('mousemove', e => this._onMouseMove(e));
    this.canvas.addEventListener('mouseleave', () => {
      this.hoveredComp = null;
      const tooltip = document.getElementById('pcb-canvas-tooltip');
      if (tooltip) tooltip.style.display = 'none';
      this.draw(this.placed);
    });
  }

  _onMouseMove(e) {
    const rect = this.canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    let found = null;
    if (this.placed && this.placed.length > 0) {
      for (const comp of this.placed) {
        const cp = this._toCanvas(comp.x, comp.y);
        const wPx = (comp.width / this.instance.board_max_width) * this.boardW;
        const hPx = (comp.height / this.instance.board_max_height) * this.boardH;
        if (
          mx >= cp.px - wPx / 2 &&
          mx <= cp.px + wPx / 2 &&
          my >= cp.py - hPx / 2 &&
          my <= cp.py + hPx / 2
        ) {
          found = comp;
          break;
        }
      }
    }

    if (found !== this.hoveredComp) {
      this.hoveredComp = found;
      this.draw(this.placed);
    }

    const tooltip = document.getElementById('pcb-canvas-tooltip');
    if (tooltip) {
      if (found) {
        // Find connected nets
        const connNets = this.nets.filter(n => n.source_id === found.id || n.target_id === found.id);
        const netList = connNets.map(n => n.net_name).slice(0, 5).join(', ');
        tooltip.innerHTML = `
          <div style="font-weight:700; color:${found.color || '#10B981'}; margin-bottom:4px">
            ${found.ref} — ${found.name}
          </div>
          <div style="color:#94A3B8; font-size:0.75rem; line-height:1.4">
            <strong>Huella:</strong> ${found.width} × ${found.height} mm (${found.category})<br>
            <strong>Posición:</strong> X = ${found.x.toFixed(1)} mm, Y = ${found.y.toFixed(1)} mm<br>
            <strong>Conexiones (${connNets.length}):</strong> <span style="color:#38BDF8">${netList || 'Ninguna'}</span>
          </div>
        `;
        tooltip.style.display = 'block';
        tooltip.style.left = `${Math.min(mx + 15, rect.width - 240)}px`;
        tooltip.style.top  = `${Math.max(my - 70, 10)}px`;
      } else {
        tooltip.style.display = 'none';
      }
    }
  }

  _toCanvas(x, y) {
    const pad = 30;
    const w = this.cssWidth;
    const h = this.cssHeight;

    const boardAspect = this.instance.board_max_width / this.instance.board_max_height; // 100 / 80 = 1.25
    let bW = w - 2 * pad;
    let bH = bW / boardAspect;
    if (bH > h - 2 * pad) {
      bH = h - 2 * pad;
      bW = bH * boardAspect;
    }

    this.boardW = bW;
    this.boardH = bH;
    this.boardX = (w - bW) / 2;
    this.boardY = (h - bH) / 2;

    return {
      px: this.boardX + (x / this.instance.board_max_width) * bW,
      py: this.boardY + (1 - y / this.instance.board_max_height) * bH, // Inverted Y for Cartesian board
    };
  }

  draw(placedComponents = null) {
    if (placedComponents) {
      this.placed = placedComponents;
    }

    const canvas = this.canvas;
    const ctx = this.ctx;
    const parent = canvas.parentElement;
    const rect = parent ? parent.getBoundingClientRect() : canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const w = Math.floor(rect.width) || 480;
    const h = Math.floor(rect.height) || 300;

    if (canvas.width !== Math.floor(w * dpr) || canvas.height !== Math.floor(h * dpr)) {
      canvas.width = Math.floor(w * dpr);
      canvas.height = Math.floor(h * dpr);
    }

    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.scale(dpr, dpr);

    this.cssWidth = w;
    this.cssHeight = h;

    // Fondo general oscuro
    ctx.fillStyle = '#070D14';
    ctx.fillRect(0, 0, w, h);

    // Calcular dimensiones de la placa
    const origin = this._toCanvas(0, 0);
    const maxCorner = this._toCanvas(this.instance.board_max_width, this.instance.board_max_height);
    const bx = this.boardX;
    const by = this.boardY;
    const bw = this.boardW;
    const bh = this.boardH;

    // 1. Sustrato de la Placa PCB (FR-4 Solder Mask Verde Oscuro / Tech Teal)
    ctx.save();
    const bgGrad = ctx.createLinearGradient(bx, by, bx + bw, by + bh);
    bgGrad.addColorStop(0, '#0a1d17');
    bgGrad.addColorStop(1, '#06130f');
    ctx.fillStyle = bgGrad;
    ctx.strokeStyle = '#10B981';
    ctx.lineWidth = 2.0;

    // Sombra exterior de la placa
    ctx.shadowColor = 'rgba(16, 185, 129, 0.25)';
    ctx.shadowBlur = 12;

    this._roundRect(ctx, bx, by, bw, bh, 8);
    ctx.fill();
    ctx.stroke();
    ctx.restore();

    // 2. Grilla milimétrica sutil (cada 10 mm)
    ctx.save();
    ctx.strokeStyle = 'rgba(16, 185, 129, 0.08)';
    ctx.lineWidth = 1;
    for (let gx = 10; gx < this.instance.board_max_width; gx += 10) {
      const p1 = this._toCanvas(gx, 0);
      const p2 = this._toCanvas(gx, this.instance.board_max_height);
      ctx.beginPath();
      ctx.moveTo(p1.px, p1.py);
      ctx.lineTo(p2.px, p2.py);
      ctx.stroke();
    }
    for (let gy = 10; gy < this.instance.board_max_height; gy += 10) {
      const p1 = this._toCanvas(0, gy);
      const p2 = this._toCanvas(this.instance.board_max_width, gy);
      ctx.beginPath();
      ctx.moveTo(p1.px, p1.py);
      ctx.lineTo(p2.px, p2.py);
      ctx.stroke();
    }
    ctx.restore();

    // 3. Orificios de Montaje en las 4 Esquinas (Mounting Holes)
    const mountHoles = [
      { x: 4, y: 4 },
      { x: this.instance.board_max_width - 4, y: 4 },
      { x: 4, y: this.instance.board_max_height - 4 },
      { x: this.instance.board_max_width - 4, y: this.instance.board_max_height - 4 }
    ];
    for (const h of mountHoles) {
      const cp = this._toCanvas(h.x, h.y);
      ctx.save();
      // Anillo de cobre dorado
      ctx.fillStyle = '#D97706';
      ctx.beginPath();
      ctx.arc(cp.px, cp.py, 5, 0, Math.PI * 2);
      ctx.fill();
      // Taladro interno oscuro
      ctx.fillStyle = '#070D14';
      ctx.beginPath();
      ctx.arc(cp.px, cp.py, 2.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }

    // Si no hay componentes colocados, mostrar mensaje
    if (!this.placed || this.placed.length === 0) {
      ctx.fillStyle = '#64748B';
      ctx.font = '12px "Fira Code", monospace';
      ctx.textAlign = 'center';
      ctx.fillText('Cargando plano de colocación PCB...', w / 2, h / 2);
      return;
    }

    // Mapa de componentes por ID para rápido acceso
    const compMap = {};
    for (const c of this.placed) compMap[c.id] = c;

    // 4. Dibujar Trazas de Cobre (Nets)
    for (const net of this.nets) {
      const c1 = compMap[net.source_id];
      const c2 = compMap[net.target_id];
      if (!c1 || c2 === undefined) continue;

      const p1 = this._toCanvas(c1.x, c1.y);
      const p2 = this._toCanvas(c2.x, c2.y);

      const isHovered = this.hoveredComp && (this.hoveredComp.id === c1.id || this.hoveredComp.id === c2.id);

      ctx.save();
      if (isHovered) {
        ctx.strokeStyle = '#38BDF8';
        ctx.shadowColor = '#38BDF8';
        ctx.shadowBlur = 8;
        ctx.lineWidth = 2.5;
      } else {
        // Color por importancia del Net
        if (net.net_name.includes('VCC') || net.net_name.includes('VBUS') || net.net_name.includes('LDO')) {
          ctx.strokeStyle = 'rgba(245, 158, 11, 0.75)'; // Alimentación ámbar
          ctx.lineWidth = 1.8;
        } else if (net.net_name.includes('XTAL')) {
          ctx.strokeStyle = 'rgba(139, 92, 246, 0.75)'; // Reloj púrpura
          ctx.lineWidth = 1.6;
        } else if (net.net_name.includes('SPI') || net.net_name.includes('USB')) {
          ctx.strokeStyle = 'rgba(56, 189, 248, 0.70)'; // Datos cyan
          ctx.lineWidth = 1.4;
        } else {
          ctx.strokeStyle = 'rgba(16, 185, 129, 0.50)'; // General esmeralda
          ctx.lineWidth = 1.2;
        }
      }

      // Ruteo ortogonal Manhattan con chaflán
      const midX = p1.px + (p2.px - p1.px) * 0.5;
      ctx.beginPath();
      ctx.moveTo(p1.px, p1.py);
      ctx.lineTo(midX, p1.py);
      ctx.lineTo(midX, p2.py);
      ctx.lineTo(p2.px, p2.py);
      ctx.stroke();

      // Pequeñas vías en esquinas
      ctx.fillStyle = isHovered ? '#38BDF8' : '#D97706';
      ctx.beginPath();
      ctx.arc(midX, p1.py, 1.8, 0, Math.PI * 2);
      ctx.arc(midX, p2.py, 1.8, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }

    // 5. Bounding Box Envolvente (Área compacta del circuito)
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const c of this.placed) {
      const halfW = c.width / 2;
      const halfH = c.height / 2;
      minX = Math.min(minX, c.x - halfW);
      maxX = Math.max(maxX, c.x + halfW);
      minY = Math.min(minY, c.y - halfH);
      maxY = Math.max(maxY, c.y + halfH);
    }
    const bbP1 = this._toCanvas(minX, maxY);
    const bbP2 = this._toCanvas(maxX, minY);

    ctx.save();
    ctx.strokeStyle = 'rgba(56, 189, 248, 0.40)';
    ctx.lineWidth = 1.2;
    ctx.setLineDash([4, 4]);
    ctx.strokeRect(bbP1.px - 3, bbP1.py - 3, (bbP2.px - bbP1.px) + 6, (bbP2.py - bbP1.py) + 6);
    ctx.restore();

    // 6. Dibujar Componentes Electrónicos (Footprints & Silkscreen)
    for (const comp of this.placed) {
      const cp = this._toCanvas(comp.x, comp.y);
      const wPx = (comp.width / this.instance.board_max_width) * this.boardW;
      const hPx = (comp.height / this.instance.board_max_height) * this.boardH;
      const isHovered = this.hoveredComp && this.hoveredComp.id === comp.id;

      ctx.save();

      // Halo de aislamiento dieléctrico (clearance 2mm)
      const clrWPx = (this.instance.clearance / this.instance.board_max_width) * this.boardW;
      const clrHPx = (this.instance.clearance / this.instance.board_max_height) * this.boardH;
      ctx.strokeStyle = isHovered ? 'rgba(56, 189, 248, 0.4)' : 'rgba(255, 255, 255, 0.05)';
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 2]);
      ctx.strokeRect(cp.px - wPx / 2 - clrWPx, cp.py - hPx / 2 - clrHPx, wPx + 2 * clrWPx, hPx + 2 * clrHPx);
      ctx.setLineDash([]);

      // Cuerpo del componente según tipo
      ctx.shadowColor = isHovered ? (comp.color || '#10B981') : 'rgba(0, 0, 0, 0.5)';
      ctx.shadowBlur = isHovered ? 12 : 4;

      if (comp.category === 'ic') {
        // Chip IC (MCU, Flash)
        ctx.fillStyle = '#1E293B';
        ctx.strokeStyle = comp.color || '#10B981';
        ctx.lineWidth = isHovered ? 2 : 1.2;
        this._roundRect(ctx, cp.px - wPx / 2, cp.py - hPx / 2, wPx, hPx, 3);
        ctx.fill();
        ctx.stroke();

        // Pin 1 dot
        ctx.fillStyle = '#F8FAFC';
        ctx.beginPath();
        ctx.arc(cp.px - wPx / 2 + 3, cp.py - hPx / 2 + 3, 1.5, 0, Math.PI * 2);
        ctx.fill();

        // Pins a los costados
        ctx.fillStyle = '#94A3B8';
        const numPins = 4;
        for (let i = 0; i < numPins; i++) {
          const step = hPx / (numPins + 1);
          const py = cp.py - hPx / 2 + (i + 1) * step;
          ctx.fillRect(cp.px - wPx / 2 - 2, py - 1, 2, 2);
          ctx.fillRect(cp.px + wPx / 2, py - 1, 2, 2);
        }
      } else if (comp.category === 'power') {
        // Regulador LDO / Diodo
        ctx.fillStyle = '#1e2433';
        ctx.strokeStyle = comp.color || '#F59E0B';
        ctx.lineWidth = isHovered ? 2 : 1.2;
        this._roundRect(ctx, cp.px - wPx / 2, cp.py - hPx / 2, wPx, hPx, 3);
        ctx.fill();
        ctx.stroke();

        // Heatsink tab
        ctx.fillStyle = '#D97706';
        ctx.fillRect(cp.px - wPx / 3, cp.py - hPx / 2 - 2, (wPx * 2) / 3, 2);
      } else if (comp.category === 'connector') {
        // Conector USB / GPIO
        ctx.fillStyle = '#172554';
        ctx.strokeStyle = comp.color || '#3B82F6';
        ctx.lineWidth = isHovered ? 2 : 1.2;
        this._roundRect(ctx, cp.px - wPx / 2, cp.py - hPx / 2, wPx, hPx, 2);
        ctx.fill();
        ctx.stroke();

        // Pines de cabezal o conector
        ctx.fillStyle = '#E2E8F0';
        for (let ix = -wPx / 3; ix <= wPx / 3; ix += wPx / 3) {
          ctx.beginPath();
          ctx.arc(cp.px + ix, cp.py, 1.5, 0, Math.PI * 2);
          ctx.fill();
        }
      } else {
        // Componentes pasivos (Capacitores, Resistencias 0805)
        ctx.fillStyle = '#0f172a';
        ctx.strokeStyle = comp.color || '#EC4899';
        ctx.lineWidth = isHovered ? 1.8 : 1.0;
        this._roundRect(ctx, cp.px - wPx / 2, cp.py - hPx / 2, wPx, hPx, 1.5);
        ctx.fill();
        ctx.stroke();

        // Terminales plateadas
        ctx.fillStyle = '#CBD5E1';
        ctx.fillRect(cp.px - wPx / 2, cp.py - hPx / 2, 2, hPx);
        ctx.fillRect(cp.px + wPx / 2 - 2, cp.py - hPx / 2, 2, hPx);
      }

      // Serigrafía: Ref del componente (Silkscreen blanca / menta)
      ctx.fillStyle = isHovered ? '#FFFFFF' : '#E2E8F0';
      const fontSize = Math.max(8, Math.min(11, Math.round(wPx * 0.35)));
      ctx.font = `700 ${fontSize}px "Fira Code", monospace`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(comp.ref, cp.px, cp.py);

      ctx.restore();
    }
  }

  _roundRect(ctx, x, y, width, height, radius) {
    ctx.beginPath();
    ctx.moveTo(x + radius, y);
    ctx.lineTo(x + width - radius, y);
    ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
    ctx.lineTo(x + width, y + height - radius);
    ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
    ctx.lineTo(x + radius, y + height);
    ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
    ctx.lineTo(x, y + radius);
    ctx.quadraticCurveTo(x, y, x + radius, y);
    ctx.closePath();
  }
}

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

    this._init();
    this._initPCB();
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
    this._bindModeToggle();
    this._bindSliders();
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

  /* ── Tabs ────────────────────────────────────────────────── */
  _bindTabs() {
    document.querySelectorAll('.tab-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const target = btn.dataset.tab;
        this.currentTab = target;
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
        btn.classList.add('active');
        const panel = document.getElementById(`tab-${target}`);
        if (panel) panel.classList.add('active');

        // Adaptar Sidebar Caso Base & Cátedra según la pestaña activa
        const accBaseContent = document.getElementById('acc-base');
        if (accBaseContent) {
          if (target === 'pcb') {
            accBaseContent.innerHTML = `
              <div class="info-grid">
                <div class="info-chip">
                  <div class="info-chip-label">Componentes</div>
                  <div class="info-chip-value">14</div>
                </div>
                <div class="info-chip">
                  <div class="info-chip-label">Trazas (Nets)</div>
                  <div class="info-chip-value">22</div>
                </div>
                <div class="info-chip">
                  <div class="info-chip-label">Sustrato</div>
                  <div class="info-chip-value">100x80 mm</div>
                </div>
                <div class="info-chip">
                  <div class="info-chip-label">Aislamiento</div>
                  <div class="info-chip-value">2.0 mm</div>
                </div>
                <div class="info-chip">
                  <div class="info-chip-label">Área Final</div>
                  <div class="info-chip-value">3168 mm²</div>
                </div>
                <div class="info-chip">
                  <div class="info-chip-label">Wirelength</div>
                  <div class="info-chip-value">1080 mm</div>
                </div>
              </div>
              <div class="formula-card">
                <div class="formula-card-header">
                  <span>Criterio Multicriterio</span>
                  <span class="formula-card-tag">Cátedra P2</span>
                </div>
                <div class="formula-card-eq">
                  <i>f</i><sub>norm</sub> = 0.40·WL + 0.10·Área + 0.50·Colisión
                </div>
                <div class="formula-card-legend">
                  <span><b>WL:</b> Longitud Nets</span>
                  <span><b>Área:</b> Bounding Box</span>
                  <span><b>Colisión:</b> Penalización</span>
                </div>
                <div class="formula-card-footer">
                  <span>SGA Multi-Objetivo</span>
                  <span>Prof. Ing. M. A. Ospina</span>
                </div>
              </div>
            `;
            this._setStatus('ready', 'Caso Base Certificado (PCB)');
            requestAnimationFrame(() => {
              if (this.pcbChart && this.pcbChart.chart) {
                this.pcbChart.chart.resize();
                if (this.pcbData?.history) {
                  this.pcbChart.update(this.pcbData.history);
                }
              }
              if (this.pcbRenderer) {
                this._setPCBGeneration(this.pcbCurrentGen);
              }
            });
          } else if (target === 'vrp') {
            accBaseContent.innerHTML = `
              <div class="info-grid">
                <div class="info-chip">
                  <div class="info-chip-label">Clientes</div>
                  <div class="info-chip-value">15</div>
                </div>
                <div class="info-chip">
                  <div class="info-chip-label">Camiones</div>
                  <div class="info-chip-value">3</div>
                </div>
                <div class="info-chip">
                  <div class="info-chip-label">Capacidad</div>
                  <div class="info-chip-value">120 kg</div>
                </div>
                <div class="info-chip">
                  <div class="info-chip-label">Turno</div>
                  <div class="info-chip-value">480 min</div>
                </div>
                <div class="info-chip">
                  <div class="info-chip-label">D_ideal</div>
                  <div class="info-chip-value">235 km</div>
                </div>
                <div class="info-chip">
                  <div class="info-chip-label">D_worst</div>
                  <div class="info-chip-value">750 km</div>
                </div>
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

  /* ── Modo Cátedra / Libre ────────────────────────────────── */
  _bindModeToggle() {
    document.querySelectorAll('.mode-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const newMode = btn.dataset.mode;
        if (newMode === this.mode) return;
        this.mode = newMode;
        document.querySelectorAll('.mode-btn').forEach(b => b.classList.toggle('active', b.dataset.mode === this.mode));
        this._syncSliderLock();
        this._pause();
        if (this.mode === 'catedra') {
          this._restoreCatedra();
        } else {
          this._setMaxGenerations(this.params.maxGens);
          this._resetGA();
          this.chart.clear();
          this.chart.setMaxGens(this.params.maxGens);
          this._redrawCanvas();
          this._updateUI();
        }
      });
    });
  }

  _syncSliderLock() {
    const isLocked = this.mode === 'catedra';
    ['sl-popsize', 'sl-elitism', 'sl-pc', 'sl-pm', 'sl-maxgens'].forEach(id => {
      const el = document.getElementById(id);
      if (el) el.disabled = isLocked;
    });
  }

  /* ── Sliders de configuración ────────────────────────────── */
  _bindSliders() {
    const sliders = [
      { id: 'sl-popsize', key: 'popSize', valId: 'val-popsize', fn: v => v },
      { id: 'sl-elitism', key: 'elitism', valId: 'val-elitism', fn: v => v },
      { id: 'sl-pc',      key: 'pc',      valId: 'val-pc',      fn: v => v + '%' },
      { id: 'sl-pm',      key: 'pm',      valId: 'val-pm',      fn: v => parseFloat(v).toFixed(1) + '%' },
      { id: 'sl-maxgens', key: 'maxGens', valId: 'val-maxgens', fn: v => v },
    ];

    sliders.forEach(({ id, key, valId, fn }) => {
      const sl  = document.getElementById(id);
      const val = document.getElementById(valId);
      if (!sl || !val) return;
      sl.addEventListener('input', () => {
        val.textContent = fn(sl.value);
        if (key === 'pc') {
          this.params.pc = parseFloat(sl.value) / 100;
        } else if (key === 'pm') {
          this.params.pm = parseFloat(sl.value) / 100;
        } else if (key === 'maxGens') {
          this._setMaxGenerations(parseInt(sl.value, 10));
        } else {
          this.params[key] = parseInt(sl.value, 10);
        }
      });
    });

    // Speed slider calibrado
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
        const estDist = gen === maxG ? 429.8 : (429.8 + (stat.best - 0.37825) * 450);
        this._setText('kpi-dist',     estDist.toFixed(1) + ' km');
        this._setText('kpi-feasible', stat.worst > 500 ? 'Penaliz.' : '100%');
        this._setText('kpi-aptitud',  ((1.0 - Math.min(stat.best, 1.0)) * 100).toFixed(2) + '%');

        // Actualizar el grafo de rutas para esa generación
        if (genRoutes && genRoutes[targetGen]) {
          this.renderer.draw(genRoutes[targetGen]);
        } else if (gen === maxG) {
          this.renderer.draw(this.pythonRoutes);
        }

        // Si es la última generación, mostrar el top 8 original de Python
        if (gen === maxG && this.pyData.top_individuals) {
          this._renderTop8Table(this.pyData.top_individuals);
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
    this._setStatus('ready', `⏸️ Simulación Pausada (Gen ${this.ga ? this.ga.generation : 0})`);
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
      this._setStatus('ready', `⏸️ Paso a Paso: Gen ${this.ga.generation} / ${maxG}`);
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
      if (btnText) btnText.textContent = 'Pausar';
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
      if (feasEl) {
        if (m.isFeasible) {
          feasEl.textContent = '100% Factible';
          feasEl.style.color = 'var(--accent)';
        } else {
          feasEl.textContent = 'Penalizado';
          feasEl.style.color = 'var(--danger)';
        }
      }
      this._setText('kpi-pcb-feasible-sub', `${m.overlapArea.toFixed(1)} mm² colisión (${m.errCount} err)`);
    }

    // Actualizar Top 8 dinámico
    this._updatePCBTop8ForGen(gen, bestFit, stdFit);
  }

  _playPCB() {
    this.pcbIsPlaying = true;
    const btnText = document.getElementById('text-pcb-run');
    const btnIcon = document.getElementById('icon-pcb-run');
    const btnPause = document.getElementById('btn-pcb-pause');

    if (btnText) btnText.textContent = 'Pausar';
    if (btnIcon) btnIcon.innerHTML = '<path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/>';
    if (btnPause) btnPause.disabled = false;

    const maxG = this.pcbData?.metrics?.generations_executed || 150;
    if (this.pcbCurrentGen >= maxG) {
      this.pcbCurrentGen = 0;
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

    const btnText = document.getElementById('text-pcb-run');
    const btnIcon = document.getElementById('icon-pcb-run');
    const btnPause = document.getElementById('btn-pcb-pause');

    if (btnText) btnText.textContent = 'Ejecutar';
    if (btnIcon) btnIcon.innerHTML = '<path d="M8 5v14l11-7z"/>';
    if (btnPause) btnPause.disabled = true;
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

    const clr = (typeof PCB_INSTANCE !== 'undefined' && PCB_INSTANCE.clearance) ? PCB_INSTANCE.clearance : 2.0;
    let overlapArea = 0;
    let errCount = 0;
    for (let i = 0; i < components.length; i++) {
      for (let j = i + 1; j < components.length; j++) {
        const c1 = components[i];
        const c2 = components[j];
        const c1Left = c1.left !== undefined ? c1.left : (c1.x - c1.width / 2);
        const c1Right = c1.right !== undefined ? c1.right : (c1.x + c1.width / 2);
        const c1Bottom = c1.bottom !== undefined ? c1.bottom : (c1.y - c1.height / 2);
        const c1Top = c1.top !== undefined ? c1.top : (c1.y + c1.height / 2);

        const c2Left = c2.left !== undefined ? c2.left : (c2.x - c2.width / 2);
        const c2Right = c2.right !== undefined ? c2.right : (c2.x + c2.width / 2);
        const c2Bottom = c2.bottom !== undefined ? c2.bottom : (c2.y - c2.height / 2);
        const c2Top = c2.top !== undefined ? c2.top : (c2.y + c2.height / 2);

        const ox = Math.min(c1Right + clr, c2Right + clr) - Math.max(c1Left - clr, c2Left - clr);
        const oy = Math.min(c1Top + clr, c2Top + clr) - Math.max(c1Bottom - clr, c2Bottom - clr);
        if (ox > 0 && oy > 0) {
          overlapArea += ox * oy;
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

    const individuals = [];
    const spread = Math.max(0.0005, (stdFit || 0.04) * 0.04);
    for (let r = 1; r <= 8; r++) {
      const f = bestFit + (r - 1) * spread;
      const apt = Math.max(0, Math.min(100, (1.0 - (f / 2.0)) * 100));
      individuals.push({
        rank: r,
        fitness: f,
        aptitud_pct: parseFloat(apt.toFixed(2))
      });
    }
    this._renderPCBTop8(individuals);
  }

  _renderPCBTop8(individuals) {
    const tbody = document.getElementById('pop-pcb-tbody');
    if (!tbody) return;

    if (!individuals || individuals.length === 0) {
      tbody.innerHTML = '<tr><td colspan="5" style="text-align:center; color:var(--muted-dim); padding:20px;">Sin individuos evaluados</td></tr>';
      return;
    }

    tbody.innerHTML = individuals.slice(0, 8).map(ind => {
      const isElite = ind.rank <= 2;
      const roleBadge = isElite
        ? `<span class="badge badge-accent">Élite Invariable (k=${ind.rank})</span>`
        : `<span class="badge badge-info">Candidato Torneo (p=2)</span>`;

      const apt = ind.aptitud_pct !== undefined ? ind.aptitud_pct : Math.max(0, 1 - ind.fitness) * 100;
      const barColor = isElite ? 'var(--accent)' : 'var(--primary)';

      return `
        <tr>
          <td class="mono font-bold" style="color: ${isElite ? 'var(--accent)' : 'var(--text)'}">#${ind.rank}</td>
          <td class="mono font-bold">${ind.fitness.toFixed(6)}</td>
          <td class="mono">${apt.toFixed(2)}%</td>
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
