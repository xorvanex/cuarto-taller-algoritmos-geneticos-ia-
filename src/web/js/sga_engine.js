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

