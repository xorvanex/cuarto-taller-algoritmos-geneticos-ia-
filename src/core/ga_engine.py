"""
Motor de Algoritmo Genético Simple (SGA) Universal.
===================================================

Implementación orientada a objetos de un Algoritmo Genético Canónico con
operadores estocásticos parametrizables, basada estrictamente en los algoritmos
y lineamientos teóricos de la cátedra de Inteligencia Artificial (Universidad de Cartagena).

Operadores implementados:
  - Elitismo estricto (Elitismo.m)
  - Selección por torneo determinístico asimétrico p=2 (seleccion_torneo_2.m)
  - Cruce en un solo punto SPX (Cruce.m)
  - Mutación puntual con reemplazo de alelo (Mutacion.m)
  - Parada por umbral de fitness < 10^-2 o máximo de generaciones

Autor: Dago David Palmera Navarro & Julián David Camargo Padilla
Cátedra: Inteligencia Artificial - Docente: Ing. Manuel Alejandro Ospina Alarcón
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np


@dataclass(frozen=True)
class GAParams:
    """
    Hiperparámetros de configuración del Algoritmo Genético.

    Atributos:
        pop_size (int): Tamaño de la población (N = 1000 según cátedra).
        elitism_count (int): Cantidad de individuos élite que pasan directos (2).
        crossover_rate (float): Probabilidad de cruce Pc (0.90 = 90%).
        mutation_rate (float): Probabilidad de mutación Pm (0.02 = 2%).
        tournament_size (int): Tamaño de participantes en torneo p (2).
        max_generations (int): Límite superior de ciclos generacionales.
        fitness_tolerance (float): Criterio de parada (fitness < 10^-2 = 0.01).
        random_seed (Optional[int]): Semilla pseudoaleatoria para reproducibilidad.
    """
    pop_size: int = 1000
    elitism_count: int = 2
    crossover_rate: float = 0.90
    mutation_rate: float = 0.02
    tournament_size: int = 2
    max_generations: int = 150
    fitness_tolerance: float = 0.01
    random_seed: Optional[int] = None


@dataclass
class GenerationStats:
    """Estadísticas descriptivas de una generación evaluada."""
    generation: int
    best_fitness: float
    avg_fitness: float
    worst_fitness: float
    std_fitness: float
    best_chromosome: np.ndarray


@dataclass
class GAResult:
    """Contenedor de resultados finales tras la ejecución del AG."""
    best_chromosome: np.ndarray
    best_fitness: float
    generations_executed: int
    converged_by_tolerance: bool
    initial_population: np.ndarray
    initial_fitness: np.ndarray
    final_population: np.ndarray
    final_fitness: np.ndarray
    history: List[GenerationStats] = field(default_factory=list)


class SimpleGeneticAlgorithm:
    """
    Motor universal del Algoritmo Genético Simple (SGA).
    
    Diseñado para problemas de minimización donde un menor fitness denota
    mayor adaptación y calidad de la solución candidata.
    """

    def __init__(self, params: Optional[GAParams] = None) -> None:
        """
        Inicializa el motor del AG con los parámetros de la cátedra.

        Args:
            params: Objeto GAParams con la parametrización oficial.
        """
        self.params = params or GAParams()
        if self.params.random_seed is not None:
            np.random.seed(self.params.random_seed)

    def _initialize_population(
        self,
        chromosome_length: int,
        bounds: Tuple[float, float]
    ) -> np.ndarray:
        """
        Genera la población inicial uniforme en el espacio [min_val, max_val].

        Args:
            chromosome_length: Longitud del cromosoma (número de genes).
            bounds: Tupla (mínimo, máximo) para el rango de cada alelo.

        Returns:
            np.ndarray de dimensiones (pop_size, chromosome_length).
        """
        low, high = bounds
        return np.random.uniform(low, high, size=(self.params.pop_size, chromosome_length))

    def _elitism(
        self,
        population: np.ndarray,
        fitness: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Aplica el operador de Elitismo estricto (fiel a Elitismo.m).

        Ordena la población de forma ascendente según el fitness y extrae
        los dos mejores individuos para transferirlos intactos a la siguiente
        generación, garantizando la monotonía del óptimo histórico.

        Args:
            population: Matriz de cromosomas actual (N x L).
            fitness: Vector de aptitud correspondiente a cada individuo (N,).

        Returns:
            Tupla (elites, poblacion_ordenada, fitness_ordenado).
        """
        sorted_indices = np.argsort(fitness)
        sorted_population = population[sorted_indices]
        sorted_fitness = fitness[sorted_indices]

        # Extraer los dos mejores individuos
        elites = copy.deepcopy(sorted_population[:self.params.elitism_count])
        return elites, sorted_population, sorted_fitness

    def _selection_tournament_2(
        self,
        population: np.ndarray,
        fitness: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Selección por torneo determinístico asimétrico p=2 (fiel a seleccion_torneo_2.m).

        - Padre 1: Seleccionado mediante torneo entre 2 individuos al azar
                   (gana el de menor fitness en minimización).
        - Padre 2: Seleccionado uniformemente al azar de la población para
                   preservar la diversidad genética y evitar convergencia prematura.

        Args:
            population: Matriz poblacional actual.
            fitness: Vector de fitness de los individuos.

        Returns:
            Tupla con (Padre1, Padre2).
        """
        n = len(population)

        # Torneo determinístico para Padre 1 (p = 2)
        idx_cand1 = np.random.randint(0, n)
        idx_cand2 = np.random.randint(0, n)

        if fitness[idx_cand1] <= fitness[idx_cand2]:
            parent1 = population[idx_cand1].copy()
        else:
            parent1 = population[idx_cand2].copy()

        # Selección estocástica uniforme para Padre 2 (asimetría de cátedra)
        idx_rand = np.random.randint(0, n)
        parent2 = population[idx_rand].copy()

        return parent1, parent2

    def _crossover_spx(
        self,
        parent1: np.ndarray,
        parent2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cruce en un solo punto SPX (Single-Point Crossover) (fiel a Cruce.m).

        Si un número aleatorio uniforme r <= Pc (0.90), se elige un punto de
        corte k aleatorio y se intercambian los segmentos genotípicos.
        De lo contrario, se aplica el operador de copia.

        Args:
            parent1: Vector cromosómico del primer progenitor.
            parent2: Vector cromosómico del segundo progenitor.

        Returns:
            Tupla con los dos descendientes (hijo1, hijo2).
        """
        length = len(parent1)
        if np.random.rand() <= self.params.crossover_rate and length > 1:
            # Seleccionar punto de corte k en el rango [1, length - 1]
            point = np.random.randint(1, length)
            child1 = np.concatenate([parent1[:point], parent2[point:]])
            child2 = np.concatenate([parent2[:point], parent1[point:]])
        else:
            # Operador de copia directa
            child1 = parent1.copy()
            child2 = parent2.copy()

        return child1, child2

    def _mutation_point(
        self,
        child1: np.ndarray,
        child2: np.ndarray,
        bounds: Tuple[float, float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Mutación puntual estocástica (fiel a Mutacion.m).

        Si un número aleatorio uniforme r <= Pm (0.02), se selecciona al azar
        uno de los dos hijos, se escoge aleatoriamente un gen k, y su valor es
        reemplazado por un nuevo alelo continuo dentro de bounds.

        Args:
            child1: Primer descendiente tras cruce.
            child2: Segundo descendiente tras cruce.
            bounds: Rango de variación admisible para el alelo mutado.

        Returns:
            Tupla con los dos descendientes (potencialmente mutados).
        """
        m1 = child1.copy()
        m2 = child2.copy()

        if np.random.rand() <= self.params.mutation_rate:
            low, high = bounds
            length = len(child1)
            target_child = np.random.choice([1, 2])
            gene_idx = np.random.randint(0, length)
            new_allele = np.random.uniform(low, high)

            if target_child == 1:
                m1[gene_idx] = new_allele
            else:
                m2[gene_idx] = new_allele

        return m1, m2

    def run(
        self,
        fitness_function: Callable[[np.ndarray], float],
        chromosome_length: int,
        bounds: Tuple[float, float] = (0.0, 100.0),
        progress_callback: Optional[Callable[[GenerationStats], None]] = None
    ) -> GAResult:
        """
        Ejecuta el ciclo generacional completo del Algoritmo Genético Simple.

        Args:
            fitness_function: Función objetivo que recibe un cromosoma (1D)
                              y retorna su valor escalar de fitness (minimización).
            chromosome_length: Número de genes por individuo.
            bounds: Rango (min, max) de los alelos del cromosoma.
            progress_callback: Función opcional para reportar métricas por generación.

        Returns:
            Objeto GAResult con la población inicial, final, estadísticas y mejor solución.
        """
        # Fase 1: Inicialización
        pop = self._initialize_population(chromosome_length, bounds)
        fitness = np.array([fitness_function(ind) for ind in pop])

        initial_pop = pop.copy()
        initial_fit = fitness.copy()

        history: List[GenerationStats] = []
        converged = False

        # Registrar estadística de Generación 0
        best_idx_0 = np.argmin(fitness)
        stats_0 = GenerationStats(
            generation=0,
            best_fitness=float(fitness[best_idx_0]),
            avg_fitness=float(np.mean(fitness)),
            worst_fitness=float(np.max(fitness)),
            std_fitness=float(np.std(fitness)),
            best_chromosome=pop[best_idx_0].copy()
        )
        history.append(stats_0)
        if progress_callback:
            progress_callback(stats_0)

        current_gen = 0

        # Fase 2: Ciclo Evolutivo
        while current_gen < self.params.max_generations:
            current_gen += 1

            # 1. Elitismo: Preservar los 2 mejores
            elites, sorted_pop, sorted_fit = self._elitism(pop, fitness)

            # Si el mejor individuo satisface la tolerancia de parada, terminar
            if sorted_fit[0] < self.params.fitness_tolerance:
                converged = True
                stats_final = GenerationStats(
                    generation=current_gen,
                    best_fitness=float(sorted_fit[0]),
                    avg_fitness=float(np.mean(sorted_fit)),
                    worst_fitness=float(np.max(sorted_fit)),
                    std_fitness=float(np.std(sorted_fit)),
                    best_chromosome=sorted_pop[0].copy()
                )
                history.append(stats_final)
                if progress_callback:
                    progress_callback(stats_final)
                break

            # 2. Generar el resto de la nueva población por reproducción
            new_population_list: List[np.ndarray] = [elites[0], elites[1]]
            needed = self.params.pop_size - self.params.elitism_count

            # Generar pares de hijos
            for _ in range(needed // 2):
                p1, p2 = self._selection_tournament_2(pop, fitness)
                c1, c2 = self._crossover_spx(p1, p2)
                m1, m2 = self._mutation_point(c1, c2, bounds)
                new_population_list.append(m1)
                new_population_list.append(m2)

            # Ajuste de tamaño por si needed fuese impar
            while len(new_population_list) < self.params.pop_size:
                p1, p2 = self._selection_tournament_2(pop, fitness)
                c1, _ = self._crossover_spx(p1, p2)
                new_population_list.append(c1)

            pop = np.array(new_population_list)
            fitness = np.array([fitness_function(ind) for ind in pop])

            # Registro de métricas
            best_idx = np.argmin(fitness)
            stats = GenerationStats(
                generation=current_gen,
                best_fitness=float(fitness[best_idx]),
                avg_fitness=float(np.mean(fitness)),
                worst_fitness=float(np.max(fitness)),
                std_fitness=float(np.std(fitness)),
                best_chromosome=pop[best_idx].copy()
            )
            history.append(stats)
            if progress_callback:
                progress_callback(stats)

        # Extracción del mejor individuo global
        final_best_idx = np.argmin(fitness)
        return GAResult(
            best_chromosome=pop[final_best_idx].copy(),
            best_fitness=float(fitness[final_best_idx]),
            generations_executed=current_gen,
            converged_by_tolerance=converged,
            initial_population=initial_pop,
            initial_fitness=initial_fit,
            final_population=pop,
            final_fitness=fitness,
            history=history
        )
