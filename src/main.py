"""
Punto de Entrada Principal e Interfaz Unificada del Proyecto.
============================================================

Trabajo Final: Algoritmos Genéticos (AG)
Inteligencia Artificial - Semestre VII
Universidad de Cartagena - Facultad de Ingeniería - Ingeniería de Sistemas

Docente: Ing. Manuel Alejandro Ospina Alarcón
Autor: Dago David Palmera Navarro

Arquitectura Modular:
  - Python ('src/'): Ejecuta los modelos matemáticos, decodificación, operadores y estadísticas.
  - Exportación ('data/' y 'src/web/data/'): Exporta los resultados reales a JSON/JS.
  - Interfaz Web ('src/web/'): Capa pura de presentación visual en Glassmorphism Dark-Tech.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional
import numpy as np

# Configurar rutas del proyecto
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

for p in (PROJECT_ROOT, SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.core.ga_engine import SimpleGeneticAlgorithm, GAParams, GenerationStats, GAResult
from src.problems.pcb import (
    PCBModel,
    PCBInstance,
    PCBSolution,
    create_default_pcb_instance,
    plot_pcb_convergence,
)
from src.problems.vrp import (
    VRPModel,
    VRPInstance,
    VRPSolution,
    create_default_vrp_instance,
    plot_vrp_solution,
    plot_vrp_convergence,
)


def print_banner() -> None:
    """Imprime el banner institucional de la cátedra."""
    print("=" * 86)
    print("   UNIVERSIDAD DE CARTAGENA - FACULTAD DE INGENIERÍA")
    print("   PROGRAMA DE INGENIERÍA DE SISTEMAS - INTELIGENCIA ARTIFICIAL (CORTE 3)")
    print("   CUARTO TALLER: ALGORITMOS GENÉTICOS - SUITE DE OPTIMIZACIÓN MULTI-PROBLEMA")
    print("   Autor: Dago David Palmera Navarro")
    print("=" * 86)


def export_vrp_results(
    result: GAResult,
    solution: VRPSolution,
    instance: VRPInstance,
    params: GAParams,
    t_elapsed: float,
) -> None:
    """
    Exporta los resultados reales del algoritmo genético VRP a JSON y JS
    para que la interfaz web ('src/web/') y el pipeline de datos ('data/')
    los consuman como única fuente de verdad computacional.
    """
    data = {
        "problem": "VRP",
        "title": "Problema 1: Optimización de Rutas de Entrega (VRP)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "execution_time_seconds": round(t_elapsed, 4),
        "params": {
            "pop_size": params.pop_size,
            "elitism_count": params.elitism_count,
            "crossover_rate": params.crossover_rate,
            "mutation_rate": params.mutation_rate,
            "tournament_size": params.tournament_size,
            "max_generations": params.max_generations,
            "fitness_tolerance": params.fitness_tolerance,
            "random_seed": params.random_seed,
        },
        "instance": {
            "depot": {"x": instance.depot[0], "y": instance.depot[1]},
            "num_vehicles": instance.num_vehicles,
            "vehicle_capacity": instance.vehicle_capacity,
            "max_shift_time": instance.max_shift_time,
            "avg_speed": instance.avg_speed,
            "customers": [
                {
                    "id": f"C{c.id}",
                    "name": c.name,
                    "x": c.x,
                    "y": c.y,
                    "demand": c.demand,
                    "service": c.service_time,
                }
                for c in instance.customers
            ],
        },
        "metrics": {
            "generations_executed": result.generations_executed,
            "converged_by_tolerance": bool(result.converged_by_tolerance),
            "best_fitness": round(float(result.best_fitness), 6),
            "best_aptitud": round(float(max(0.0, 1.0 - result.best_fitness)), 6),
            "total_distance_km": round(float(solution.total_distance_km), 2),
            "capacity_violation_kg": round(float(solution.total_capacity_violation), 2),
            "time_violation_min": round(float(solution.total_time_violation), 2),
            "is_feasible": bool(solution.is_feasible),
        },
        "history": [
            {
                "gen": s.generation,
                "best": round(float(s.best_fitness), 6),
                "avg": round(float(s.avg_fitness), 6),
                "worst": round(float(s.worst_fitness), 6),
                "std": round(float(s.std_fitness), 6),
            }
            for s in result.history
        ],
        "routes": [
            {
                "truck_id": r.truck_id,
                "stops": [f"C{cid}" for cid in r.stops],
                "stop_indices": [cid - 1 for cid in r.stops],
                "distance_km": round(float(r.distance_km), 2),
                "load_kg": round(float(r.load_kg), 2),
                "total_time_min": round(float(r.total_time_min), 2),
                "capacity_violation": round(float(r.capacity_violation), 2),
                "time_violation": round(float(r.time_violation), 2),
            }
            for r in solution.routes
        ],
        "top_individuals": [
            {
                "rank": rank + 1,
                "fitness": round(float(result.final_fitness[idx]), 6),
                "aptitud": round(float(max(0.0, 1.0 - result.final_fitness[idx])), 6),
                "chromosome_sample": [round(float(g), 2) for g in result.final_population[idx][:6]],
            }
            for rank, idx in enumerate(np.argsort(result.final_fitness)[:8])
        ],
        "generation_routes": [
            [
                [cid - 1 for cid in r.stops]
                for r in VRPModel(instance).decode_chromosome(s.best_chromosome).routes
            ]
            for s in result.history
        ],
    }

    # 1. Guardar en data/vrp_results.json
    global_dir = os.path.join(PROJECT_ROOT, "data")
    os.makedirs(global_dir, exist_ok=True)
    global_json = os.path.join(global_dir, "vrp_results.json")
    with open(global_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # 2. Guardar en src/web/data/vrp_results.json
    web_dir = os.path.join(SRC_DIR, "web", "data")
    os.makedirs(web_dir, exist_ok=True)
    web_json = os.path.join(web_dir, "vrp_results.json")
    with open(web_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # 3. Guardar en src/web/data/vrp_results.js (garantiza ejecución autónoma con doble clic)
    web_js = os.path.join(web_dir, "vrp_results.js")
    with open(web_js, "w", encoding="utf-8") as f:
        f.write("/** Datos de simulación generados automáticamente por Python (src/main.py) */\n")
        f.write("window.VRP_SIMULATION_DATA = ")
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write(";\n")

    print(f"\n[EXPORTACIÓN MODULAR DE DATOS A LA WEB]")
    print(f"  [OK] Dataset global:               {global_json}")
    print(f"  [OK] Dataset web JSON:             {web_json}")
    print(f"  [OK] Dataset web JS (autónomo):    {web_js}")


def run_problem_1_vrp(custom_params: bool = False) -> None:
    """
    Ejecuta la optimización del Problema 1: Ruteo de Vehículos (VRP).

    Args:
        custom_params: Si es True, permite al usuario ingresar parámetros manuales.
    """
    print("\n" + "#" * 86)
    print("  EJECUTANDO PROBLEMA 1: OPTIMIZACIÓN DE RUTAS DE ENTREGA (VRP)")
    print("#" * 86)

    # 1. Configuración de hiperparámetros de cátedra
    pop_size = 1000
    elitism = 2
    crossover = 0.90
    mutation = 0.02
    max_gens = 150
    tolerance = 0.01

    if custom_params:
        print("\n--- CONFIGURACIÓN PERSONALIZADA ---")
        try:
            val = input(f"Tamaño de población N [{pop_size}]: ").strip()
            if val: pop_size = int(val)
            val = input(f"Probabilidad de cruce Pc [{crossover}]: ").strip()
            if val: crossover = float(val)
            val = input(f"Probabilidad de mutación Pm [{mutation}]: ").strip()
            if val: mutation = float(val)
            val = input(f"Máximo de generaciones [{max_gens}]: ").strip()
            if val: max_gens = int(val)
        except ValueError as err:
            print(f"[!] Entrada inválida ({err}). Usando valores por defecto de cátedra.")

    params = GAParams(
        pop_size=pop_size,
        elitism_count=elitism,
        crossover_rate=crossover,
        mutation_rate=mutation,
        tournament_size=2,
        max_generations=max_gens,
        fitness_tolerance=tolerance,
        random_seed=42  # Semilla estocástica para consistencia reproducible
    )

    print(f"\n[CONFIGURACIÓN ACTIVA]")
    print(f"  * Población (N):              {params.pop_size} individuos")
    print(f"  * Elitismo:                   {params.elitism_count} mejores conservados intactos")
    print(f"  * Selección:                  Torneo determinístico p={params.tournament_size} (Padre 1) + Aleatorio (Padre 2)")
    print(f"  * Cruce:                      Un solo punto (SPX) con Pc = {params.crossover_rate * 100:.1f}%")
    print(f"  * Mutación:                   Puntual con Pm = {params.mutation_rate * 100:.1f}%")
    print(f"  * Criterio de Parada:         Fitness normalizado < {params.fitness_tolerance:.4f} (10^-2) o {params.max_generations} generaciones")

    # 2. Inicializar Modelo e Instancia
    instance = create_default_vrp_instance()
    model = VRPModel(instance)

    print(f"\n[ESCENARIO LOGÍSTICO VRP]")
    print(f"  * Depósito Central:           ({instance.depot[0]:.0f}, {instance.depot[1]:.0f}) km")
    print(f"  * Cantidad de Clientes:       {len(instance.customers)} nodos de entrega")
    print(f"  * Demanda Total de Carga:     {sum(c.demand for c in instance.customers):.1f} kg")
    print(f"  * Flota de Camiones:          {instance.num_vehicles} vehículos homogéneos")
    print(f"  * Capacidad Máxima:           {instance.vehicle_capacity:.1f} kg/camión")
    print(f"  * Jornada Máxima:             {instance.max_shift_time:.1f} minutos (8 horas)")

    # Hook para mostrar progreso cada 5 generaciones
    print("\n[INICIANDO CICLO GENERACIONAL]")
    print("-" * 86)
    print(f"{'Gen':<6} | {'Mejor Fitness':<15} | {'Promedio':<15} | {'Peor':<15} | {'Desv. Est.':<12} | {'Estado'}")
    print("-" * 86)

    def on_progress(stats: GenerationStats) -> None:
        if stats.generation == 0 or stats.generation % 5 == 0 or stats.best_fitness < params.fitness_tolerance:
            tag = "PARADA (10^-2)" if stats.best_fitness < params.fitness_tolerance else "En evolución"
            print(f"{stats.generation:<6} | {stats.best_fitness:<15.5f} | {stats.avg_fitness:<15.5f} | "
                  f"{stats.worst_fitness:<15.5f} | {stats.std_fitness:<12.5f} | {tag}")

    # 3. Ejecución del Algoritmo Genético
    t_start = time.time()
    engine = SimpleGeneticAlgorithm(params)
    result = engine.run(
        fitness_function=model.evaluate_fitness,
        chromosome_length=model.num_customers,
        bounds=(0.0, 100.0),
        progress_callback=on_progress
    )
    t_elapsed = time.time() - t_start

    # 4. Decodificación y Auditoría de la Mejor Solución
    best_solution = model.decode_chromosome(result.best_chromosome)

    print("-" * 86)
    print(f"\n[OPTIMIZACIÓN CONCLUIDA EN {t_elapsed:.2f} SEGUNDOS]")
    print(f"  * Generaciones ejecutadas:    {result.generations_executed}")
    print(f"  * Criterio alcanzado:         {'SI (Fitness < 10^-2)' if result.converged_by_tolerance else 'Límite de generaciones'}")
    print(f"  * Mejor Fitness alcanzado:    {result.best_fitness:.6f}")
    print(f"  * Distancia Total Óptima:     {best_solution.total_distance_km:.2f} km")
    print(f"  * Violación de Capacidad:     {best_solution.total_capacity_violation:.2f} kg ({'FACTIBLE' if best_solution.total_capacity_violation == 0 else 'INFACTIBLE'})")
    print(f"  * Violación de Horario:       {best_solution.total_time_violation:.2f} min ({'FACTIBLE' if best_solution.total_time_violation == 0 else 'INFACTIBLE'})")

    print("\n[DESGLOSE DE RUTAS POR CAMIÓN]")
    for r in best_solution.routes:
        stops_str = " -> ".join([f"C{cid}" for cid in r.stops])
        print(f"  * Camión {r.truck_id}: Depósito -> {stops_str} -> Depósito")
        print(f"      Distancia: {r.distance_km:.2f} km | Carga: {r.load_kg:.1f}/{instance.vehicle_capacity:.0f} kg | Tiempo: {r.total_time_min:.1f}/{instance.max_shift_time:.0f} min")

    # 5. Generación y Exportación de Figuras de Publicación (300 DPI)
    img_dir = os.path.join(PROJECT_ROOT, "docs", "img")
    os.makedirs(img_dir, exist_ok=True)
    fig1_path = os.path.join(img_dir, "figura1_vrp_rutas_optimas.png")
    fig2_path = os.path.join(img_dir, "figura2_vrp_convergencia.png")

    print(f"\n[GENERANDO GRÁFICAS EN ALTA RESOLUCIÓN (300 DPI)]")
    plot_vrp_solution(best_solution, instance, fig1_path)
    plot_vrp_convergence(result.history, params.fitness_tolerance, fig2_path)
    print(f"  [OK] Mapa de Rutas:           {fig1_path}")
    print(f"  [OK] Curvas de Convergencia:  {fig2_path}")

    # 6. Exportación de Datos para la Interfaz Web (Pipeline Desacoplado)
    export_vrp_results(result, best_solution, instance, params, t_elapsed)

    # 7. Tabulación para el Informe (Subsecciones 7.1 y 7.2)
    print("\n" + "=" * 86)
    print("  TABLA DE POBLACIÓN INICIAL (Muestra primeros 5 individuos - Gen 0)")
    print("=" * 86)
    print(f"{'Individuo':<10} | {'Genes de Cromosoma (Muestra primeros 5)':<42} | {'Fitness Normalizado'}")
    print("-" * 86)
    for i in range(5):
        sample_genes = ", ".join([f"{g:.1f}" for g in result.initial_population[i][:5]]) + "..."
        print(f"Ind. {i+1:<5} | [{sample_genes:<38}] | {result.initial_fitness[i]:.5f}")
    print(f"\nEstadísticas Iniciales -> Min: {np.min(result.initial_fitness):.4f} | Prom: {np.mean(result.initial_fitness):.4f} | Max: {np.max(result.initial_fitness):.4f} | Desv: {np.std(result.initial_fitness):.4f}")

    print("\n" + "=" * 86)
    print(f"  TABLA DE POBLACIÓN FINAL (Top 5 individuos - Generación {result.generations_executed})")
    print("=" * 86)
    sorted_final_indices = np.argsort(result.final_fitness)
    print(f"{'Rango':<10} | {'Genes de Cromosoma (Muestra primeros 5)':<42} | {'Fitness Normalizado'}")
    print("-" * 86)
    for rank, idx in enumerate(sorted_final_indices[:5]):
        sample_genes = ", ".join([f"{g:.1f}" for g in result.final_population[idx][:5]]) + "..."
        print(f"Top {rank+1:<6} | [{sample_genes:<38}] | {result.final_fitness[idx]:.5f}")
    print(f"\nEstadísticas Finales -> Mejor: {result.best_fitness:.5f} | Prom: {np.mean(result.final_fitness):.5f} | Max: {np.max(result.final_fitness):.5f} | Desv: {np.std(result.final_fitness):.5f}")
    print("=" * 86)


def export_pcb_results(
    result: GAResult,
    solution: PCBSolution,
    instance: PCBInstance,
    params: GAParams,
    t_elapsed: float,
) -> None:
    """
    Exporta los resultados reales del algoritmo genético PCB a JSON y JS
    para que la interfaz web ('src/web/') y el pipeline de datos ('data/')
    los consuman como única fuente de verdad computacional.
    """
    model = PCBModel(instance)
    generation_layouts = []
    for s in result.history:
        sol = model.decode_chromosome(s.best_chromosome)
        generation_layouts.append([
            {
                "id": c.id,
                "ref": c.ref,
                "name": c.name,
                "x": round(c.x, 2),
                "y": round(c.y, 2),
                "width": c.width,
                "height": c.height,
                "category": c.category,
                "color": c.color,
                "left": round(c.left, 2),
                "right": round(c.right, 2),
                "bottom": round(c.bottom, 2),
                "top": round(c.top, 2),
            }
            for c in sol.placed_components
        ])

    data = {
        "problem": "PCB",
        "title": "Problema 2: Optimización del Diseño de Circuitos Electrónicos (PCB Layout)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "execution_time_seconds": round(t_elapsed, 4),
        "params": {
            "pop_size": params.pop_size,
            "elitism_count": params.elitism_count,
            "crossover_rate": params.crossover_rate,
            "mutation_rate": params.mutation_rate,
            "tournament_size": params.tournament_size,
            "max_generations": params.max_generations,
            "fitness_tolerance": params.fitness_tolerance,
            "random_seed": params.random_seed,
        },
        "instance": {
            "board_max_width": instance.board_max_width,
            "board_max_height": instance.board_max_height,
            "clearance": instance.clearance,
            "components": [
                {
                    "id": c.id,
                    "ref": c.ref,
                    "name": c.name,
                    "width": c.width,
                    "height": c.height,
                    "category": c.category,
                    "color": c.color,
                }
                for c in instance.components
            ],
            "nets": [
                {
                    "id": n.id,
                    "net_name": n.net_name,
                    "source_id": n.source_id,
                    "target_id": n.target_id,
                    "weight": n.weight,
                }
                for n in instance.nets
            ],
        },
        "metrics": {
            "generations_executed": result.generations_executed,
            "converged_by_tolerance": bool(result.converged_by_tolerance),
            "best_fitness": round(float(result.best_fitness), 6),
            "best_aptitud": round(float(max(0.0, 1.0 - result.best_fitness)), 6),
            "manhattan_wirelength_mm": round(float(solution.manhattan_wirelength_mm), 2),
            "euclidean_wirelength_mm": round(float(solution.total_wirelength_mm), 2),
            "bounding_box_width_mm": round(float(solution.bounding_box_width), 2),
            "bounding_box_height_mm": round(float(solution.bounding_box_height), 2),
            "bounding_box_area_mm2": round(float(solution.bounding_box_area_mm2), 2),
            "overlap_area_mm2": round(float(solution.total_overlap_area_mm2), 2),
            "overlap_count": int(solution.overlap_count),
            "is_feasible": bool(solution.is_feasible),
        },
        "placed_components": [
            {
                "id": c.id,
                "ref": c.ref,
                "name": c.name,
                "x": round(c.x, 2),
                "y": round(c.y, 2),
                "width": c.width,
                "height": c.height,
                "category": c.category,
                "color": c.color,
                "left": round(c.left, 2),
                "right": round(c.right, 2),
                "bottom": round(c.bottom, 2),
                "top": round(c.top, 2),
            }
            for c in solution.placed_components
        ],
        "history": [
            {
                "gen": s.generation,
                "best": round(float(s.best_fitness), 6),
                "avg": round(float(s.avg_fitness), 6),
                "worst": round(float(s.worst_fitness), 6),
                "std": round(float(s.std_fitness), 6),
            }
            for s in result.history
        ],
        "top_individuals": [
            {
                "rank": rank + 1,
                "fitness": round(float(result.final_fitness[idx]), 6),
                "aptitud_pct": round(float(max(0.0, 1.0 - result.final_fitness[idx]) * 100), 2),
                "chromosome_sample": [round(float(g), 2) for g in result.final_population[idx][:6]],
            }
            for rank, idx in enumerate(np.argsort(result.final_fitness)[:8])
        ],
        "generation_layouts": generation_layouts,
    }

    # 1. Guardar en data/pcb_results.json
    global_dir = os.path.join(PROJECT_ROOT, "data")
    os.makedirs(global_dir, exist_ok=True)
    global_json = os.path.join(global_dir, "pcb_results.json")
    with open(global_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # 2. Guardar en src/web/data/pcb_results.json
    web_dir = os.path.join(SRC_DIR, "web", "data")
    os.makedirs(web_dir, exist_ok=True)
    web_json = os.path.join(web_dir, "pcb_results.json")
    with open(web_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # 3. Guardar en src/web/data/pcb_results.js (garantiza ejecución autónoma)
    web_js = os.path.join(web_dir, "pcb_results.js")
    with open(web_js, "w", encoding="utf-8") as f:
        f.write("/** Datos certificados de simulación PCB generados por Python (src/main.py) */\n")
        f.write("window.PCB_SIMULATION_DATA = ")
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write(";\n")

    print(f"\n[EXPORTACIÓN MODULAR DE DATOS PCB A LA WEB]")
    print(f"  [OK] Dataset global:               {global_json}")
    print(f"  [OK] Dataset web JSON:             {web_json}")
    print(f"  [OK] Dataset web JS (autónomo):    {web_js}")


def run_problem_2_pcb(custom_params: bool = False) -> None:
    """
    Ejecuta la optimización del Problema 2: Diseño de Circuitos Electrónicos (PCB Layout).

    Args:
        custom_params: Si es True, permite al usuario ingresar parámetros manuales.
    """
    print("\n" + "#" * 86)
    print("  EJECUTANDO PROBLEMA 2: DISEÑO DE CIRCUITOS ELECTRÓNICOS (PCB LAYOUT)")
    print("#" * 86)

    # 1. Configuración de hiperparámetros de cátedra
    pop_size = 1000
    elitism = 2
    crossover = 0.90
    mutation = 0.02
    max_gens = 150
    tolerance = 0.01

    if custom_params:
        print("\n--- CONFIGURACIÓN PERSONALIZADA ---")
        try:
            val = input(f"Tamaño de población N [{pop_size}]: ").strip()
            if val: pop_size = int(val)
            val = input(f"Probabilidad de cruce Pc [{crossover}]: ").strip()
            if val: crossover = float(val)
            val = input(f"Probabilidad de mutación Pm [{mutation}]: ").strip()
            if val: mutation = float(val)
            val = input(f"Máximo de generaciones [{max_gens}]: ").strip()
            if val: max_gens = int(val)
        except ValueError as err:
            print(f"[!] Entrada inválida ({err}). Usando valores por defecto de cátedra.")

    params = GAParams(
        pop_size=pop_size,
        elitism_count=elitism,
        crossover_rate=crossover,
        mutation_rate=mutation,
        tournament_size=2,
        max_generations=max_gens,
        fitness_tolerance=tolerance,
        random_seed=123
    )

    print(f"\n[CONFIGURACIÓN ACTIVA]")
    print(f"  * Población (N):              {params.pop_size} individuos")
    print(f"  * Elitismo:                   {params.elitism_count} mejores conservados intactos")
    print(f"  * Selección:                  Torneo determinístico p={params.tournament_size} (Padre 1) + Aleatorio (Padre 2)")
    print(f"  * Cruce:                      Un solo punto (SPX) con Pc = {params.crossover_rate * 100:.1f}%")
    print(f"  * Mutación:                   Puntual con Pm = {params.mutation_rate * 100:.1f}%")
    print(f"  * Criterio de Parada:         Fitness normalizado < {params.fitness_tolerance:.4f} (10^-2) o {params.max_generations} generaciones")

    # 2. Inicializar Modelo e Instancia
    instance = create_default_pcb_instance()
    model = PCBModel(instance)

    print(f"\n[ESCENARIO PCB LAYOUT]")
    print(f"  * Área Máxima de Sustrato:    {instance.board_max_width:.0f} x {instance.board_max_height:.0f} mm ({instance.board_max_width * instance.board_max_height:.0f} mm²)")
    print(f"  * Componentes Electrónicos:   {len(instance.components)} chips y componentes pasivos")
    print(f"  * Red de Conexiones (Nets):   {len(instance.nets)} trazas de cobre lógicas")
    print(f"  * Holgura de Seguridad:       {instance.clearance:.1f} mm de aislamiento mínimo")

    print("\n[INICIANDO CICLO GENERACIONAL]")
    print("-" * 86)
    print(f"{'Gen':<6} | {'Mejor Fitness':<15} | {'Promedio':<15} | {'Peor':<15} | {'Desv. Est.':<12} | {'Estado'}")
    print("-" * 86)

    def on_progress(stats: GenerationStats) -> None:
        if stats.generation == 0 or stats.generation % 5 == 0 or stats.best_fitness < params.fitness_tolerance:
            tag = "PARADA (10^-2)" if stats.best_fitness < params.fitness_tolerance else "En evolución"
            print(f"{stats.generation:<6} | {stats.best_fitness:<15.5f} | {stats.avg_fitness:<15.5f} | "
                  f"{stats.worst_fitness:<15.5f} | {stats.std_fitness:<12.5f} | {tag}")

    # 3. Ejecución del Algoritmo Genético
    t_start = time.time()
    engine = SimpleGeneticAlgorithm(params)
    result = engine.run(
        fitness_function=model.evaluate_fitness,
        chromosome_length=model.num_genes,
        bounds=(0.0, 1.0),
        progress_callback=on_progress
    )
    t_elapsed = time.time() - t_start

    # 4. Decodificación de la Mejor Solución
    best_sol = model.decode_chromosome(result.best_chromosome)

    print("\n" + "=" * 86)
    print("  RESULTADOS DE LA OPTIMIZACIÓN DEL PCB (ALGORITMO GENÉTICO)")
    print("=" * 86)
    print(f"Tiempo de cómputo:             {t_elapsed:.2f} segundos")
    print(f"Generaciones ejecutadas:       {result.generations_executed} / {params.max_generations}")
    print(f"Convergencia por umbral:       {'SÍ' if result.converged_by_tolerance else 'NO (Límite de épocas alcanzado)'}")
    print(f"Mejor Fitness normalizado:     {result.best_fitness:.6f}")
    print(f"Aptitud Biológica:             {max(0.0, 1.0 - result.best_fitness) * 100:.2f}%")
    print(f"Longitud de Pistas Manhattan:  {best_sol.manhattan_wirelength_mm:.1f} mm")
    print(f"Longitud de Pistas Euclídea:   {best_sol.total_wirelength_mm:.1f} mm")
    print(f"Área Envolvente Bounding Box:  {best_sol.bounding_box_area_mm2:.1f} mm² ({best_sol.bounding_box_width:.1f} x {best_sol.bounding_box_height:.1f} mm)")
    print(f"Área de Solapamiento:          {best_sol.total_overlap_area_mm2:.1f} mm² (Colisiones: {best_sol.overlap_count})")
    print(f"Factibilidad del Diseño:       {'100% FACTIBLE (Cero solapamientos)' if best_sol.is_feasible else 'NO FACTIBLE (Presenta solapamiento)'}")

    # 5. Generación de Artefactos Gráficos
    data_dir = os.path.join(PROJECT_ROOT, "data")
    os.makedirs(data_dir, exist_ok=True)

    pcb_plot_path = os.path.join(data_dir, "pcb_solution.png")
    model.plot_solution(best_sol, pcb_plot_path, dpi=300)

    conv_plot_path = os.path.join(data_dir, "pcb_convergence.png")
    plot_pcb_convergence(result.history, params.fitness_tolerance, conv_plot_path, dpi=300)

    print(f"\n[GRÁFICOS]")
    print(f"  [OK] Plano técnico del PCB:        {pcb_plot_path}")
    print(f"  [OK] Curva de convergencia:        {conv_plot_path}")

    # 6. Exportación de Datos
    export_pcb_results(result, best_sol, instance, params, t_elapsed)


def main() -> None:
    """Función de despacho principal."""
    parser = argparse.ArgumentParser(description="Optimizador de Algoritmos Genéticos - IA")
    parser.add_argument("--problem", type=int, choices=[1, 2, 3], help="Problema a ejecutar (1: VRP, 2: PCB, 3: Markowitz)")
    parser.add_argument("--custom", action="store_true", help="Permite configurar parámetros en consola")
    args = parser.parse_args()

    print_banner()

    if args.problem:
        if args.problem == 1:
            run_problem_1_vrp(custom_params=args.custom)
        elif args.problem == 2:
            run_problem_2_pcb(custom_params=args.custom)
        elif args.problem == 3:
            print("\n[!] Problema 3 (Markowitz) en fase de ensamblaje.")
        return

    # Menú interactivo si se ejecuta sin flags
    while True:
        print("\nMENÚ PRINCIPAL:")
        print("  [1] Ejecutar Problema 1: Optimización de Rutas de Entrega (VRP) - Modo Cátedra")
        print("  [2] Ejecutar Problema 1: VRP - Modo Parámetros Personalizados")
        print("  [3] Ejecutar Problema 2: Diseño de Circuitos PCB (Modo Cátedra)")
        print("  [4] Ejecutar Problema 2: PCB - Modo Parámetros Personalizados")
        print("  [5] Ejecutar Problema 3: Cartera de Inversiones Markowitz (Próximamente)")
        print("  [0] Salir")

        try:
            choice = input("\nSeleccione una opción: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nSaliendo...")
            break

        if choice == "1":
            run_problem_1_vrp(custom_params=False)
        elif choice == "2":
            run_problem_1_vrp(custom_params=True)
        elif choice == "3":
            run_problem_2_pcb(custom_params=False)
        elif choice == "4":
            run_problem_2_pcb(custom_params=True)
        elif choice == "5":
            print("\n[+] El módulo Markowitz se cargará a continuación.")
        elif choice == "0":
            print("\nPrograma finalizado exitosamente.")
            break
        else:
            print("[!] Opción no reconocida.")


if __name__ == "__main__":
    main()
