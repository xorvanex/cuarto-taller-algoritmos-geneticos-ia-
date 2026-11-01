"""
Orquestador Principal y CLI Unificado - Algoritmos Genéticos (SGA Cátedra)
==========================================================================
Cuarto Taller de Inteligencia Artificial - Semestre VII
Universidad de Cartagena - Facultad de Ingeniería

Problemas Modelados:
  1. Optimización de Rutas de Entrega con Ventanas de Tiempo (VRP)
  2. Distribución y Ruteo de Componentes en Placas de Circuito Impreso (PCB)
  3. Selección y Optimización de Cartera de Inversiones (Markowitz)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Callable, Optional
import numpy as np

# Configurar rutas del proyecto
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

for p in (PROJECT_ROOT, SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.core.ga_engine import SimpleGeneticAlgorithm, GAParams, GenerationStats, GAResult
from src.core.exporter import (
    export_vrp_results,
    export_pcb_results,
    export_markowitz_results,
)
from src.problems.vrp import (
    VRPModel,
    VRPInstance,
    create_default_vrp_instance,
    plot_vrp_solution,
    plot_vrp_convergence,
)
from src.problems.pcb import (
    PCBModel,
    PCBInstance,
    create_default_pcb_instance,
    plot_pcb_convergence,
)
from src.problems.markowitz import (
    MarkowitzModel,
    PortfolioInstance,
    create_default_portfolio_instance,
    plot_markowitz_convergence,
)


def print_banner() -> None:
    """Muestra el banner institucional del taller académico."""
    print("=" * 86)
    print("  UNIVERSIDAD DE CARTAGENA - FACULTAD DE INGENIERIA DE SISTEMAS")
    print("  TRABAJO ACADEMICO: ALGORITMOS GENETICOS CANONICOS (SGA)")
    print("  Semestre VII - Inteligencia Artificial")
    print("=" * 86)


def get_ga_params(
    custom_params: bool = False,
    pop_size: int = 1000,
    elitism: int = 2,
    crossover: float = 0.90,
    mutation: float = 0.02,
    max_gens: int = 150,
    tolerance: float = 0.01,
    seed: int = 42,
) -> GAParams:
    """Configura los hiperparámetros del SGA (Cátedra o Personalizado)."""
    if custom_params:
        print("\n--- CONFIGURACION PERSONALIZADA ---")
        try:
            val = input(f"Tamano de poblacion N [{pop_size}]: ").strip()
            if val: pop_size = int(val)
            val = input(f"Probabilidad de cruce Pc [{crossover}]: ").strip()
            if val: crossover = float(val)
            val = input(f"Probabilidad de mutacion Pm [{mutation}]: ").strip()
            if val: mutation = float(val)
            val = input(f"Maximo de generaciones [{max_gens}]: ").strip()
            if val: max_gens = int(val)
        except ValueError as err:
            print(f"[!] Entrada invalida ({err}). Usando valores por defecto de catedra.")

    params = GAParams(
        pop_size=pop_size,
        elitism_count=elitism,
        crossover_rate=crossover,
        mutation_rate=mutation,
        tournament_size=2,
        max_generations=max_gens,
        fitness_tolerance=tolerance,
        random_seed=seed,
    )

    print("\n[CONFIGURACION ACTIVA]")
    print(f"  * Poblacion (N):              {params.pop_size} individuos")
    print(f"  * Elitismo:                   {params.elitism_count} mejores conservados intactos")
    print(f"  * Seleccion:                  Torneo deterministico p={params.tournament_size} (Padre 1) + Aleatorio (Padre 2)")
    print(f"  * Cruce:                      Un solo punto (SPX) con Pc = {params.crossover_rate * 100:.1f}%")
    print(f"  * Mutacion:                   Puntual con Pm = {params.mutation_rate * 100:.1f}%")
    print(f"  * Criterio de Parada:         Fitness normalizado < {params.fitness_tolerance:.4f} (10^-2) o {params.max_generations} generaciones")
    return params


def make_progress_callback(tolerance: float) -> Callable[[GenerationStats], None]:
    """Crea el hook de consola para monitorear la convergencia cada 5 épocas."""
    print("\n[INICIANDO CICLO GENERACIONAL]")
    print("-" * 86)
    print(f"{'Gen':<6} | {'Mejor Fitness':<15} | {'Promedio':<15} | {'Peor':<15} | {'Desv. Est.':<12} | {'Estado'}")
    print("-" * 86)

    def on_progress(stats: GenerationStats) -> None:
        if stats.generation == 0 or stats.generation % 5 == 0 or stats.best_fitness < tolerance:
            tag = "PARADA (10^-2)" if stats.best_fitness < tolerance else "En evolucion"
            print(f"{stats.generation:<6} | {stats.best_fitness:<15.5f} | {stats.avg_fitness:<15.5f} | "
                  f"{stats.worst_fitness:<15.5f} | {stats.std_fitness:<12.5f} | {tag}")

    return on_progress


def print_population_tables(result: GAResult, decimal_places: int = 1) -> None:
    """Imprime las tablas representativas de la población inicial (Gen 0) y final."""
    fmt = f"{{:.{decimal_places}f}}"

    print("\n" + "=" * 86)
    print("  TABLA DE POBLACION INICIAL (Muestra primeros 5 individuos - Gen 0)")
    print("=" * 86)
    print(f"{'Individuo':<10} | {'Genes de Cromosoma (Muestra primeros 5)':<42} | {'Fitness Normalizado'}")
    print("-" * 86)
    for i in range(min(5, len(result.initial_population))):
        sample_genes = ", ".join([fmt.format(g) for g in result.initial_population[i][:5]]) + "..."
        print(f"Ind. {i+1:<5} | [{sample_genes:<38}] | {result.initial_fitness[i]:.5f}")
    print(f"\nEstadisticas Iniciales -> Min: {np.min(result.initial_fitness):.4f} | Prom: {np.mean(result.initial_fitness):.4f} | Max: {np.max(result.initial_fitness):.4f} | Desv: {np.std(result.initial_fitness):.4f}")

    print("\n" + "=" * 86)
    print(f"  TABLA DE POBLACION FINAL (Top 5 individuos - Generacion {result.generations_executed})")
    print("=" * 86)
    sorted_final_indices = np.argsort(result.final_fitness)
    print(f"{'Rango':<10} | {'Genes de Cromosoma (Muestra primeros 5)':<42} | {'Fitness Normalizado'}")
    print("-" * 86)
    for rank, idx in enumerate(sorted_final_indices[:5]):
        sample_genes = ", ".join([fmt.format(g) for g in result.final_population[idx][:5]]) + "..."
        print(f"Top {rank+1:<6} | [{sample_genes:<38}] | {result.final_fitness[idx]:.5f}")
    print(f"\nEstadisticas Finales -> Mejor: {result.best_fitness:.5f} | Prom: {np.mean(result.final_fitness):.5f} | Max: {np.max(result.final_fitness):.5f} | Desv: {np.std(result.final_fitness):.5f}")
    print("=" * 86)


# ==============================================================================
# PROBLEMA 1: OPTIMIZACIÓN DE RUTAS DE ENTREGA (VRP)
# ==============================================================================
def run_problem_1_vrp(custom_params: bool = False) -> None:
    """Ejecuta la optimización del Problema 1: Ruteo de Vehículos (VRP)."""
    print("\n" + "#" * 86)
    print("  EJECUTANDO PROBLEMA 1: OPTIMIZACION DE RUTAS DE ENTREGA (VRP)")
    print("#" * 86)

    params = get_ga_params(custom_params)
    instance = create_default_vrp_instance()
    model = VRPModel(instance)

    print(f"\n[ESCENARIO LOGISTICO VRP]")
    print(f"  * Deposito Central:           ({instance.depot[0]:.0f}, {instance.depot[1]:.0f}) km")
    print(f"  * Cantidad de Clientes:       {len(instance.customers)} nodos de entrega")
    print(f"  * Demanda Total de Carga:     {sum(c.demand for c in instance.customers):.1f} kg")
    print(f"  * Flota de Camiones:          {instance.num_vehicles} vehiculos homogeneos")
    print(f"  * Capacidad Maxima:           {instance.vehicle_capacity:.1f} kg/camion")
    print(f"  * Jornada Maxima:             {instance.max_shift_time:.1f} minutos (8 horas)")

    t_start = time.time()
    result = SimpleGeneticAlgorithm(params).run(
        fitness_function=model.evaluate_fitness,
        chromosome_length=model.num_customers,
        bounds=(0.0, 100.0),
        progress_callback=make_progress_callback(params.fitness_tolerance),
    )
    t_elapsed = time.time() - t_start
    best_solution = model.decode_chromosome(result.best_chromosome)

    print("-" * 86)
    print(f"\n[OPTIMIZACION CONCLUIDA EN {t_elapsed:.2f} SEGUNDOS]")
    print(f"  * Generaciones ejecutadas:    {result.generations_executed}")
    print(f"  * Criterio alcanzado:         {'SI (Fitness < 10^-2)' if result.converged_by_tolerance else 'Limite de generaciones alcanzado'}")
    print(f"  * Mejor Fitness normalizado:  {result.best_fitness:.6f}")
    print(f"  * Distancia Total Acumulada:  {best_solution.total_distance_km:.2f} km")
    print(f"  * Exceso de Capacidad Total:  {best_solution.total_capacity_violation:.2f} kg")
    print(f"  * Exceso de Tiempo de Turno:  {best_solution.total_time_violation:.2f} min")
    print(f"  * Factibilidad de la Flota:   {'100% FACTIBLE (Cero violaciones)' if best_solution.is_feasible else 'NO FACTIBLE (Presenta sobrecargas)'}")

    print("\n[DETALLE DE RUTAS ASIGNADAS A CADA CAMION]")
    for r in best_solution.routes:
        stops_str = " -> ".join([f"C{cid}" for cid in r.stops]) if r.stops else "Sin paradas"
        print(f"  * Camion {r.truck_id}: Depot -> {stops_str} -> Depot | Dist: {r.distance_km:.1f} km | Carga: {r.load_kg:.1f} kg | Tiempo: {r.total_time_min:.1f} min")

    data_dir = os.path.join(PROJECT_ROOT, "data")
    os.makedirs(data_dir, exist_ok=True)
    fig1_path = os.path.join(data_dir, "vrp_solution.png")
    fig2_path = os.path.join(data_dir, "vrp_convergence.png")
    plot_vrp_solution(best_solution, instance, fig1_path, dpi=300)
    plot_vrp_convergence(result.history, params.fitness_tolerance, fig2_path, dpi=300)

    print(f"\n[GRAFICOS GENERADOS]")
    print(f"  [OK] Mapa de Rutas:           {fig1_path}")
    print(f"  [OK] Curvas de Convergencia:  {fig2_path}")

    export_vrp_results(result, best_solution, instance, params, t_elapsed, PROJECT_ROOT)
    print_population_tables(result, decimal_places=1)


# ==============================================================================
# PROBLEMA 2: DISTRIBUCIÓN Y RUTEO DE COMPONENTES PCB
# ==============================================================================
def run_problem_2_pcb(custom_params: bool = False) -> None:
    """Ejecuta la optimización del Problema 2: Layout de Circuitos PCB."""
    print("\n" + "#" * 86)
    print("  EJECUTANDO PROBLEMA 2: DISENO DE CIRCUITOS IMPRESOS (PCB LAYOUT)")
    print("#" * 86)

    params = get_ga_params(custom_params)
    instance = create_default_pcb_instance()
    model = PCBModel(instance)

    print(f"\n[ESCENARIO TECNOLOGICO PCB]")
    print(f"  * Dimensiones del Sustrato:   {instance.board_max_width:.1f} x {instance.board_max_height:.1f} mm")
    print(f"  * Componentes a Ubicar:       {len(instance.components)} modulos electronicos")
    print(f"  * Redes de Conexion (Nets):   {len(instance.nets)} pistas de cobre")
    print(f"  * Aislamiento Minimo:         {instance.clearance:.1f} mm")

    t_start = time.time()
    result = SimpleGeneticAlgorithm(params).run(
        fitness_function=model.evaluate_fitness,
        chromosome_length=model.chromosome_length,
        bounds=(0.0, 100.0),
        progress_callback=make_progress_callback(params.fitness_tolerance),
    )
    t_elapsed = time.time() - t_start
    best_sol = model.decode_chromosome(result.best_chromosome)

    print("\n" + "=" * 86)
    print("  RESULTADOS DE LA OPTIMIZACION DEL PCB (ALGORITMO GENETICO)")
    print("=" * 86)
    print(f"Tiempo de computo:             {t_elapsed:.2f} segundos")
    print(f"Generaciones ejecutadas:       {result.generations_executed} / {params.max_generations}")
    print(f"Convergencia por umbral:       {'SI' if result.converged_by_tolerance else 'NO (Limite alcanzado)'}")
    print(f"Mejor Fitness normalizado:     {result.best_fitness:.6f}")
    print(f"Aptitud Biologica:             {max(0.0, 1.0 - result.best_fitness) * 100:.2f}%")
    print(f"Longitud Pistas Manhattan:     {best_sol.manhattan_wirelength_mm:.1f} mm")
    print(f"Longitud Pistas Euclidea:      {best_sol.total_wirelength_mm:.1f} mm")
    print(f"Area Envolvente Bounding Box:  {best_sol.bounding_box_area_mm2:.1f} mm2 ({best_sol.bounding_box_width:.1f} x {best_sol.bounding_box_height:.1f} mm)")
    print(f"Area de Solapamiento:          {best_sol.total_overlap_area_mm2:.1f} mm2 (Colisiones: {best_sol.overlap_count})")
    print(f"Factibilidad del Diseno:       {'100% FACTIBLE (Cero solapamientos)' if best_sol.is_feasible else 'NO FACTIBLE (Presenta solapamiento)'}")

    data_dir = os.path.join(PROJECT_ROOT, "data")
    os.makedirs(data_dir, exist_ok=True)
    pcb_plot_path = os.path.join(data_dir, "pcb_solution.png")
    conv_plot_path = os.path.join(data_dir, "pcb_convergence.png")
    model.plot_solution(best_sol, pcb_plot_path, dpi=300)
    plot_pcb_convergence(result.history, params.fitness_tolerance, conv_plot_path, dpi=300)

    print(f"\n[GRAFICOS]")
    print(f"  [OK] Plano tecnico del PCB:        {pcb_plot_path}")
    print(f"  [OK] Curva de convergencia:        {conv_plot_path}")

    export_pcb_results(result, best_sol, instance, params, t_elapsed, PROJECT_ROOT)
    print_population_tables(result, decimal_places=2)


# ==============================================================================
# PROBLEMA 3: SELECCIÓN Y OPTIMIZACIÓN DE CARTERA DE MARKOWITZ
# ==============================================================================
def run_problem_3_markowitz(custom_params: bool = False) -> None:
    """Ejecuta la optimización del Problema 3: Cartera de Inversiones de Markowitz."""
    print("\n" + "#" * 86)
    print("  EJECUTANDO PROBLEMA 3: CARTERA DE INVERSIONES (TEORIA DE MARKOWITZ)")
    print("#" * 86)

    params = get_ga_params(custom_params)
    instance = create_default_portfolio_instance()
    model = MarkowitzModel(instance)

    print(f"\n[ESCENARIO FINANCIERO MARKOWITZ]")
    print(f"  * Universo de Activos:        {len(instance.assets)} instrumentos diversificados")
    print(f"  * Tasa Libre de Riesgo (Rf):  {instance.risk_free_rate * 100:.2f}% anual")
    print(f"  * Exposicion Maxima / Activo: {instance.max_asset_exposure * 100:.1f}%")
    print(f"  * Ponderaciones Multiobjetivo: Riesgo={instance.weight_risk:.2f} | Retorno={instance.weight_return:.2f}")

    t_start = time.time()
    result = SimpleGeneticAlgorithm(params).run(
        fitness_function=model.evaluate_fitness,
        chromosome_length=model.num_assets,
        bounds=(0.0, 100.0),
        progress_callback=make_progress_callback(params.fitness_tolerance),
    )
    t_elapsed = time.time() - t_start
    best_sol = model.decode_chromosome(result.best_chromosome)

    print("\n" + "=" * 86)
    print("  RESULTADOS DE LA OPTIMIZACION DE CARTERA (ALGORITMO GENETICO)")
    print("=" * 86)
    print(f"Tiempo de computo:             {t_elapsed:.2f} segundos")
    print(f"Generaciones ejecutadas:       {result.generations_executed} / {params.max_generations}")
    print(f"Convergencia por umbral:       {'SI' if result.converged_by_tolerance else 'NO (Limite alcanzado)'}")
    print(f"Mejor Fitness normalizado:     {result.best_fitness:.6f}")
    print(f"Aptitud Biologica:             {max(0.0, 1.0 - result.best_fitness) * 100:.2f}%")
    print(f"Retorno Esperado E[Rp]:        {best_sol.expected_return * 100:.2f}% anual")
    print(f"Riesgo / Volatilidad sigma_p:  {best_sol.volatility * 100:.2f}% anual")
    print(f"Ratio de Sharpe:               {best_sol.sharpe_ratio:.4f}")
    print(f"Suma de Pesos Presupuestal:    {best_sol.budget_sum * 100:.2f}% (Error: {best_sol.budget_error:.5f})")
    print(f"Factibilidad Financiera:       {'100% FACTIBLE' if best_sol.is_feasible else 'NO FACTIBLE (Violacion de restricciones)'}")

    print("\n[DISTRIBUCION DE CAPITAL OPTIMA]")
    for a in best_sol.asset_allocations:
        print(f"  * {a['ticker']:<6} ({a['name']:<25}): {a['percentage']:>6.2f}%")

    data_dir = os.path.join(PROJECT_ROOT, "data")
    os.makedirs(data_dir, exist_ok=True)
    frontier_path = os.path.join(data_dir, "markowitz_efficient_frontier.png")
    alloc_path = os.path.join(data_dir, "markowitz_allocation.png")
    conv_path = os.path.join(data_dir, "markowitz_convergence.png")

    model.plot_efficient_frontier(best_sol, frontier_path, n_samples=2500, dpi=300)
    model.plot_asset_allocation(best_sol, alloc_path, dpi=300)
    plot_markowitz_convergence(result.history, params.fitness_tolerance, conv_path, dpi=300)

    print(f"\n[GRAFICOS]")
    print(f"  [OK] Frontera Eficiente:           {frontier_path}")
    print(f"  [OK] Distribucion de Cartera:      {alloc_path}")
    print(f"  [OK] Curva de Convergencia:        {conv_path}")

    export_markowitz_results(result, best_sol, instance, params, t_elapsed, PROJECT_ROOT)
    print_population_tables(result, decimal_places=3)


# ==============================================================================
# DESPACHADOR PRINCIPAL (CLI Y MENÚ INTERACTIVO)
# ==============================================================================
def main() -> None:
    """Funcion de despacho principal con soporte para flags CLI y menu interactivo."""
    parser = argparse.ArgumentParser(description="Optimizador de Algoritmos Geneticos - IA")
    parser.add_argument("--problem", type=int, choices=[1, 2, 3], help="Problema a ejecutar (1: VRP, 2: PCB, 3: Markowitz)")
    parser.add_argument("--custom", action="store_true", help="Permite configurar parametros en consola")
    args = parser.parse_args()

    print_banner()

    if args.problem:
        if args.problem == 1:
            run_problem_1_vrp(custom_params=args.custom)
        elif args.problem == 2:
            run_problem_2_pcb(custom_params=args.custom)
        elif args.problem == 3:
            run_problem_3_markowitz(custom_params=args.custom)
        return

    while True:
        print("\nMENU PRINCIPAL:")
        print("  [1] Ejecutar Problema 1: Optimizacion de Rutas de Entrega (VRP) - Modo Catedra")
        print("  [2] Ejecutar Problema 1: VRP - Modo Parametros Personalizados")
        print("  [3] Ejecutar Problema 2: Diseno de Circuitos PCB (Modo Catedra)")
        print("  [4] Ejecutar Problema 2: PCB - Modo Parametros Personalizados")
        print("  [5] Ejecutar Problema 3: Cartera de Inversiones Markowitz (Modo Catedra)")
        print("  [6] Ejecutar Problema 3: Markowitz - Modo Parametros Personalizados")
        print("  [0] Salir")

        try:
            choice = input("\nSeleccione una opcion: ").strip()
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
            run_problem_3_markowitz(custom_params=False)
        elif choice == "6":
            run_problem_3_markowitz(custom_params=True)
        elif choice == "0":
            print("\nPrograma finalizado exitosamente.")
            break
        else:
            print("[!] Opcion no reconocida.")


if __name__ == "__main__":
    main()
