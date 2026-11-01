"""
Módulo de exportación desacoplada de datasets analíticos (JSON y JS).
====================================================================
Garantiza el desacoplamiento entre el backend de simulación Python
y la interfaz gráfica web Glassmorphism / Dark-Tech.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from src.core.ga_engine import GAResult, GAParams
    from src.problems.vrp import VRPSolution, VRPInstance
    from src.problems.pcb import PCBSolution, PCBInstance
    from src.problems.markowitz import PortfolioSolution, PortfolioInstance

DEFAULT_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)


def export_dataset(
    data: Dict[str, Any],
    var_name: str,
    file_stem: str,
    project_root: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Exporta simultáneamente el dataset en dos destinos:
      1. data/{file_stem}.json : Archivo estándar para análisis offline y persistencia.
      2. src/web/data/{file_stem}.js : Inyección en window.{var_name} para la web sin CORS.
    """
    root = project_root or DEFAULT_PROJECT_ROOT
    data_dir = os.path.join(root, "data")
    web_data_dir = os.path.join(root, "src", "web", "data")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(web_data_dir, exist_ok=True)

    json_path = os.path.join(data_dir, f"{file_stem}.json")
    js_path = os.path.join(web_data_dir, f"{file_stem}.js")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    js_content = (
        f"/** Dataset analitico oficial - Problema: {file_stem} */\n"
        f"window.{var_name} = {json.dumps(data, ensure_ascii=False)};\n"
    )
    with open(js_path, "w", encoding="utf-8") as f:
        f.write(js_content)

    print(f"\n[EXPORTACION DE RESULTADOS: {file_stem.upper()}]")
    print(f"  [OK] Dataset JSON: {json_path}")
    print(f"  [OK] Dataset JS:   {js_path}")

    return json_path, js_path


def export_vrp_results(
    result: GAResult,
    solution: VRPSolution,
    instance: VRPInstance,
    params: GAParams,
    t_elapsed: float,
    project_root: Optional[str] = None,
) -> None:
    """
    Exporta los resultados reales del algoritmo genético VRP a JSON y JS
    para que la interfaz web ('src/web/') y el pipeline de datos ('data/')
    los consuman como única fuente de verdad computacional.
    """
    from src.problems.vrp import VRPModel

    root = project_root or DEFAULT_PROJECT_ROOT
    vrp_model = VRPModel(instance)

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
                for r in vrp_model.decode_chromosome(s.best_chromosome).routes
            ]
            for s in result.history
        ],
    }

    export_dataset(data, "VRP_SIMULATION_DATA", "vrp_results", root)


def export_pcb_results(
    result: GAResult,
    solution: PCBSolution,
    instance: PCBInstance,
    params: GAParams,
    t_elapsed: float,
    project_root: Optional[str] = None,
) -> None:
    """
    Exporta los resultados reales del algoritmo genético PCB a JSON y JS
    para que la interfaz web ('src/web/') y el pipeline de datos ('data/')
    los consuman como única fuente de verdad computacional.
    """
    from src.problems.pcb import PCBModel

    root = project_root or DEFAULT_PROJECT_ROOT
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
                "wirelength_mm": round(float(model.decode_chromosome(result.final_population[idx]).manhattan_wirelength_mm), 2),
                "area_mm2": round(float(model.decode_chromosome(result.final_population[idx]).bounding_box_area_mm2), 2),
                "overlap_mm2": round(float(model.decode_chromosome(result.final_population[idx]).total_overlap_area_mm2), 2),
                "overlap_count": int(model.decode_chromosome(result.final_population[idx]).overlap_count),
                "is_feasible": bool(model.decode_chromosome(result.final_population[idx]).is_feasible),
                "chromosome_sample": [round(float(g), 2) for g in result.final_population[idx][:6]],
            }
            for rank, idx in enumerate(np.argsort(result.final_fitness)[:8])
        ],
        "generation_layouts": generation_layouts,
    }

    export_dataset(data, "PCB_SIMULATION_DATA", "pcb_results", root)


def export_markowitz_results(
    result: GAResult,
    solution: PortfolioSolution,
    instance: PortfolioInstance,
    params: GAParams,
    t_elapsed: float,
    project_root: Optional[str] = None,
) -> None:
    """
    Exporta los resultados reales del algoritmo genético de Markowitz a JSON y JS
    para que la interfaz web ('src/web/') y el pipeline de datos ('data/')
    los consuman como única fuente de verdad computacional.
    """
    from src.problems.markowitz import MarkowitzModel

    root = project_root or DEFAULT_PROJECT_ROOT
    model = MarkowitzModel(instance)

    generation_portfolios = []
    for s in result.history:
        sol = model.decode_chromosome(s.best_chromosome)
        generation_portfolios.append({
            "gen": s.generation,
            "expected_return": round(float(sol.expected_return * 100), 2),
            "volatility": round(float(sol.volatility * 100), 2),
            "sharpe_ratio": round(float(sol.sharpe_ratio), 4),
            "budget_sum": round(float(sol.budget_sum * 100), 2),
            "budget_error": round(float(sol.budget_error), 5),
            "is_feasible": bool(sol.is_feasible),
            "allocations": [
                {
                    "ticker": a["ticker"],
                    "pct": a["percentage"],
                    "color": a["color"]
                }
                for a in sol.asset_allocations
            ]
        })

    mc_vols, mc_rets, mc_sharpes = model.generate_monte_carlo_cloud(n_samples=1000, seed=42)
    monte_carlo_cloud = [
        {
            "vol": round(float(v * 100), 2),
            "ret": round(float(r * 100), 2),
            "sharpe": round(float(sh), 3)
        }
        for v, r, sh in zip(mc_vols, mc_rets, mc_sharpes)
    ]

    data = {
        "problem": "MARKOWITZ",
        "title": "Problema 3: Optimización de Cartera de Inversiones (Markowitz Portfolio Selection)",
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
            "risk_free_rate": instance.risk_free_rate,
            "max_asset_exposure": instance.max_asset_exposure,
            "weight_risk": instance.weight_risk,
            "weight_return": instance.weight_return,
            "assets": [
                {
                    "id": a.id,
                    "ticker": a.ticker,
                    "name": a.name,
                    "sector": a.sector,
                    "expected_return": a.expected_return,
                    "volatility": a.volatility,
                    "color": a.color,
                }
                for a in instance.assets
            ],
            "correlation_matrix": [[round(float(v), 2) for v in row] for row in instance.correlation_matrix],
        },
        "metrics": {
            "generations_executed": result.generations_executed,
            "converged_by_tolerance": bool(result.converged_by_tolerance),
            "best_fitness": round(float(result.best_fitness), 6),
            "best_aptitud": round(float(max(0.0, 1.0 - result.best_fitness)), 6),
            "expected_return_pct": round(float(solution.expected_return * 100), 2),
            "volatility_pct": round(float(solution.volatility * 100), 2),
            "variance": round(float(solution.variance), 6),
            "sharpe_ratio": round(float(solution.sharpe_ratio), 4),
            "budget_sum_pct": round(float(solution.budget_sum * 100), 2),
            "budget_error": round(float(solution.budget_error), 5),
            "max_weight_pct": round(float(solution.max_weight * 100), 2),
            "exposure_violations": int(solution.exposure_violations),
            "is_feasible": bool(solution.is_feasible),
        },
        "allocations": solution.asset_allocations,
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
                "expected_return_pct": round(float(model.decode_chromosome(result.final_population[idx]).expected_return * 100), 2),
                "volatility_pct": round(float(model.decode_chromosome(result.final_population[idx]).volatility * 100), 2),
                "sharpe_ratio": round(float(model.decode_chromosome(result.final_population[idx]).sharpe_ratio), 4),
                "budget_error": round(float(model.decode_chromosome(result.final_population[idx]).budget_error), 5),
                "is_feasible": bool(model.decode_chromosome(result.final_population[idx]).is_feasible),
                "chromosome_sample": [round(float(g), 4) for g in result.final_population[idx][:5]],
            }
            for rank, idx in enumerate(np.argsort(result.final_fitness)[:8])
        ],
        "generation_portfolios": generation_portfolios,
        "monte_carlo_cloud": monte_carlo_cloud,
    }

    export_dataset(data, "MARKOWITZ_SIMULATION_DATA", "markowitz_results", root)
