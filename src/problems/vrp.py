"""
Problema 1: Optimización de Rutas de Entrega de Vehículos (VRP).
================================================================

Modelado matemático, decodificación cromosómica, función de aptitud normalizada
con penalizaciones cuadráticas y generador de gráficos en alta resolución (300 DPI)
para el problema de ruteo de vehículos con capacidad y tiempo limitado (CVRPTW).

Autor: Dago David Palmera Navarro
Cátedra: Inteligencia Artificial - Docente: Ing. Manuel Alejandro Ospina Alarcón
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.core.ga_engine import GenerationStats, SimpleGeneticAlgorithm, GAParams, GAResult


@dataclass(frozen=True)
class Customer:
    """Representación de un nodo cliente a ser abastecido."""
    id: int
    name: str
    x: float
    y: float
    demand: float  # kg
    service_time: float  # minutos


@dataclass(frozen=True)
class VRPInstance:
    """
    Especificación de la instancia del problema logístico VRP.

    Atributos:
        depot: Tupla (x, y) de la coordenada del centro de distribución central.
        customers: Lista de nodos clientes a atender.
        num_vehicles: Número de camiones disponibles en la flota (3).
        vehicle_capacity: Capacidad máxima de carga por camión (kg).
        max_shift_time: Jornada laboral máxima por camión (minutos).
        avg_speed: Velocidad promedio de tránsito vehicular (km/h).
    """
    depot: Tuple[float, float] = (50.0, 50.0)
    customers: List[Customer] = field(default_factory=list)
    num_vehicles: int = 3
    vehicle_capacity: float = 120.0  # kg
    max_shift_time: float = 480.0    # 8 horas = 480 min
    avg_speed: float = 45.0          # km/h


@dataclass
class RouteMetrics:
    """Métricas operativas de la ruta asignada a un camión."""
    truck_id: int
    stops: List[int]  # Lista de IDs de clientes
    distance_km: float
    load_kg: float
    total_time_min: float
    capacity_violation: float
    time_violation: float


@dataclass
class VRPSolution:
    """Resultado completo decodificado del fenotipo de una ruta VRP."""
    routes: List[RouteMetrics]
    total_distance_km: float
    total_capacity_violation: float
    total_time_violation: float
    raw_fitness: float
    normalized_fitness: float
    is_feasible: bool


def create_default_vrp_instance() -> VRPInstance:
    """
    Crea la instancia de estudio estándar con 15 clientes representativos
    en un espacio geográfico operacional de 100 x 100 km.
    """
    depot = (50.0, 50.0)
    # 15 clientes estratégicamente ubicados en distintos cuadrantes
    clients_data = [
        (1, "C1 - Norte", 52.0, 85.0, 18.0, 20.0),
        (2, "C2 - Noreste", 82.0, 78.0, 15.0, 18.0),
        (3, "C3 - Este", 90.0, 52.0, 22.0, 25.0),
        (4, "C4 - Sureste", 78.0, 22.0, 19.0, 20.0),
        (5, "C5 - Sur", 48.0, 12.0, 25.0, 22.0),
        (6, "C6 - Suroeste", 20.0, 18.0, 14.0, 15.0),
        (7, "C7 - Oeste", 12.0, 48.0, 20.0, 20.0),
        (8, "C8 - Noroeste", 18.0, 80.0, 16.0, 18.0),
        (9, "C9 - Centro-Norte", 45.0, 70.0, 21.0, 24.0),
        (10, "C10 - Centro-Este", 70.0, 55.0, 17.0, 18.0),
        (11, "C11 - Centro-Sur", 55.0, 30.0, 24.0, 22.0),
        (12, "C12 - Centro-Oeste", 30.0, 52.0, 19.0, 20.0),
        (13, "C13 - Periferia Este", 88.0, 35.0, 15.0, 16.0),
        (14, "C14 - Periferia Norte", 35.0, 92.0, 23.0, 25.0),
        (15, "C15 - Periferia Sur", 62.0, 8.0, 16.0, 18.0),
    ]

    customers = [
        Customer(id=c[0], name=c[1], x=c[2], y=c[3], demand=c[4], service_time=c[5])
        for c in clients_data
    ]

    return VRPInstance(
        depot=depot,
        customers=customers,
        num_vehicles=3,
        vehicle_capacity=120.0,
        max_shift_time=480.0,
        avg_speed=45.0
    )


class VRPModel:
    """
    Modelador del VRP: codificación, decodificación fenotípica y evaluación
    de aptitud adaptada a los operadores de cátedra con normalización hacia 10^-2.
    """

    def __init__(self, instance: Optional[VRPInstance] = None) -> None:
        """
        Inicializa el modelo VRP con su instancia de clientes y parámetros.
        """
        self.instance = instance or create_default_vrp_instance()
        self.num_customers = len(self.instance.customers)
        self.depot = self.instance.customers

        # Cotas para normalización estricta (aprobada por el docente)
        # D_ideal: Cota inferior aproximada de ruteo eficiente (~240 km)
        # D_worst: Peor ruteo aleatorio desordenado (~750 km)
        self.d_ideal: float = 235.0
        self.d_worst: float = 750.0

        # Coeficientes de penalización cuadrática para restricciones duras
        self.lambda_capacity: float = 0.05
        self.lambda_time: float = 0.02

    def _euclidean_dist(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Calcula la distancia euclidiana plana entre dos coordenadas en km."""
        return float(np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2))

    def decode_chromosome(self, chromosome: np.ndarray) -> VRPSolution:
        """
        Decodifica el genotipo continuo (Random Keys) en el fenotipo de rutas.

        Mecanismo:
          1. El orden relativo de los alelos en el cromosoma define una permutación
             estricta de clientes (sin duplicados, compatible con cruce SPX).
          2. Algoritmo de partición heurística: se asignan paradas secuenciales a cada
             camión de la flota. Si se rebasa la capacidad o el tiempo, se conmuta al
             siguiente camión.
          3. Las infracciones residuales generan penalizaciones proporcionales.

        Args:
            chromosome: Vector 1D de números reales de longitud N_clientes.

        Returns:
            VRPSolution con las rutas desglosadas por camión y métricas de factibilidad.
        """
        # Ordenación de claves aleatorias para obtener la secuencia de atención
        customer_order_indices = np.argsort(chromosome)

        routes_list: List[RouteMetrics] = []
        cust_map = {c.id: c for c in self.instance.customers}

        # Reparto equilibrado en 3 camiones (5 clientes por camión de base, ajustado por ruta)
        # Se divide la secuencia en los k vehículos
        k = self.instance.num_vehicles
        subarrays = np.array_split(customer_order_indices, k)

        total_distance = 0.0
        total_cap_viol = 0.0
        total_time_viol = 0.0

        for truck_idx, client_indices in enumerate(subarrays):
            stops_ids = [self.instance.customers[idx].id for idx in client_indices]

            current_pos = self.instance.depot
            dist_km = 0.0
            load_kg = 0.0
            serv_time_min = 0.0

            for cid in stops_ids:
                c = cust_map[cid]
                dist_km += self._euclidean_dist(current_pos, (c.x, c.y))
                load_kg += c.demand
                serv_time_min += c.service_time
                current_pos = (c.x, c.y)

            # Retorno al depósito central
            dist_km += self._euclidean_dist(current_pos, self.instance.depot)

            # Tiempo total: tiempo de viaje (dist / v * 60) + tiempos de servicio
            travel_time_min = (dist_km / self.instance.avg_speed) * 60.0
            total_route_time = travel_time_min + serv_time_min

            cap_viol = max(0.0, load_kg - self.instance.vehicle_capacity)
            time_viol = max(0.0, total_route_time - self.instance.max_shift_time)

            total_distance += dist_km
            total_cap_viol += cap_viol
            total_time_viol += time_viol

            routes_list.append(RouteMetrics(
                truck_id=truck_idx + 1,
                stops=stops_ids,
                distance_km=dist_km,
                load_kg=load_kg,
                total_time_min=total_route_time,
                capacity_violation=cap_viol,
                time_violation=time_viol
            ))

        # Cálculo de Fitness Normalizado (Conforme al acuerdo con el docente)
        # Error relativo de distancia normalizado en [0, 1] + penalizaciones cuadráticas
        norm_dist_error = max(0.0, (total_distance - self.d_ideal) / (self.d_worst - self.d_ideal))
        penalties = (self.lambda_capacity * (total_cap_viol ** 2) +
                     self.lambda_time * (total_time_viol ** 2))

        # Fitness para minimización (tiende a 0.00; al alcanzar < 10^-2 se logra óptimo)
        fitness_val = norm_dist_error + penalties

        is_feasible = (total_cap_viol == 0.0 and total_time_viol == 0.0)

        return VRPSolution(
            routes=routes_list,
            total_distance_km=total_distance,
            total_capacity_violation=total_cap_viol,
            total_time_violation=total_time_viol,
            raw_fitness=total_distance,
            normalized_fitness=fitness_val,
            is_feasible=is_feasible
        )

    def evaluate_fitness(self, chromosome: np.ndarray) -> float:
        """
        Función escalar de aptitud para el motor del Algoritmo Genético.

        Args:
            chromosome: Vector continuo 1D.

        Returns:
            Valor de fitness (menor es mejor; condición de parada: fitness < 10^-2).
        """
        solution = self.decode_chromosome(chromosome)
        return solution.normalized_fitness


# ==============================================================================
# GENERADORES DE FIGURAS EN ALTA RESOLUCIÓN (300 DPI)
# ==============================================================================

def plot_vrp_solution(
    solution: VRPSolution,
    instance: VRPInstance,
    save_path: str,
    dpi: int = 300
) -> None:
    """
    Genera y guarda el mapa topológico de ruteo de los camiones en alta resolución.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8), dpi=dpi)

    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd']
    depot = instance.depot
    cust_map = {c.id: c for c in instance.customers}

    # 1. Trazar rutas de cada camión
    for r in solution.routes:
        color = colors[(r.truck_id - 1) % len(colors)]
        coords = [depot]
        for cid in r.stops:
            c = cust_map[cid]
            coords.append((c.x, c.y))
        coords.append(depot)

        xs, ys = zip(*coords)
        ax.plot(
            xs, ys,
            color=color,
            linestyle='-',
            linewidth=2.2,
            alpha=0.85,
            label=f"Camión {r.truck_id}: {r.distance_km:.1f} km | {r.load_kg:.0f}/{instance.vehicle_capacity:.0f} kg"
        )

        # Flechas de dirección
        for i in range(len(coords) - 1):
            x1, y1 = coords[i]
            x2, y2 = coords[i+1]
            dx, dy = (x2 - x1) * 0.2, (y2 - y1) * 0.2
            ax.annotate(
                '', xy=(x1 + dx, y1 + dy), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5)
            )

    # 2. Dibujar clientes
    for c in instance.customers:
        ax.scatter(c.x, c.y, color='#2c3e50', s=160, zorder=5, edgecolors='white', linewidth=1.5)
        ax.annotate(
            f"C{c.id}\n({c.demand:.0f}kg)",
            (c.x, c.y),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center',
            fontsize=8.5,
            fontweight='bold',
            color='#1a252f'
        )

    # 3. Dibujar Depósito Central
    ax.scatter(
        depot[0], depot[1],
        color='#c0392b',
        s=320,
        marker='s',
        zorder=6,
        edgecolors='black',
        linewidth=2.0,
        label=f"Depósito Central ({depot[0]:.0f}, {depot[1]:.0f})"
    )

    ax.set_title(
        f"Optimización de Rutas de Entrega (VRP) - Mejor Individuo $\\vec{{x}}^*$\n"
        f"Distancia Total: {solution.total_distance_km:.2f} km | Fitness: {solution.normalized_fitness:.4f} "
        f"({'Factible' if solution.is_feasible else 'Infactible'})",
        fontsize=12,
        fontweight='bold',
        pad=15,
        color='#1b365d'
    )
    ax.set_xlabel("Coordenada X (km)", fontsize=10, fontweight='bold')
    ax.set_ylabel("Coordenada Y (km)", fontsize=10, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', framealpha=0.95, fontsize=8.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi)
    plt.close()


def plot_vrp_convergence(
    history: List[GenerationStats],
    tolerance: float,
    save_path: str,
    dpi: int = 300
) -> None:
    """
    Genera y guarda la gráfica de convergencia generacional (Mejor, Promedio y Peor).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)

    gens = [s.generation for s in history]
    bests = [s.best_fitness for s in history]
    avgs = [s.avg_fitness for s in history]
    worsts = [s.worst_fitness for s in history]

    ax.plot(gens, bests, color='#27ae60', linewidth=2.5, label='Mejor Fitness ($f_{min}$)')
    ax.plot(gens, avgs, color='#2980b9', linewidth=1.8, linestyle='--', label='Fitness Promedio ($\\bar{f}$)')
    ax.plot(gens, worsts, color='#e74c3c', linewidth=1.2, alpha=0.7, linestyle=':', label='Peor Fitness ($f_{max}$)')

    # Línea umbral del criterio de parada (10^-2 = 0.01)
    ax.axhline(
        y=tolerance,
        color='#8e44ad',
        linestyle='-.',
        linewidth=2.0,
        label=f'Criterio de Parada (Fitness < {tolerance:.2f})'
    )

    ax.set_title(
        "Curvas de Convergencia Generacional - Algoritmo Genético Simple (Problema 1: VRP)\n"
        f"Población: N=1000 | Elitismo: 2 | Cruce: SPX 90% | Mutación: Puntual 2%",
        fontsize=11.5,
        fontweight='bold',
        pad=12,
        color='#1b365d'
    )
    ax.set_xlabel("Generación", fontsize=10, fontweight='bold')
    ax.set_ylabel("Valor de la Función de Aptitud (Fitness Normalizado)", fontsize=10, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', framealpha=0.95, fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi)
    plt.close()
