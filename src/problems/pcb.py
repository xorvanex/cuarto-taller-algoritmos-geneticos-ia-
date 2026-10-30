"""
Problema 2: Optimización del Diseño de Circuitos Electrónicos (PCB Layout).
===========================================================================

Modelado matemático, decodificación cromosómica, función de aptitud multiobjetivo
y generador de planos técnicos para el problema de emplazamiento de componentes
y minimización de longitud de trazas de cobre (Component Placement & Wirelength Minimization).

Autor: Dago David Palmera Navarro (Cód. 0222321003)
Cátedra: Inteligencia Artificial - Docente: Ing. Manuel Alejandro Ospina Alarcón
Universidad de Cartagena - Facultad de Ingeniería (Semestre VII)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from src.core.ga_engine import GenerationStats, SimpleGeneticAlgorithm, GAParams, GAResult


@dataclass(frozen=True)
class PCBComponent:
    """
    Componente electrónico a posicionar sobre el sustrato del PCB.

    Atributos:
        id (int): Identificador numérico único (0 a N-1).
        ref (str): Designador de referencia (ej. 'U1', 'C1', 'J1').
        name (str): Nombre comercial o función técnica.
        width (float): Ancho físico del encapsulado en milímetros (mm).
        height (float): Alto físico del encapsulado en milímetros (mm).
        category (str): Familia tecnológica ('ic', 'power', 'passive', 'connector', 'clock').
        color (str): Código hexadecimal de color para visualización de serigrafía.
    """
    id: int
    ref: str
    name: str
    width: float
    height: float
    category: str
    color: str


@dataclass(frozen=True)
class PCBNet:
    """
    Conexión física (pista / traza de cobre) requerida entre dos componentes.

    Atributos:
        id (int): Identificador de la red.
        net_name (str): Nombre de la señal (ej. 'SPI_SCK', 'VCC_3V3', 'XTAL_IN').
        source_id (int): ID del componente de origen.
        target_id (int): ID del componente de destino.
        weight (float): Factor de ponderación por criticidad (señales de reloj y desacoplo tienen mayor peso).
    """
    id: int
    net_name: str
    source_id: int
    target_id: int
    weight: float = 1.0


@dataclass(frozen=True)
class PCBInstance:
    """
    Especificación del circuito impreso y su conjunto de componentes y conexiones.

    Atributos:
        board_max_width (float): Límite máximo horizontal del área de diseño (mm).
        board_max_height (float): Límite máximo vertical del área de diseño (mm).
        clearance (float): Holgura mínima de seguridad entre componentes contiguos (mm).
        components (List[PCBComponent]): Catálogo de componentes electrónicos a ubicar.
        nets (List[PCBNet]): Netlist con la lista de interconexiones requeridas.
    """
    board_max_width: float = 100.0   # mm
    board_max_height: float = 80.0   # mm
    clearance: float = 2.0           # mm de aislamiento preventivo
    components: List[PCBComponent] = field(default_factory=list)
    nets: List[PCBNet] = field(default_factory=list)


@dataclass
class PlacedComponent:
    """Componente con coordenadas absolutas calculadas sobre el sustrato."""
    id: int
    ref: str
    name: str
    x: float          # Centro en X (mm)
    y: float          # Centro en Y (mm)
    width: float
    height: float
    category: str
    color: str
    left: float
    right: float
    bottom: float
    top: float


@dataclass
class PCBSolution:
    """Resultado completo decodificado de la disposición física del PCB."""
    placed_components: List[PlacedComponent]
    total_wirelength_mm: float
    manhattan_wirelength_mm: float
    bounding_box_width: float
    bounding_box_height: float
    bounding_box_area_mm2: float
    total_overlap_area_mm2: float
    raw_fitness: float
    normalized_fitness: float
    is_feasible: bool
    overlap_count: int


def create_default_pcb_instance() -> PCBInstance:
    """
    Crea la instancia canónica de evaluación para el Problema 2:
    Placa de circuito con 14 componentes electrónicos y 22 conexiones lógicas.
    """
    components = [
        # Microcontrolador central y potencia
        PCBComponent(id=0,  ref="U1",   name="MCU ARM-Cortex",    width=16.0, height=16.0, category="ic",        color="#10B981"),
        PCBComponent(id=1,  ref="U2",   name="Regulador LDO 3.3V",width=10.0, height=8.0,  category="power",     color="#F59E0B"),
        PCBComponent(id=2,  ref="Y1",   name="Cristal 16MHz",     width=8.0,  height=5.0,  category="clock",     color="#8B5CF6"),
        PCBComponent(id=3,  ref="U3",   name="Memoria SPI Flash", width=8.0,  height=8.0,  category="ic",        color="#06B6D4"),
        
        # Conectores externos
        PCBComponent(id=4,  ref="J1",   name="USB-C Power/Data",  width=10.0, height=8.0,  category="connector", color="#3B82F6"),
        PCBComponent(id=5,  ref="J2",   name="Header GPIO 2x5",   width=20.0, height=6.0,  category="connector", color="#6366F1"),
        
        # Condensadores de desacoplo y potencia
        PCBComponent(id=6,  ref="C1",   name="Cap Desacoplo 100n",width=4.0,  height=4.0,  category="passive",   color="#EC4899"),
        PCBComponent(id=7,  ref="C2",   name="Cap Desacoplo 100n",width=4.0,  height=4.0,  category="passive",   color="#EC4899"),
        PCBComponent(id=8,  ref="C3",   name="Cap Entrada 10uF",  width=6.0,  height=6.0,  category="power",     color="#F97316"),
        PCBComponent(id=9,  ref="C4",   name="Cap Salida 10uF",   width=6.0,  height=6.0,  category="power",     color="#F97316"),
        
        # Resistencias y protección
        PCBComponent(id=10, ref="R1",   name="Res Reset Pull-up", width=4.0,  height=3.0,  category="passive",   color="#A855F7"),
        PCBComponent(id=11, ref="R2",   name="Res SPI Term",      width=4.0,  height=3.0,  category="passive",   color="#A855F7"),
        PCBComponent(id=12, ref="R3",   name="Res LED Estado",    width=4.0,  height=3.0,  category="passive",   color="#A855F7"),
        PCBComponent(id=13, ref="D1",   name="Diodo ESD USB",     width=5.0,  height=4.0,  category="power",     color="#EF4444"),
    ]

    nets = [
        # Alimentación USB y entrada LDO
        PCBNet(id=1,  net_name="USB_VBUS",   source_id=4, target_id=13, weight=1.5),  # J1 -> D1
        PCBNet(id=2,  net_name="VBUS_PROT",  source_id=13, target_id=1, weight=1.5),  # D1 -> U2
        PCBNet(id=3,  net_name="LDO_IN_CAP", source_id=1, target_id=8,  weight=2.0),  # U2 -> C3
        PCBNet(id=4,  net_name="LDO_OUT_CAP",source_id=1, target_id=9,  weight=2.0),  # U2 -> C4
        PCBNet(id=5,  net_name="VCC_3V3",    source_id=9, target_id=0,  weight=2.0),  # C4 -> U1 (VCC a MCU)
        
        # Desacoplo crítico MCU
        PCBNet(id=6,  net_name="MCU_VDD1",   source_id=0, target_id=6,  weight=2.5),  # U1 -> C1
        PCBNet(id=7,  net_name="MCU_VDD2",   source_id=0, target_id=7,  weight=2.5),  # U1 -> C2
        
        # Reloj de alta frecuencia (XTAL)
        PCBNet(id=8,  net_name="XTAL_IN",    source_id=0, target_id=2,  weight=3.0),  # U1 -> Y1
        PCBNet(id=9,  net_name="XTAL_OUT",   source_id=2, target_id=0,  weight=3.0),  # Y1 -> U1
        
        # Bus SPI Memoria Flash
        PCBNet(id=10, net_name="SPI_SCK",    source_id=0, target_id=3,  weight=2.0),  # U1 -> U3
        PCBNet(id=11, net_name="SPI_MOSI",   source_id=0, target_id=3,  weight=2.0),  # U1 -> U3
        PCBNet(id=12, net_name="SPI_MISO",   source_id=0, target_id=3,  weight=2.0),  # U1 -> U3
        PCBNet(id=13, net_name="SPI_CS",     source_id=0, target_id=3,  weight=1.5),  # U1 -> U3
        PCBNet(id=14, net_name="SPI_HOLD",   source_id=3, target_id=11, weight=1.2),  # U3 -> R2
        
        # Señales de control y periféricos
        PCBNet(id=15, net_name="NRST_PULL",  source_id=0, target_id=10, weight=1.2),  # U1 -> R1
        PCBNet(id=16, net_name="USB_DP",     source_id=4, target_id=0,  weight=2.0),  # J1 -> U1
        PCBNet(id=17, net_name="USB_DM",     source_id=4, target_id=0,  weight=2.0),  # J1 -> U1
        PCBNet(id=18, net_name="GPIO_LED",   source_id=0, target_id=12, weight=1.0),  # U1 -> R3
        PCBNet(id=19, net_name="UART_TX",    source_id=0, target_id=5,  weight=1.2),  # U1 -> J2
        PCBNet(id=20, net_name="UART_RX",    source_id=0, target_id=5,  weight=1.2),  # U1 -> J2
        PCBNet(id=21, net_name="I2C_SDA",    source_id=0, target_id=5,  weight=1.2),  # U1 -> J2
        PCBNet(id=22, net_name="I2C_SCL",    source_id=0, target_id=5,  weight=1.2),  # U1 -> J2
    ]

    return PCBInstance(
        board_max_width=100.0,
        board_max_height=80.0,
        clearance=2.0,
        components=components,
        nets=nets
    )


class PCBModel:
    """
    Modelo matemático y evaluador de aptitud para el diseño de PCB.

    El genotipo es un vector continuo de longitud 2 * N_componentes en el rango [0.0, 1.0].
    Para cada componente i:
      - Gen 2*i     -> Coordenada X normalizada dentro del sustrato.
      - Gen 2*i + 1 -> Coordenada Y normalizada dentro del sustrato.

    Esto garantiza compatibilidad nativa con Cruce SPX (1 punto) y Mutación Puntual.
    """

    def __init__(self, instance: PCBInstance | None = None) -> None:
        self.instance = instance or create_default_pcb_instance()
        self.num_components = len(self.instance.components)
        self.num_genes = self.num_components * 2

        # Límites de normalización basados en análisis de casos extremos
        # Wirelength ideal estimado: ~220 mm con agrupamiento óptimo de nets
        # Wirelength peor caso aleatorio: ~1200 mm
        self.wl_ideal: float = 210.0
        self.wl_worst: float = 1200.0

        # Área ideal estimada: ~1200 mm² (ej. 40mm x 30mm empaquetada)
        # Área máxima sustrato: 8000 mm² (100mm x 80mm)
        self.area_ideal: float = 1200.0
        self.area_worst: float = 6500.0

        # Ponderaciones de la función multiobjetivo
        self.w_wirelength: float = 0.60
        self.w_area: float       = 0.40
        self.lambda_overlap: float = 0.50  # Coeficiente de penalización por mm² de solapamiento

    def decode_chromosome(self, chromosome: np.ndarray) -> PCBSolution:
        """
        Decodifica el genotipo continuo en coordenadas milimétricas y evalúa solapamientos y trazas.

        Args:
            chromosome: Vector 1D de números reales de tamaño 2 * N_componentes en [0, 1].

        Returns:
            PCBSolution con el desglose geométrico y métricas de calidad.
        """
        comp_list: List[PlacedComponent] = []
        max_w = self.instance.board_max_width
        max_h = self.instance.board_max_height

        for i, comp in enumerate(self.instance.components):
            gx = float(np.clip(chromosome[2 * i], 0.0, 1.0))
            gy = float(np.clip(chromosome[2 * i + 1], 0.0, 1.0))

            half_w = comp.width / 2.0
            half_h = comp.height / 2.0

            # Mapear al área válida respetando los bordes del sustrato
            min_x_valid = half_w
            max_x_valid = max_w - half_w
            min_y_valid = half_h
            max_y_valid = max_h - half_h

            x = min_x_valid + gx * (max_x_valid - min_x_valid)
            y = min_y_valid + gy * (max_y_valid - min_y_valid)

            left   = x - half_w
            right  = x + half_w
            bottom = y - half_h
            top    = y + half_h

            comp_list.append(PlacedComponent(
                id=comp.id,
                ref=comp.ref,
                name=comp.name,
                x=x,
                y=y,
                width=comp.width,
                height=comp.height,
                category=comp.category,
                color=comp.color,
                left=left,
                right=right,
                bottom=bottom,
                top=top
            ))

        # 1. Cálculo de Bounding Box (Área envolvente del circuito)
        min_bb_x = min(c.left for c in comp_list)
        max_bb_x = max(c.right for c in comp_list)
        min_bb_y = min(c.bottom for c in comp_list)
        max_bb_y = max(c.top for c in comp_list)

        bb_w = max(1.0, max_bb_x - min_bb_x)
        bb_h = max(1.0, max_bb_y - min_bb_y)
        bb_area = bb_w * bb_h

        # 2. Detección y cuantificación de solapamientos físicos (con holgura de seguridad)
        clearance = self.instance.clearance
        total_overlap_area = 0.0
        overlap_count = 0
        n = len(comp_list)

        for i in range(n):
            c1 = comp_list[i]
            for j in range(i + 1, n):
                c2 = comp_list[j]
                # Distancia entre centros vs suma de semianchos + holgura
                dx = (c1.width + c2.width) / 2.0 + clearance - abs(c1.x - c2.x)
                dy = (c1.height + c2.height) / 2.0 + clearance - abs(c1.y - c2.y)

                if dx > 0.0 and dy > 0.0:
                    overlap_count += 1
                    total_overlap_area += (dx * dy)

        # 3. Longitud de trazas (Wirelength Manhattan y Euclidiano ponderado)
        total_manhattan_wl = 0.0
        total_euclidean_wl = 0.0

        for net in self.instance.nets:
            c_src = comp_list[net.source_id]
            c_tgt = comp_list[net.target_id]

            manhattan_dist = abs(c_src.x - c_tgt.x) + abs(c_src.y - c_tgt.y)
            euclidean_dist = np.sqrt((c_src.x - c_tgt.x)**2 + (c_src.y - c_tgt.y)**2)

            total_manhattan_wl += (manhattan_dist * net.weight)
            total_euclidean_wl += (euclidean_dist * net.weight)

        # 4. Función de Aptitud Normalizada (Minimización en [0, 1])
        norm_wl = max(0.0, (total_manhattan_wl - self.wl_ideal) / (self.wl_worst - self.wl_ideal))
        norm_area = max(0.0, (bb_area - self.area_ideal) / (self.area_worst - self.area_ideal))

        penalty = self.lambda_overlap * total_overlap_area

        fitness_val = (self.w_wirelength * norm_wl + self.w_area * norm_area) + penalty
        is_feasible = (overlap_count == 0)

        return PCBSolution(
            placed_components=comp_list,
            total_wirelength_mm=total_euclidean_wl,
            manhattan_wirelength_mm=total_manhattan_wl,
            bounding_box_width=bb_w,
            bounding_box_height=bb_h,
            bounding_box_area_mm2=bb_area,
            total_overlap_area_mm2=total_overlap_area,
            raw_fitness=fitness_val,
            normalized_fitness=fitness_val,
            is_feasible=is_feasible,
            overlap_count=overlap_count
        )

    def evaluate_fitness(self, chromosome: np.ndarray) -> float:
        """Función objetivo para el motor SGA (retorna float a minimizar)."""
        sol = self.decode_chromosome(chromosome)
        return sol.normalized_fitness

    def plot_solution(self, solution: PCBSolution, save_path: str, dpi: int = 300) -> None:
        """
        Genera una figura técnica con el diseño del PCB y el trazado de pistas.
        """
        fig, ax = plt.subplots(figsize=(10, 8), dpi=dpi)
        fig.patch.set_facecolor('#0F172A')
        ax.set_facecolor('#0B1120')

        # Dibujar sustrato del PCB (Board outline)
        board_w = self.instance.board_max_width
        board_h = self.instance.board_max_height
        board_rect = patches.Rectangle(
            (0, 0), board_w, board_h,
            linewidth=2, edgecolor='#1E293B', facecolor='#0B1120',
            linestyle='--', alpha=0.8
        )
        ax.add_patch(board_rect)

        # Dibujar Bounding Box envolvente del circuito optimizado
        comp_xs = [c.x for c in solution.placed_components]
        comp_ys = [c.y for c in solution.placed_components]
        min_x = min(c.left for c in solution.placed_components)
        min_y = min(c.bottom for c in solution.placed_components)
        bb_rect = patches.Rectangle(
            (min_x, min_y), solution.bounding_box_width, solution.bounding_box_height,
            linewidth=1.5, edgecolor='#10B981', facecolor='none',
            linestyle=':', alpha=0.9, label=f"Bounding Box: {solution.bounding_box_area_mm2:.1f} mm²"
        )
        ax.add_patch(bb_rect)

        # Dibujar trazas de cobre (Nets)
        comp_map = {c.id: c for c in solution.placed_components}
        for net in self.instance.nets:
            c1 = comp_map[net.source_id]
            c2 = comp_map[net.target_id]
            # Trazado ortogonal (Manhattan con estilo de circuito)
            mid_x = (c1.x + c2.x) / 2.0
            ax.plot([c1.x, mid_x, mid_x, c2.x], [c1.y, c1.y, c2.y, c2.y],
                    color='#38BDF8', alpha=0.35, linewidth=1.2, zorder=2)

        # Dibujar componentes electrónicos (Pads y serigrafía)
        for c in solution.placed_components:
            # Rectángulo del componente
            rect = patches.FancyBboxPatch(
                (c.left, c.bottom), c.width, c.height,
                boxstyle="round,pad=0.5",
                linewidth=1.5, edgecolor=c.color,
                facecolor='#1E293B', alpha=0.9, zorder=3
            )
            ax.add_patch(rect)

            # Serigrafía (Texto identificador)
            ax.text(c.x, c.y + 1.0, c.ref,
                    color='#F8FAFC', fontsize=8, fontweight='bold',
                    ha='center', va='center', zorder=4)
            ax.text(c.x, c.y - 1.5, f"{c.width:.0f}x{c.height:.0f}",
                    color='#94A3B8', fontsize=6,
                    ha='center', va='center', zorder=4)

        ax.set_xlim(-5, board_w + 5)
        ax.set_ylim(-5, board_h + 5)
        ax.set_aspect('equal', adjustable='box')

        ax.grid(True, linestyle=':', color='#334155', alpha=0.4)
        ax.set_title(
            f"PCB Layout Optimizado — Wirelength: {solution.manhattan_wirelength_mm:.1f} mm | Área: {solution.bounding_box_area_mm2:.1f} mm²\n"
            f"Fitness: {solution.normalized_fitness:.5f} | Factibilidad: {'100% (Sin Solapamientos)' if solution.is_feasible else 'Infracción'}",
            color='#F8FAFC', fontsize=11, fontweight='bold', pad=14
        )
        ax.set_xlabel("Eje X (mm)", color='#94A3B8', fontsize=9)
        ax.set_ylabel("Eje Y (mm)", color='#94A3B8', fontsize=9)
        ax.tick_params(colors='#94A3B8', labelsize=8)

        # Marco exterior y estética
        for spine in ax.spines.values():
            spine.set_color('#334155')

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, facecolor=fig.get_facecolor(), edgecolor='none', dpi=dpi)
        plt.close(fig)


def plot_pcb_convergence(
    history: List[GenerationStats],
    tolerance: float,
    save_path: str,
    dpi: int = 300
) -> None:
    """
    Genera y guarda la gráfica de convergencia generacional para el Problema 2 (PCB Layout).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)

    gens = [s.generation for s in history]
    bests = [s.best_fitness for s in history]
    avgs = [s.avg_fitness for s in history]
    worsts = [s.worst_fitness for s in history]

    ax.plot(gens, bests, color='#10B981', linewidth=2.5, label=r'Mejor Fitness ($f_{min}$)')
    ax.plot(gens, avgs, color='#3B82F6', linewidth=1.8, linestyle='--', label=r'Fitness Promedio ($\bar{f}$)')
    ax.plot(gens, worsts, color='#EF4444', linewidth=1.2, alpha=0.7, linestyle=':', label=r'Peor Fitness ($f_{max}$)')

    # Línea umbral del criterio de parada (10^-2 = 0.01)
    ax.axhline(
        y=tolerance,
        color='#8B5CF6',
        linestyle='-.',
        linewidth=2.0,
        label=f'Criterio de Parada (Fitness < {tolerance:.2f})'
    )

    ax.set_title(
        "Curvas de Convergencia Generacional - Algoritmo Genético Simple (Problema 2: PCB Layout)\n"
        "Población: N=1000 | Elitismo: 2 | Cruce: SPX 90% | Mutación: Puntual 2%",
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
    plt.close(fig)
