"""
Problema 3: Optimización de Cartera de Inversiones (Markowitz Portfolio Selection).
==================================================================================

Modelado matemático, decodificación cromosómica, función de aptitud multiobjetivo
con penalizaciones cuadráticas y trazado de frontera eficiente para la selección
óptima de cartera de activos bajo la Teoría Moderna de Portafolios (Markowitz, 1952).

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

from src.core.ga_engine import GenerationStats, GAParams, GAResult


@dataclass(frozen=True)
class Asset:
    """
    Activo financiero disponible en el universo de inversión.

    Atributos:
        id (int): Identificador numérico único (0 a M-1).
        ticker (str): Símbolo bursátil (ej. 'AAPL', 'MSFT', 'TLT').
        name (str): Razón social o denominación del instrumento.
        sector (str): Categoría económica o clase de activo.
        expected_return (float): Rendimiento medio anualizado proyectado (mu_i).
        volatility (float): Desviación estándar anualizada del rendimiento (sigma_i).
        color (str): Código hexadecimal de color para visualización de gráficos.
    """
    id: int
    ticker: str
    name: str
    sector: str
    expected_return: float
    volatility: float
    color: str


@dataclass(frozen=True)
class PortfolioInstance:
    """
    Especificación del universo de inversión y parámetros del problema de Markowitz.

    Atributos:
        assets (List[Asset]): Conjunto de activos elegibles.
        correlation_matrix (np.ndarray): Matriz de correlaciones lineales de Pearson (M x M).
        covariance_matrix (np.ndarray): Matriz de varianzas-covarianzas D * R * D (M x M).
        risk_free_rate (float): Tasa libre de riesgo anualizada (Rf, ej. 3.5%).
        max_asset_exposure (float): Límite superior de asignación por activo individual (w_max).
        weight_risk (float): Ponderación del riesgo en la función multicriterio (alpha).
        weight_return (float): Ponderación del retorno en la función multicriterio (1 - alpha).
        lambda_budget (float): Coeficiente de penalización cuadrática por violación del presupuesto.
        lambda_exposure (float): Coeficiente de penalización cuadrática por exceso de exposición.
    """
    assets: List[Asset] = field(default_factory=list)
    correlation_matrix: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    covariance_matrix: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    risk_free_rate: float = 0.035
    max_asset_exposure: float = 0.35
    weight_risk: float = 0.50
    weight_return: float = 0.50
    lambda_budget: float = 15.0
    lambda_exposure: float = 20.0


@dataclass
class PortfolioSolution:
    """
    Solución decodificada de una asignación patrimonial en la cartera.

    Atributos:
        raw_weights (np.ndarray): Pesos directos decodificados del genotipo (suman approx 1.0).
        normalized_weights (np.ndarray): Pesos normalizados al 100% estricto de capital.
        expected_return (float): Retorno medio esperado del portafolio E[Rp] = w^T * mu.
        volatility (float): Volatilidad (riesgo) anualizada sigma_p = sqrt(w^T * Sigma * w).
        variance (float): Varianza total del portafolio sigma_p^2.
        sharpe_ratio (float): Ratio de Sharpe = (E[Rp] - Rf) / sigma_p.
        budget_sum (float): Suma bruta de las ponderaciones patrimoniales.
        budget_error (float): Discrepancia absoluta con respecto al presupuesto unitario |sum(w) - 1.0|.
        max_weight (float): Máxima exposición porcentual asignada a un único activo.
        exposure_violations (int): Cantidad de activos que exceden el límite w_max.
        is_feasible (bool): True si budget_error <= 0.015 y ningún activo supera w_max + 1e-4.
        raw_fitness (float): Valor directo evaluado de aptitud (menor es mejor).
        normalized_fitness (float): Aptitud escalada en espacio de minimización.
        asset_allocations (List[Dict]): Desglose detallado por activo para reporte y tablas.
    """
    raw_weights: np.ndarray
    normalized_weights: np.ndarray
    expected_return: float
    volatility: float
    variance: float
    sharpe_ratio: float
    budget_sum: float
    budget_error: float
    max_weight: float
    exposure_violations: int
    is_feasible: bool
    raw_fitness: float
    normalized_fitness: float
    asset_allocations: List[Dict] = field(default_factory=list)


def create_default_portfolio_instance() -> PortfolioInstance:
    """
    Crea la instancia canónica de evaluación para el Problema 3:
    Canasta diversificada de 10 activos financieros que representan sectores clave de la economía
    global (Tecnología, Consumo Cíclico, Salud Defensiva, Finanzas, Energía, Bonos Soberanos,
    Materias Primas / Oro y Fideicomisos Inmobiliarios REITs).
    """
    assets = [
        Asset(id=0, ticker="AAPL",  name="Apple Inc.",               sector="Tecnología",      expected_return=0.185, volatility=0.220, color="#10B981"),
        Asset(id=1, ticker="MSFT",  name="Microsoft Corp.",          sector="Tecnología",      expected_return=0.170, volatility=0.205, color="#06B6D4"),
        Asset(id=2, ticker="GOOGL", name="Alphabet Inc.",            sector="Tecnología",      expected_return=0.160, volatility=0.210, color="#3B82F6"),
        Asset(id=3, ticker="AMZN",  name="Amazon.com Inc.",          sector="Consumo Cíclico", expected_return=0.155, volatility=0.235, color="#F59E0B"),
        Asset(id=4, ticker="JNJ",   name="Johnson & Johnson",        sector="Salud",           expected_return=0.085, volatility=0.120, color="#EC4899"),
        Asset(id=5, ticker="JPM",   name="JPMorgan Chase",           sector="Finanzas",        expected_return=0.120, volatility=0.180, color="#6366F1"),
        Asset(id=6, ticker="XOM",   name="Exxon Mobil",              sector="Energía",         expected_return=0.105, volatility=0.195, color="#EF4444"),
        Asset(id=7, ticker="TLT",   name="iShares 20+ Y Treasury",   sector="Renta Fija",      expected_return=0.045, volatility=0.095, color="#8B5CF6"),
        Asset(id=8, ticker="GLD",   name="SPDR Gold Shares",         sector="Materias Primas", expected_return=0.070, volatility=0.130, color="#EAB308"),
        Asset(id=9, ticker="VNQ",   name="Vanguard Real Estate REIT",sector="Bienes Raíces",   expected_return=0.090, volatility=0.165, color="#14B8A6"),
    ]

    # Matriz empírica de correlaciones de Pearson (10 x 10)
    # Refleja: alta correlación intrasectorial tecnológica (~0.7), correlación negativa
    # de bonos del tesoro (TLT) con renta variable (~ -0.25), y descorrelación del oro (GLD).
    corr = np.array([
        # AAPL  MSFT  GOOGL AMZN  JNJ   JPM   XOM   TLT   GLD   VNQ
        [ 1.00, 0.72, 0.68, 0.65, 0.25, 0.42, 0.30,-0.22, 0.05, 0.38], # AAPL
        [ 0.72, 1.00, 0.75, 0.62, 0.28, 0.40, 0.28,-0.25, 0.02, 0.35], # MSFT
        [ 0.68, 0.75, 1.00, 0.60, 0.22, 0.38, 0.25,-0.20, 0.04, 0.32], # GOOGL
        [ 0.65, 0.62, 0.60, 1.00, 0.20, 0.35, 0.22,-0.18, 0.08, 0.40], # AMZN
        [ 0.25, 0.28, 0.22, 0.20, 1.00, 0.30, 0.25, 0.15, 0.10, 0.28], # JNJ
        [ 0.42, 0.40, 0.38, 0.35, 0.30, 1.00, 0.45,-0.30, 0.00, 0.48], # JPM
        [ 0.30, 0.28, 0.25, 0.22, 0.25, 0.45, 1.00,-0.15, 0.20, 0.30], # XOM
        [-0.22,-0.25,-0.20,-0.18, 0.15,-0.30,-0.15, 1.00, 0.25, 0.10], # TLT
        [ 0.05, 0.02, 0.04, 0.08, 0.10, 0.00, 0.20, 0.25, 1.00, 0.12], # GLD
        [ 0.38, 0.35, 0.32, 0.40, 0.28, 0.48, 0.30, 0.10, 0.12, 1.00], # VNQ
    ], dtype=float)

    # Construcción de la matriz de covarianzas Sigma = D * R * D
    vols = np.array([a.volatility for a in assets], dtype=float)
    D = np.diag(vols)
    cov = D @ corr @ D

    return PortfolioInstance(
        assets=assets,
        correlation_matrix=corr,
        covariance_matrix=cov,
        risk_free_rate=0.035,        # 3.5% anual
        max_asset_exposure=0.35,     # Máximo 35% por activo individual
        weight_risk=0.50,            # Ponderación 50% riesgo
        weight_return=0.50,          # Ponderación 50% retorno
        lambda_budget=15.0,          # Penalización estricta por desviación de suma unitaria
        lambda_exposure=20.0         # Penalización estricta por superación de techo w_max
    )


class MarkowitzModel:
    """
    Modelo matemático y evaluador de aptitud para la optimización de carteras según Markowitz.

    El genotipo es un vector continuo de M genes reales en el rango [0.0, 0.40],
    donde cada gen representa la asignación presupuestal directa del activo i.
    La presión selectiva del SGA, mediante penalizaciones cuadráticas exteriores,
    obliga a los individuos a converger hacia sum(w_i) = 1.0 y w_i <= 0.35.
    """

    def __init__(self, instance: PortfolioInstance) -> None:
        self.instance = instance
        self.num_assets = len(instance.assets)
        self.num_genes = self.num_assets

        # Vectores y matrices precalculadas
        self.returns_vec = np.array([a.expected_return for a in instance.assets], dtype=float)
        self.vols_vec = np.array([a.volatility for a in instance.assets], dtype=float)
        self.cov_mat = instance.covariance_matrix

        # Parámetros de normalización de la función de aptitud
        self.sigma_max = float(np.max(self.vols_vec) * 1.10)  # Cota superior de volatilidad (~0.258)
        self.mu_min = float(np.min(self.returns_vec) * 0.90)   # Cota inferior de retorno (~0.040)
        self.mu_max = float(np.max(self.returns_vec) * 1.05)   # Cota superior de retorno (~0.194)

    def decode_chromosome(self, chromosome: np.ndarray) -> PortfolioSolution:
        """
        Decodifica el genotipo continuo en una solución formal de portafolio de inversión.

        Args:
            chromosome: Vector continuo de M genes en [0.0, 0.40].

        Returns:
            PortfolioSolution con todos los estadísticos financieros y de factibilidad.
        """
        raw_w = np.array(chromosome, dtype=float)
        budget_sum = float(np.sum(raw_w))
        budget_err = abs(budget_sum - 1.0)

        # Vector de pesos normalizado para cálculo representativo al 100%
        if budget_sum > 1e-6:
            norm_w = raw_w / budget_sum
        else:
            norm_w = np.ones(self.num_assets) / self.num_assets

        # Métricas financieras sobre el vector bruto
        ret = float(np.dot(raw_w, self.returns_vec))
        var = float(raw_w @ self.cov_mat @ raw_w)
        vol = float(np.sqrt(max(0.0, var)))

        # Ratio de Sharpe
        rf = self.instance.risk_free_rate
        sharpe = (ret - rf) / vol if vol > 1e-6 else 0.0

        # Conteo de violaciones de exposición
        w_max = self.instance.max_asset_exposure
        violations = int(np.sum(raw_w > (w_max + 1e-4)))
        max_w = float(np.max(raw_w))

        # Factibilidad estricta
        is_feasible = (budget_err <= 0.015) and (violations == 0)

        # Cálculo de penalizaciones cuadráticas
        pen_budget = self.instance.lambda_budget * ((budget_sum - 1.0) ** 2)
        pen_exposure = self.instance.lambda_exposure * float(np.sum(np.maximum(0.0, raw_w - w_max) ** 2))
        pen_negative = 30.0 * float(np.sum(np.maximum(0.0, -raw_w) ** 2))

        # Términos normalizados de la función objetivo
        f_risk = vol / self.sigma_max
        f_ret = (self.mu_max - ret) / (self.mu_max - self.mu_min)

        f_base = (self.instance.weight_risk * f_risk) + (self.instance.weight_return * f_ret)
        f_total = f_base + pen_budget + pen_exposure + pen_negative

        # Desglose estructurado por activo
        allocations = []
        for i, asset in enumerate(self.instance.assets):
            allocations.append({
                "id": asset.id,
                "ticker": asset.ticker,
                "name": asset.name,
                "sector": asset.sector,
                "raw_weight": round(float(raw_w[i]), 4),
                "norm_weight": round(float(norm_w[i]), 4),
                "percentage": round(float(norm_w[i]) * 100, 2),
                "color": asset.color,
                "expected_return": asset.expected_return,
                "volatility": asset.volatility,
            })

        return PortfolioSolution(
            raw_weights=raw_w,
            normalized_weights=norm_w,
            expected_return=ret,
            volatility=vol,
            variance=var,
            sharpe_ratio=sharpe,
            budget_sum=budget_sum,
            budget_error=budget_err,
            max_weight=max_w,
            exposure_violations=violations,
            is_feasible=is_feasible,
            raw_fitness=f_total,
            normalized_fitness=f_total,
            asset_allocations=allocations
        )

    def evaluate_fitness(self, chromosome: np.ndarray) -> float:
        """
        Función de aptitud escalarizada para el motor genético simple (SGA).
        El SGA minimiza este valor.
        """
        w = np.array(chromosome, dtype=float)
        budget_sum = float(np.sum(w))
        
        pen_budget = self.instance.lambda_budget * ((budget_sum - 1.0) ** 2)
        pen_exposure = self.instance.lambda_exposure * float(np.sum(np.maximum(0.0, w - self.instance.max_asset_exposure) ** 2))
        pen_negative = 30.0 * float(np.sum(np.maximum(0.0, -w) ** 2))

        ret = float(np.dot(w, self.returns_vec))
        var = float(w @ self.cov_mat @ w)
        vol = float(np.sqrt(max(0.0, var)))

        f_risk = vol / self.sigma_max
        f_ret = (self.mu_max - ret) / (self.mu_max - self.mu_min)

        f_total = (self.instance.weight_risk * f_risk) + (self.instance.weight_return * f_ret) + pen_budget + pen_exposure + pen_negative
        return float(f_total)

    def generate_monte_carlo_cloud(self, n_samples: int = 3000, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Genera una nube representativa de portafolios factibles mediante muestreo estocástico
        con distribución Dirichlet uniforme para proyectar la región factible y la frontera eficiente.

        Returns:
            Tuple[vols, returns, sharpes]
        """
        rng = np.random.default_rng(seed)
        vols = []
        rets = []
        sharpes = []
        rf = self.instance.risk_free_rate
        w_max = self.instance.max_asset_exposure

        # Muestreo con rechazo para cumplir cota de exposición
        count = 0
        while count < n_samples:
            raw_sample = rng.dirichlet(np.ones(self.num_assets))
            if np.all(raw_sample <= w_max + 0.05):
                ret = float(np.dot(raw_sample, self.returns_vec))
                vol = float(np.sqrt(max(0.0, raw_sample @ self.cov_mat @ raw_sample)))
                sharpe = (ret - rf) / vol if vol > 1e-6 else 0.0
                vols.append(vol)
                rets.append(ret)
                sharpes.append(sharpe)
                count += 1

        return np.array(vols), np.array(rets), np.array(sharpes)

    def plot_efficient_frontier(
        self,
        best_solution: PortfolioSolution,
        output_path: str,
        n_samples: int = 3000,
        dpi: int = 300
    ) -> None:
        """
        Genera el gráfico comparativo de la Frontera Eficiente de Markowitz,
        incluyendo la nube Monte Carlo, los activos individuales y el portafolio óptimo del SGA.
        """
        vols, rets, sharpes = self.generate_monte_carlo_cloud(n_samples=n_samples)

        plt.figure(figsize=(10, 6.5), facecolor="#0B132B")
        ax = plt.gca()
        ax.set_facecolor("#0F172A")

        # Nube Monte Carlo
        sc = ax.scatter(
            vols * 100,
            rets * 100,
            c=sharpes,
            cmap="viridis",
            alpha=0.45,
            s=12,
            edgecolors="none",
            label="Portafolios Factibles (Monte Carlo)"
        )

        cbar = plt.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label("Ratio de Sharpe (SR)", color="#94A3B8", fontsize=10)
        cbar.ax.yaxis.set_tick_params(color="#94A3B8")
        plt.setp(cbar.ax.get_yticklabels(), color="#94A3B8")

        # Activos individuales
        for asset in self.instance.assets:
            ax.scatter(
                asset.volatility * 100,
                asset.expected_return * 100,
                color=asset.color,
                s=70,
                zorder=5,
                edgecolors="#FFFFFF",
                linewidth=1.2
            )
            ax.annotate(
                asset.ticker,
                (asset.volatility * 100 + 0.35, asset.expected_return * 100 - 0.2),
                color="#E2E8F0",
                fontsize=8.5,
                weight="bold",
                zorder=6
            )

        # Portafolio Óptimo SGA
        opt_vol = best_solution.volatility * 100
        opt_ret = best_solution.expected_return * 100
        ax.scatter(
            opt_vol,
            opt_ret,
            color="#10B981",
            s=220,
            marker="*",
            zorder=10,
            edgecolors="#FFFFFF",
            linewidth=1.5,
            label=f"Óptimo SGA (Ret: {opt_ret:.2f}%, Vol: {opt_vol:.2f}%, SR: {best_solution.sharpe_ratio:.2f})"
        )

        # Capital Allocation Line (CAL) aproximada desde Rf
        rf_pct = self.instance.risk_free_rate * 100
        cal_x = np.linspace(0, max(vols) * 100, 100)
        cal_y = rf_pct + best_solution.sharpe_ratio * cal_x
        ax.plot(cal_x, cal_y, color="#38BDF8", linestyle="--", linewidth=1.5, alpha=0.75, label="Línea de Asignación de Capital (CAL)")

        ax.set_title(
            "Frontera Eficiente de Markowitz y Portafolio Óptimo SGA\n"
            "Optimización de Rendimiento vs. Riesgo (10 Activos Globales)",
            color="#F8FAFC",
            fontsize=12,
            weight="bold",
            pad=15
        )
        ax.set_xlabel("Riesgo / Volatilidad Anualizada sigma_p (%)", color="#94A3B8", fontsize=10)
        ax.set_ylabel("Rendimiento Medio Anual Esperado E[Rp] (%)", color="#94A3B8", fontsize=10)

        ax.tick_params(colors="#94A3B8")
        ax.grid(color="#334155", linestyle=":", alpha=0.6)
        ax.set_xlim(left=5.0, right=26.0)
        ax.set_ylim(bottom=3.0, top=21.0)

        legend = ax.legend(loc="upper left", facecolor="#1E293B", edgecolor="#334155")
        for text in legend.get_texts():
            text.set_color("#F1F5F9")

        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=dpi, facecolor="#0B132B")
        plt.close()

    def plot_asset_allocation(
        self,
        best_solution: PortfolioSolution,
        output_path: str,
        dpi: int = 300
    ) -> None:
        """Genera un gráfico de barras horizontales con la distribución de capital del portafolio."""
        tickers = [a["ticker"] for a in best_solution.asset_allocations]
        pcts = [a["percentage"] for a in best_solution.asset_allocations]
        colors = [a["color"] for a in best_solution.asset_allocations]

        plt.figure(figsize=(9, 5.5), facecolor="#0B132B")
        ax = plt.gca()
        ax.set_facecolor("#0F172A")

        y_pos = np.arange(len(tickers))
        bars = ax.barh(y_pos, pcts, color=colors, edgecolor="#1E293B", height=0.65)

        # Límite de exposición de cátedra (35%)
        ax.axvline(
            self.instance.max_asset_exposure * 100,
            color="#EF4444",
            linestyle="--",
            linewidth=1.5,
            label=f"Límite Máximo de Exposición ({self.instance.max_asset_exposure * 100:.0f}%)"
        )

        for bar, pct in zip(bars, pcts):
            width = bar.get_width()
            ax.text(
                width + 0.6,
                bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%",
                va="center",
                ha="left",
                color="#F8FAFC",
                fontsize=9,
                weight="bold"
            )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(tickers, color="#F1F5F9", fontsize=10, weight="bold")
        ax.invert_yaxis()  # El activo 1 arriba
        ax.set_xlabel("Asignación de Capital (%)", color="#94A3B8", fontsize=10)
        ax.set_title(
            f"Estructura Patrimonial del Portafolio Óptimo (SGA)\n"
            f"Presupuesto: {best_solution.budget_sum * 100:.2f}% | Diversificación: 100% Factible",
            color="#F8FAFC",
            fontsize=12,
            weight="bold",
            pad=15
        )

        ax.tick_params(colors="#94A3B8")
        ax.grid(axis="x", color="#334155", linestyle=":", alpha=0.6)
        ax.set_xlim(0, 42.0)

        legend = ax.legend(loc="lower right", facecolor="#1E293B", edgecolor="#334155")
        for text in legend.get_texts():
            text.set_color("#F1F5F9")

        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=dpi, facecolor="#0B132B")
        plt.close()


def plot_markowitz_convergence(
    history: List[GenerationStats],
    tolerance: float,
    output_path: str,
    dpi: int = 300
) -> None:
    """Genera la curva de convergencia generacional del SGA para el Problema 3."""
    gens = [s.generation for s in history]
    best = [s.best_fitness for s in history]
    avg = [s.avg_fitness for s in history]
    worst = [s.worst_fitness for s in history]

    plt.figure(figsize=(10, 5.5), facecolor="#0B132B")
    ax = plt.gca()
    ax.set_facecolor("#0F172A")

    ax.plot(gens, worst, color="#EF4444", linestyle=":", alpha=0.65, label="Peor de la Población")
    ax.plot(gens, avg, color="#38BDF8", linestyle="--", linewidth=1.6, label="Promedio Poblacional")
    ax.plot(gens, best, color="#10B981", linewidth=2.4, label="Mejor Individuo (Élite k=2)")

    ax.axhline(tolerance, color="#F59E0B", linestyle="-.", linewidth=1.2, label=f"Tolerancia Parada ({tolerance})")

    ax.set_yscale("log")
    ax.set_title(
        "Curva de Convergencia del SGA — Problema 3: Cartera de Markowitz\n"
        "(Población N=1000, Cruce SPX Pc=90%, Mutación Pm=2%, Selección Torneo p=2)",
        color="#F8FAFC",
        fontsize=11.5,
        weight="bold",
        pad=15
    )
    ax.set_xlabel("Generación", color="#94A3B8", fontsize=10)
    ax.set_ylabel("Fitness Normalizado (Escala Log)", color="#94A3B8", fontsize=10)

    ax.tick_params(colors="#94A3B8")
    ax.grid(True, which="both", color="#334155", linestyle=":", alpha=0.6)

    legend = ax.legend(loc="upper right", facecolor="#1E293B", edgecolor="#334155")
    for text in legend.get_texts():
        text.set_color("#F1F5F9")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=dpi, facecolor="#0B132B")
    plt.close()
