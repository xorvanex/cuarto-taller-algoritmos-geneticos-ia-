# Cuarto Taller — Trabajo Final de Algoritmos Genéticos (AG)

# Inteligencia Artificial — Corte 3

> **Institución:** Universidad de Cartagena — Facultad de Ingeniería  
> **Programa:** Ingeniería de Sistemas — Semestre VII  
> **Asignatura:** Inteligencia Artificial (Corte 3)  
> **Docente:** Manuel Alejandro Ospina Alarcón  
> **Estudiante:** Dago David Palmera Navarro  
> **Fecha de Publicación:** 27 de octubre de 2026  
> **Fecha Máxima de Entrega:** Jueves 05 de noviembre de 2026  
> **Ponderación:** 10 puntos de la calificación definitiva del Tercer Corte  
> **Modalidad:** Grupos de máximo 2 personas  

---

## 1. Estructura del Proyecto

La estructura del directorio sigue el estándar establecido en los talleres anteriores del curso (`primer-proyecto-integrador-ia`, `segundo-taller-perceptron-ia` y `tercer-taller-redes-multicapa-ia`):

```text
cuarto-taller-algoritmos-geneticos-ia/
├── data/                                 # Datos de entrada, matrices y configuración de problemas
├── docs/                                 # Documentación oficial y especificaciones de la prueba
│   └── Prueba final AG.pdf               # Enunciado oficial de la actividad de trabajo colaborativo
├── algoritmos_referencia/                # Códigos base vistos en clase (convertidos a .txt limpios)
│   ├── Cruce.txt                         # Operador de cruce en un punto (SPX) con probabilidad Pc=0.9
│   ├── Elitismo.txt                      # Operador de elitismo (preserva los 2 mejores individuos)
│   ├── Fitness.txt                       # Función de evaluación de bondad/aptitud (error absoluto)
│   ├── Mutacion.txt                      # Operador de mutación puntual con alelo aleatorio (Pm)
│   ├── seleccion_torneo_2.txt            # Selección por torneo determinístico p=2 y selección aleatoria
│   └── SELPruebaAG.txt                   # Algoritmo genético canónico completo para sistema 3x3
├── src/                                  # Implementación en desarrollo para los problemas del trabajo final
├── Entrega/                              # Informes finales, capturas de convergencia y entregables
└── README.md                             # Documentación global del entorno de trabajo
```

---

## 2. Parámetros de Configuración Canónicos

De acuerdo con las directrices dadas en el documento de prueba:

* **Tamaño de población:** 1000 individuos.
* **Elitismo:** 2 individuos directos a la siguiente generación.
* **Algoritmo de selección:** Torneo determinístico ($p = 2$).
* **Algoritmo de cruce:** Cruce en un punto (SPX — información después del corte).
* **Tasa de cruce ($P_c$):** 90% ($0.90$).
* **Tasa de mutación ($P_m$):** 2% ($0.02$) — (o $2.5\%$).
* **Algoritmo de mutación:** Mutación puntual de un gen con alelo aleatorio.
* **Criterio de parada:** Individuo con fitness inferior a $10^{-2}$ ($< 0.01$) o límite generacional.

---

## 3. Problemas Asignados en la Prueba Final

### Problema 1: Optimización de Rutas de Entrega (Vehicle Routing Problem - VRP)
* **Contexto:** Minimización de costos de transporte (distancia recorrida y tiempo) en una flota de reparto logístico.
* **Restricciones:** Capacidad máxima de carga por camión y tiempo límite de jornada laboral.
* **Estrategia AG:** Cromosoma representativo de secuencias de paradas/rutas, función de fitness basada en longitud euclidiana y penalización por exceso de peso/tiempo.

### Problema 2: Optimización del Diseño de Circuitos Electrónicos (PCB Layout)
* **Contexto:** Ubicación óptima de componentes electrónicos sobre una placa de circuito impreso para minimizar espacio y reducir interferencia electromagnética.
* **Restricciones:** Conexiones cortas, distancias mínimas de seguridad sin colisiones físicas de pistas.
* **Estrategia AG:** Cromosoma que codifica coordenadas $(x, y)$ y capas de conexionado, evaluado mediante área convexa y longitud de pistas.

### Problema 3: Optimización de Cartera de Inversiones (Markowitz Portfolio Optimization)
* **Contexto:** Selección de ponderación de activos financieros para maximizar rendimiento esperado y minimizar riesgo (volatilidad/varianza).
* **Restricciones:** Presupuesto fijo total ($\sum w_i = 1$) y límites máximos por activo ($w_i \le w_{\max}$).
* **Estrategia AG:** Cromosoma continuo de pesos de capital normalizados con penalizaciones por superación de riesgo.

---

## 4. Requisitos de Entrega
1. Código fuente funcional e interactivo.
2. Población inicial documentada y evaluada.
3. Población final resultante tras el proceso evolutivo.
4. Valor o curva del mejor fitness obtenido (mínimo o máximo según el caso).
5. Análisis detallado de convergencia y evolución generacional.
6. Gráficas de aptitud (mejor, promedio y peor individuo por época), análisis de errores y conclusiones.
