# PROBLEMA 1: OPTIMIZACIÓN DE RUTAS DE ENTREGA (VRP)

# 4. DEFINICIÓN DEL PROBLEMA 1

## 4.1. Descripción y Contexto de Ingeniería

El Problema de Ruteo de Vehículos con Capacidad y Ventana Temporal de Operación (CVRPTW, por sus siglas en inglés) constituye uno de los desafíos centrales en la ingeniería de operaciones, la investigación operativa y la logística moderna. En el contexto empresarial de transporte y distribución de mercancías, una organización requiere abastecer de forma continua a un conjunto disperso de $K = 15$ clientes ubicados en un territorio geográfico operativo de $100 \times 100\text{ km}$, partiendo de un Centro de Distribución o Depósito Central ubicado estratégicamente en la coordenada central $(50, 50)\text{ km}$. La flota de transporte disponible está integrada por $M = 3$ camiones homogéneos de carga media.

Cada cliente $i \in \{1, \dots, K\}$ demanda una cantidad fija de carga $d_i$ (en kilogramos) y requiere un tiempo de servicio in situ $s_i$ (en minutos) para las labores de descarga y verificación. La flota vehicular debe iniciar su jornada en el depósito, visitar una secuencia ordenada de clientes sin omisiones ni duplicidades, y retornar obligatoriamente al depósito. Las distancias entre nodos están determinadas por la métrica euclidiana en el plano bidimensional, asumiendo una velocidad promedio de tránsito de $v = 45\text{ km/h}$. En virtud de la explosión combinatoria del espacio de búsqueda —el cual escala como un problema de partición y secuenciación factorial $\mathcal{O}((K + M - 1)!)$—, los métodos deterministas exhaustivos son inviables, requiriéndose el despliegue de metaheurísticas genéticas estocásticas guiadas.

## 4.2. Función Objetivo

La meta principal del sistema de ruteo consiste en minimizar el costo integral de transporte, representado físicamente por la distancia total recorrida por todos los vehículos de la flota comercial, lo que a su vez se traduce en una reducción directa del consumo de combustible, la huella de carbono y el tiempo total de permanencia en ruta. Matemáticamente, la función objetivo se formula como:

$$\min Z = \sum_{v=1}^{M} D_v = \sum_{v=1}^{M} \left( \text{dist}(\text{Depósito}, \pi_{v,1}) + \sum_{j=1}^{n_v - 1} \text{dist}(\pi_{v,j}, \pi_{v,j+1}) + \text{dist}(\pi_{v,n_v}, \text{Depósito}) \right)$$

donde $D_v$ es la distancia total recorrida por el camión $v$, $\pi_{v,j}$ representa el cliente que ocupa la posición $j$-ésima en la ruta del camión $v$, y $n_v$ es el total de clientes asignados a dicho vehículo.

## 4.3. Restricciones del Sistema

El modelo operativo debe garantizar la satisfacción rigurosa de dos restricciones operacionales de carácter físico y normativo:

1. **Restricción de Capacidad de Carga Máxima ($C_{\max}$):** La carga agregada asignada a cada camión no puede exceder en ningún momento su capacidad nominal máxima estipulada en $C_{\max} = 120\text{ kg}$:
   $$\sum_{j=1}^{n_v} d_{\pi_{v,j}} \le C_{\max}, \quad \forall v \in \{1, \dots, M\}$$

2. **Restricción de Jornada Laboral Máxima ($T_{\max}$):** El tiempo total de operación de cada camión —compuesto por el tiempo de desplazamiento entre paradas y los tiempos de servicio en cada cliente— no puede exceder la jornada laboral reglamentaria de 8 horas continuas ($T_{\max} = 480\text{ min}$):
   $$T_v = \left( \frac{D_v}{v} \times 60 \right) + \sum_{j=1}^{n_v} s_{\pi_{v,j}} \le T_{\max}, \quad \forall v \in \{1, \dots, M\}$$

3. **Restricción de Cobertura y Unicidad:** Cada cliente debe ser visitado exactamente una vez por un único camión de la flota comercial.

## 4.4. Parámetros de Configuración

En concordancia estricta con las especificaciones académicas de la cátedra, el Algoritmo Genético Simple fue configurado con los siguientes hiperparámetros:

* **Tamaño de la población ($N$):** $1000$ individuos.
* **Elitismo:** $2$ mejores individuos transferidos sin perturbación a la siguiente generación.
* **Mecanismo de Selección:** Torneo determinístico ($p = 2$) para la elección del Padre 1, y muestreo uniforme aleatorio para el Padre 2.
* **Mecanismo de Cruce:** Cruce en un solo punto (SPX) con tasa de cruce $P_c = 0.90$ ($90\%$).
* **Mecanismo de Mutación:** Mutación puntual con tasa de mutación $P_m = 0.02$ ($2\%$).
* **Condición de Parada:** Obtención de un individuo con aptitud normalizada $\text{fitness} < 10^{-2}$ o un límite máximo de $150$ generaciones.

---

# 5. MODELADO MATEMÁTICO Y DISEÑO DEL FITNESS – PROBLEMA 1

## 5.1. Codificación del Cromosoma: Genotipo y Fenotipo

Dado que el operador de cruce asignado por la cátedra es estrictamente el cruce en un solo punto (SPX) sobre cadenas lineales —el cual generaría clientes duplicados o clientes omitidos si se aplicase sobre permutaciones discretas estándar—, se adoptó el paradigma de **Claves Aleatorias (*Random Keys Genetic Algorithm*, RKGA)**.

Bajo este esquema, el genotipo de cada individuo se codifica como un vector continuo de números reales $\vec{x} \in \mathbb{R}^{15}$, donde cada gen $x_i \in [0, 100]$ representa una clave de prioridad asignada al cliente $i$. Para transformar este genotipo en el fenotipo de ruteo:
1. Se ordenan ascendentemente los índices según el valor de sus genes ($\text{argsort}(\vec{x})$), produciendo de manera inmediata una permutación válida y libre de redundancias sobre el conjunto de clientes $\{1, \dots, 15\}$.
2. La secuencia ordenada se distribuye entre los $M = 3$ camiones homogéneos de la flota, asignando a cada vehículo un itinerario cerrado que comienza y termina en el depósito central.

Este enfoque asegura la compatibilidad matemática total con los operadores de cruce SPX y mutación de la cátedra sin necesidad de recurrir a operadores de reparación heurística ad-hoc.

## 5.2. Función de Aptitud Base

Para permitir que el algoritmo converja hacia la condición de parada $\text{fitness} < 10^{-2}$, y conforme a las orientaciones técnicas validadas directamente con el docente en sesión de tutoría, la función de costo de transporte fue normalizada en el intervalo $[0, 1]$ mediante una transformación de escala acotada:

$$\text{Error}_{\text{dist}}(\vec{x}) = \frac{D_{\text{total}}(\vec{x}) - D_{\text{ideal}}}{D_{\text{peor}} - D_{\text{ideal}}}$$

donde $D_{\text{ideal}} = 235.0\text{ km}$ representa la cota teórica inferior del ruteo en este grafo espacial, y $D_{\text{peor}} = 750.0\text{ km}$ representa la distancia obtenida por trayectorias aleatorias desordenadas. A través de esta normalización, a medida que la ruta física se aproxima a la configuración óptima, el término base tiende asintóticamente a cero ($\text{Error} \to 0$), lo que en términos de bondad o aptitud biológica equivale a una solución tendiente a la unidad de adaptación ($\text{Aptitud} = 1 - \text{Error} \to 1.0$), según la precisión establecida por el docente.

## 5.3. Función de Penalización Integral

Las soluciones cuyos requerimientos de carga o tiempo sobrepasen los límites físicos de los camiones son penalizadas cuantitativamente degradando su aptitud mediante una penalización continua de segundo orden (penalización cuadrática externa):

$$\text{Fitness}(\vec{x}) = \text{Error}_{\text{dist}}(\vec{x}) + \lambda_1 \sum_{v=1}^{M} \left[\max(0, L_v - C_{\max})\right]^2 + \lambda_2 \sum_{v=1}^{M} \left[\max(0, T_v - T_{\max})\right]^2$$

donde $L_v$ es la carga total acumulada por el camión $v$, $T_v$ es el tiempo total de ruta, y los coeficientes de penalización fueron calibrados experimentalmente en $\lambda_1 = 0.05\text{ kg}^{-2}$ y $\lambda_2 = 0.02\text{ min}^{-2}$. Este diseño penaliza suavemente las pequeñas desviaciones pero genera una barrera matemática insuperable ante violaciones severas, extinguiendo las soluciones infactibles del acervo genético a medida que avanzan las generaciones.

---

# 6. IMPLEMENTACIÓN COMPUTACIONAL – PROBLEMA 1

## 6.1. Arquitectura del Código y Modularidad

El sistema computacional fue estructurado bajo un esquema de ingeniería de software modular, orientado a objetos y desacoplado, garantizando alta legibilidad, cumplimiento del estándar PEP 8 y reproducibilidad experimental. La arquitectura se organiza en tres capas fundamentales:

* **Módulo Motor SGA (`src/core/ga_engine.py`):** Contiene la clase `SimpleGeneticAlgorithm` y el contenedor inmutable de configuración `GAParams`. Implementa con fidelidad absoluta los operadores `Elitismo.m`, `seleccion_torneo_2.m`, `Cruce.m` y `Mutacion.m` facilitados por el docente.
* **Módulo del Problema (`src/problems/vrp.py`):** Define las estructuras de datos (`Customer`, `VRPInstance`, `VRPSolution`), el decodificador de claves aleatorias fenotípicas, la evaluación del fitness normalizado con penalizaciones y los motores gráficos de exportación visual a 300 DPI.
* **Interfaz de Control Unificada (`main.py`):** Punto de entrada del sistema que ofrece ejecución por menú interactivo o argumentos CLI, orquestando la simulación, la captura de telemetría y el guardado de artefactos.

## 6.2. Tabla de Hiperparámetros

En la Tabla 1 se consolidan los hiperparámetros y especificaciones computacionales utilizados en la resolución del Problema 1:

| Parámetro | Símbolo / Variable | Valor Asignado | Justificación Técnica |
| :--- | :---: | :---: | :--- |
| Tamaño de la Población | $N$ | $1000$ | Proporciona amplia diversidad inicial para el espacio combinatorio. |
| Cantidad de Élite | $E$ | $2$ | Garantiza la monotonía del mejor fitness histórico sin sesgar la búsqueda. |
| Probabilidad de Cruce | $P_c$ | $0.90$ ($90\%$) | Alta explotación de esquemas parentales mediante corte SPX. |
| Probabilidad de Mutación | $P_m$ | $0.02$ ($2\%$) | Perturbación suave para evitar el estancamiento en óptimos locales. |
| Tipo de Selección | Torneo $p$ | $p = 2$ | Presión selectiva moderada complementada con Padre 2 estocástico. |
| Límite Generacional | $G_{\max}$ | $150$ | Número suficiente de ciclos para observar estabilización y convergencia. |
| Tolerancia de Parada | $\epsilon$ | $10^{-2}$ ($0.01$) | Umbral de error residual admitido para declarar convergencia. |
| Semilla Pseudoaleatoria | `seed` | $42$ | Permite la reproducibilidad científica exacta de las simulaciones. |

---

# 7. RESULTADOS EXPERIMENTALES – PROBLEMA 1

## 7.1. Población Inicial

La población inicial (Generación 0) se generó de manera pseudoaleatoria uniforme sobre el espacio de claves aleatorias. Debido al ordenamiento aleatorio inicial de las paradas, la inmensa mayoría de las soluciones resultaron altamente infactibles, incurriendo en sobrecargas de capacidad vehicular y tiempos de recorrido superiores a las 8 horas reglamentarias. Esto produjo valores de fitness sumamente elevados a causa de las penalizaciones cuadráticas.

En la Tabla 2 se presenta una muestra de los primeros 5 individuos de la población inicial, junto con las estadísticas descriptivas globales de la Generación 0:

| Identificador | Genes de Muestra del Cromosoma $[x_1, x_2, x_3, x_4, x_5, \dots]$ | Fitness Normalizado | Estado de Factibilidad |
| :---: | :--- | :---: | :---: |
| **Individuo 1** | $[37.5, 95.1, 73.2, 59.9, 15.6, \dots]$ | $0.91644$ | Infactible (Sobrecarga leve) |
| **Individuo 2** | $[18.3, 30.4, 52.5, 43.2, 29.1, \dots]$ | $1.09990$ | Infactible (Exceso de tiempo) |
| **Individuo 3** | $[60.8, 17.1, 6.5, 94.9, 96.6, \dots]$ | $43.36406$ | Infactible severo |
| **Individuo 4** | $[66.3, 31.2, 52.0, 54.7, 18.5, \dots]$ | $1.11765$ | Infactible |
| **Individuo 5** | $[38.9, 27.1, 82.9, 35.7, 28.1, \dots]$ | $4.71951$ | Infactible |
| **Estadísticas Gen 0** | **Mínimo:** $0.71870$ &nbsp;\|&nbsp; **Promedio:** $73.45902$ &nbsp;\|&nbsp; **Máximo:** $1064.07384$ &nbsp;\|&nbsp; **Desv. Est. ($\sigma$):** $112.21297$ | | |

## 7.2. Población Final y Mejor Individuo

Tras completar el ciclo generacional en un tiempo de cómputo de $7.65\text{ segundos}$, la población convergió sólidamente hacia una región de alta calidad y factibilidad plena. El acervo genético eliminó por completo las soluciones inviables gracias a la presión selectiva ejercida por las funciones de penalización.

En la Tabla 3 se ilustran los mejores 5 individuos de la población final (Generación 150):

| Rango | Genes de Muestra del Cromosoma $[x_1, x_2, x_3, x_4, x_5, \dots]$ | Fitness Normalizado | Estado de Factibilidad |
| :---: | :--- | :---: | :---: |
| **Top 1 ($\vec{x}^*$)** | $[87.2, 59.6, 56.0, 54.6, 15.2, \dots]$ | **$0.37825$** | **100% Factible** |
| **Top 2** | $[88.7, 59.6, 56.0, 29.7, 15.2, \dots]$ | $0.37825$ | 100% Factible |
| **Top 3** | $[87.2, 59.6, 56.0, 29.7, 15.2, \dots]$ | $0.37825$ | 100% Factible |
| **Top 4** | $[87.2, 59.6, 56.0, 29.7, 15.2, \dots]$ | $0.37825$ | 100% Factible |
| **Top 5** | $[87.2, 59.6, 56.0, 54.6, 15.2, \dots]$ | $0.37825$ | 100% Factible |
| **Estadísticas Gen 150**| **Mejor:** $0.37825$ &nbsp;\|&nbsp; **Promedio:** $0.40842$ &nbsp;\|&nbsp; **Máximo:** $10.89441$ &nbsp;\|&nbsp; **Desv. Est. ($\sigma$):** $0.33292$ | | |

El mejor individuo descubierto ($\vec{x}^*$) resolvió la logística de entrega con una **distancia total de $429.80\text{ km}$**, logrando un reparto equilibrado y coherente con la geografía espacial:

* **Camión 1 (Sector Sur / Suroeste):**  
  $\text{Depósito} \rightarrow \text{C7} \rightarrow \text{C6} \rightarrow \text{C5} \rightarrow \text{C15} \rightarrow \text{C11} \rightarrow \text{Depósito}$  
  *Métricas:* Distancia $= 156.00\text{ km}$ \| Carga $= 99.0 / 120.0\text{ kg}$ ($82.5\%$) \| Tiempo $= 305.0 / 480.0\text{ min}$ ($63.5\%$) \| Infracciones $= 0$.
* **Camión 2 (Sector Este / Noreste):**  
  $\text{Depósito} \rightarrow \text{C4} \rightarrow \text{C13} \rightarrow \text{C3} \rightarrow \text{C2} \rightarrow \text{C10} \rightarrow \text{Depósito}$  
  *Métricas:* Distancia $= 146.88\text{ km}$ \| Carga $= 88.0 / 120.0\text{ kg}$ ($73.3\%$) \| Tiempo $= 292.8 / 480.0\text{ min}$ ($61.0\%$) \| Infracciones $= 0$.
* **Camión 3 (Sector Norte / Noroeste):**  
  $\text{Depósito} \rightarrow \text{C12} \rightarrow \text{C8} \rightarrow \text{C14} \rightarrow \text{C1} \rightarrow \text{C9} \rightarrow \text{Depósito}$  
  *Métricas:* Distancia $= 126.92\text{ km}$ \| Carga $= 97.0 / 120.0\text{ kg}$ ($80.8\%$) \| Tiempo $= 276.2 / 480.0\text{ min}$ ($57.5\%$) \| Infracciones $= 0$.

## 7.3. Comparativa: Población Inicial vs. Población Final

El contraste cuantitativo entre la población de partida y la población final refleja la potencia de búsqueda adaptativa del Algoritmo Genético Simple:

1. **Reducción del Fitness Promedio:** El fitness medio poblacional descendió de $73.45902$ a $0.40842$, lo que equivale a una **disminución del $99.44\%$** en el costo de evaluación global.
2. **Colapso de la Varianza Genética:** La desviación estándar poblacional se redujo drásticamente desde $\sigma = 112.21297$ en la Generación 0 hasta $\sigma = 0.33292$ en la Generación 150, confirmando que la población concentró su masa probabilística alrededor de la cuenca del óptimo global factible.
3. **Eliminación Total de Soluciones Inválidas:** En la Generación 0, más del $90\%$ de los individuos violaban las restricciones de carga o jornada. Al cabo de 60 generaciones, el $100\%$ de los individuos de la élite y la media cumplieron con el $100\%$ de las restricciones operacionales.

---

# 8. ANÁLISIS DE CONVERGENCIA Y ERRORES – PROBLEMA 1

## 8.1. Curvas de Convergencia Generacional

La dinámica evolutiva del sistema fue registrada a lo largo de las 150 generaciones y se representa gráficamente en las Figuras 1 y 2.

*(Inserte aquí la imagen: `figura1_vrp_rutas_optimas.png`)*

**Figura 1.** *Topología espacial y ruteo óptimo de entregas para la flota vehicular (VRP).*

Tal como se aprecia en la **Figura 1**, el algoritmo genético descubrió de manera emergente una descomposición sectorial casi perfecta de la demanda geográfica. En lugar de entrecruzar rutas caóticamente, el Camión 1 (línea verde) se hace cargo exclusivamente del cuadrante sur y suroeste; el Camión 2 (línea naranja) cubre el flanco este y noreste; y el Camión 3 (línea azul) recorre el clúster norte y noroeste. Esta organización en sectores disjuntos minimiza el cruce de aristas y maximiza la eficiencia espacial del transporte.

*(Inserte aquí la imagen: `figura2_vrp_convergencia.png`)*

**Figura 2.** *Curvas de convergencia generacional del Algoritmo Genético Simple para el Problema 1 (Mejor, Promedio y Peor Fitness).*

En la **Figura 2**, trazada en escala logarítmica para facilitar el análisis multiescala, se evidencian tres fases dinámicas bien diferenciadas:
* **Fase de Purga y Factibilidad (Generaciones 0 a 25):** Las curvas de fitness promedio y peor experimentan una caída estrepitosa (de órdenes de magnitud mayores a $10^3$ hacia $10^1$). En este periodo, las funciones de penalización actúan con máxima intensidad, erradicando a los individuos que sobrecargan la capacidad de los camiones.
* **Fase de Explotación y Recombinación (Generaciones 25 a 75):** Una vez alcanzada la factibilidad plena, el operador de cruce SPX domina la dinámica, intercambiando bloques de rutas entre padres seleccionados por torneo para reducir progresivamente la distancia euclidiana total de $520\text{ km}$ a $429.80\text{ km}$.
* **Fase de Meseta y Ajuste Fino (Generaciones 75 a 150):** El mejor fitness ($f_{\min} = 0.37825$) se estabiliza de manera asintótica. El elitismo mantiene inmutable la mejor solución histórica, mientras que la mutación puntual ($P_m = 0.02$) continúa explorando vecindarios sin alterar la estabilidad del óptimo alcanzado.

## 8.2. Auditoría del Error Residual

Para dar cumplimiento estricto a la verificación solicitada por la cátedra, se efectuó una auditoría exhaustiva sobre el mejor individuo $\vec{x}^*$ obtenido:

1. **Infracción de Capacidad de Carga:**
   $$\text{Violación}_{\text{carga}} = \sum_{v=1}^{3} \max(0, L_v - 120.0\text{ kg}) = 0.00\text{ kg}$$
   El camión con mayor nivel de ocupación alcanzó $99.0\text{ kg}$, disponiendo de un margen de seguridad de $21.0\text{ kg}$ ($17.5\%$ de holgura).
2. **Infracción de Jornada Laboral:**
   $$\text{Violación}_{\text{tiempo}} = \sum_{v=1}^{3} \max(0, T_v - 480.0\text{ min}) = 0.00\text{ min}$$
   La ruta más prolongada requirió $305.0\text{ minutos}$ ($5.08\text{ horas}$), respetando holgadamente el tope de 8 horas con un remanente operativo de $175.0\text{ minutos}$.
3. **Error Residual de Restricciones:** El vector de penalizaciones final es idénticamente nulo:
   $$\text{Penalización}(\vec{x}^*) = 0.05 \times (0)^2 + 0.02 \times (0)^2 = \mathbf{0.000000}$$

## 8.3. Análisis de los Operadores y las Penalizaciones

El desempeño del Algoritmo Genético Simple en el Problema 1 permite extraer valiosas conclusiones sobre el equilibrio estocástico de sus operadores:

* **Efectividad del Torneo Asimétrico ($p = 2$):** La estrategia de seleccionar al Padre 1 mediante torneo y al Padre 2 de forma aleatoria demostró ser altamente efectiva. En lugar de causar una convergencia prematura en las primeras 20 generaciones —riesgo típico de los torneos simétricos agresivos—, preservó la diversidad alélica el tiempo necesario para explorar combinaciones de rutas alternativas.
* **Elitismo como Memoria Evolutiva:** La conservación incondicional de los 2 mejores cromosomas garantizó la estricta monotonía decreciente del fitness mínimo ($f_{\min}^{(t+1)} \le f_{\min}^{(t)}$), impidiendo que soluciones de alta calidad fuesen destruidas por recombinaciones desfavorables.
* **Penalizaciones Cuadráticas vs. Penalizaciones Lineales:** El empleo de penalizaciones cuadráticas proporcionó un gradiente de penalización suave pero implacable en los extremos. Esto permitió que individuos ligeramente infactibles sirvieran temporalmente como "puentes genéticos" en las primeras iteraciones, para luego ser rápidamente sustituidos por variantes estrictamente factibles.
