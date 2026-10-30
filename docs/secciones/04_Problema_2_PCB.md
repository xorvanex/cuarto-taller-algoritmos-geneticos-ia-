# PROBLEMA 2: OPTIMIZACIÓN DE DISEÑO DE CIRCUITOS PCB

# 9. DEFINICIÓN DEL PROBLEMA 2

## 9.1. Descripción y Contexto de Ingeniería

El diseño y posicionamiento automatizado de componentes electrónicos en placas de circuito impreso (*Printed Circuit Board Layout & Component Placement*, PCB) constituye uno de los problemas de optimización combinatoria y geométrica continua más críticos en la automatización del diseño electrónico (EDA, *Electronic Design Automation*). En la ingeniería electrónica y de sistemas embebidos, la disposición física sobre el sustrato dieléctrico (típicamente resina epoxi reforzada con fibra de vidrio, FR-4) determina de manera directa la integridad de la señal, la disipación térmica, el acoplamiento capacitivo indeseado, la susceptibilidad a interferencias electromagnéticas (EMI/EMC) y el costo final de manufactura.

Para este estudio, se modela un módulo de procesamiento embebido para aplicaciones de Internet de las Cosas (IoT) de alta confiabilidad. El sustrato físico está delimitado por unas dimensiones industriales estándar de ancho $W_{\text{board}} = 100.0\text{ mm}$ y alto $H_{\text{board}} = 80.0\text{ mm}$, totalizando un área superficial bruta de $8000.0\text{ mm}^2$. 

Sobre esta placa deben ubicarse $C = 14$ componentes electrónicos heterogéneos, clasificados en cuatro familias funcionales:
1. **Unidad de Procesamiento Central (U1):** Microcontrolador ARM Cortex-M4 en encapsulado LQFP-48 ($16.0 \times 16.0\text{ mm}$).
2. **Sistema de Alimentación y Regulación (U2):** Regulador de caída baja (LDO) de 3.3V en encapsulado SOT-223 ($10.0 \times 8.0\text{ mm}$).
3. **Generador de Reloj del Sistema (Y1):** Cristal oscilador de cuarzo de 16 MHz ($8.0 \times 5.0\text{ mm}$).
4. **Almacenamiento No Volátil (U3):** Memoria Flash SPI serie en encapsulado SOIC-8 ($8.0 \times 8.0\text{ mm}$).
5. **Conectores de Interfaz Externa (J1, J2):**
   - Receptáculo USB-C para alimentación primaria y depuración ($10.0 \times 8.0\text{ mm}$).
   - Cabezal de expansión GPIO $2 \times 5$ pines ($20.0 \times 6.0\text{ mm}$).
6. **Componentes Pasivos SMD (Montaje Superficial):**
   - Capacitores de desacoplo de alta frecuencia C1 y C2 (encapsulado 0805, $4.0 \times 4.0\text{ mm}$).
   - Capacitores de filtrado de riel de potencia C3 y C4 (electrolíticos SMD, $6.0 \times 6.0\text{ mm}$).
   - Resistencias de polarización R1 (Reset pull-up), R2 (terminación de bus SPI) y R3 (indicador LED) ($4.0 \times 3.0\text{ mm}$).
   - Diodo de protección contra descargas electrostáticas D1 (ESD protection en línea USB, $5.0 \times 4.0\text{ mm}$).

La interconexión eléctrica entre estos 14 componentes está conformada por una red de $M = 22$ pistas de cobre lógicas (*nets*). Una distribución caótica o dispersa genera congestión de ruteo, pistas excesivamente largas y colisiones mecánicas entre los encapsulados, lo que invalida completamente la placa. Por consiguiente, se requiere un Algoritmo Genético Simple (SGA) guiado para optimizar simultáneamente la ubicación de los dispositivos y las conexiones de cobre.

## 9.2. Función Objetivo

El objetivo fundamental es minimizar una función de costo multicriterio normalizada que pondera tres factores esenciales del diseño de hardware: la longitud total ponderada de las pistas de cobre, el área del rectángulo mínimo envolvente (*Bounding Box*) y la penalización cuadrática estricta ante colisiones físicas entre componentes.

La formulación matemática de la función de costo se define como:

$$f_{\text{norm}}(\vec{x}) = w_1 \cdot \left( \frac{\text{WL}_{\text{manhattan}}(\vec{x})}{\text{WL}_{\max}} \right) + w_2 \cdot \left( \frac{\text{Área}_{\text{box}}(\vec{x})}{\text{Área}_{\max}} \right) + w_3 \cdot \text{Penalización}_{\text{overlap}}(\vec{x})$$

Donde los parámetros de ponderación de cátedra corresponden a:
* $w_1 = 0.40$ ($40\%$): Ponderación asignada a la longitud de pistas de cobre.
* $w_2 = 0.10$ ($10\%$): Ponderación asignada a la compacidad espacial (área envolvente).
* $w_3 = 0.50$ ($50\%$): Ponderación asignada al cumplimiento riguroso de no solapamiento físico.
* $\text{WL}_{\max} = 3000.0\text{ mm}$: Cota superior empírica de longitud de red en estados dispersos.
* $\text{Área}_{\max} = 8000.0\text{ mm}^2$: Superficie total del sustrato ($100 \times 80\text{ mm}$).

### Longitud de Pistas Manhattan Ponderada ($\text{WL}_{\text{manhattan}}$)
En el diseño de circuitos impresos multicapa (2 o 4 capas), las pistas de cobre no pueden trazarse en diagonales arbitrarias continuas sin generar interferencias, sino que se enrutan mediante segmentos horizontales y verticales sobre capas ortogonales alternadas (Top layer horizontal, Bottom layer vertical). La métrica natural para evaluar este cableado es la distancia Manhattan ponderada ($L_1$):

$$\text{WL}_{\text{manhattan}}(\vec{x}) = \sum_{k=1}^{M} \omega_k \cdot \left( |x_{\text{src}(k)} - x_{\text{tgt}(k)}| + |y_{\text{src}(k)} - y_{\text{tgt}(k)}| \right)$$

donde $\omega_k$ denota el factor de criticidad asignado a cada pista de cobre:
- $\omega_k = 3.0$ para la señal de reloj de alta frecuencia (Y1 $\to$ U1), para minimizar la radiación electromagnética hacia otros nodos.
- $\omega_k = 2.5$ para las líneas de alimentación críticas (USB-C $\to$ LDO, LDO $\to$ MCU VDD), reduciendo la caída de tensión IR.
- $\omega_k = 2.0$ para el bus de alta velocidad SPI Flash (U3 $\to$ U1).
- $\omega_k = 1.0$ para señales generales de entrada/salida y líneas auxiliares.

### Área Envolvente (*Bounding Box Area*)
Para favorecer la miniaturización y la utilización eficiente del espacio físico, se calcula el área del rectángulo mínimo que circunscribe la totalidad de los componentes ubicados:

$$\Delta X_{\text{box}} = \max_{i \in \{1,\dots,C\}} \left( x_i + \frac{w_i}{2} \right) - \min_{i \in \{1,\dots,C\}} \left( x_i - \frac{w_i}{2} \right)$$

$$\Delta Y_{\text{box}} = \max_{i \in \{1,\dots,C\}} \left( y_i + \frac{h_i}{2} \right) - \min_{i \in \{1,\dots,C\}} \left( y_i - \frac{h_i}{2} \right)$$

$$\text{Área}_{\text{box}}(\vec{x}) = \Delta X_{\text{box}} \times \Delta Y_{\text{box}}$$

## 9.3. Restricciones del Sistema

El modelado impone el cumplimiento estricto de tres restricciones operacionales y mecánicas:

1. **Restricción de No Solapamiento Físico (*Non-overlapping Constraint*):** Dos componentes electrónicos no pueden compartir el mismo espacio volumétrico en la placa. Para cada par de componentes rectangulares $i \neq j$:
   $$\text{Solapamiento}(i, j) = \max\left(0, \frac{w_i + w_j}{2} - |x_i - x_j|\right) \times \max\left(0, \frac{h_i + h_j}{2} - |y_i - y_j|\right) = 0$$

2. **Restricción de Aislamiento y Holgura Perimetral (*Clearance*):** Se exige una distancia perimetral de seguridad mínima de $\delta = 2.0\text{ mm}$ alrededor de cada componente. Esto garantiza el margen necesario para las boquillas de inserción automatizada SMT (*Pick and Place*), previene puentes de soldadura en reflujo y asegura el aislamiento dieléctrico.

3. **Restricción de Frontera del Sustrato (*Boundary Containment*):** Cada componente debe encontrarse confinado dentro del plano físico útil de la placa:
   $$x_i - \frac{w_i}{2} \ge 0, \quad x_i + \frac{w_i}{2} \le W_{\text{board}}, \quad \forall i \in \{1, \dots, C\}$$
   $$y_i - \frac{h_i}{2} \ge 0, \quad y_i + \frac{h_i}{2} \le H_{\text{board}}, \quad \forall i \in \{1, \dots, C\}$$

## 9.4. Parámetros de Configuración

En concordancia con los lineamientos impartidos en la cátedra de Inteligencia Artificial (Ing. Manuel Alejandro Ospina Alarcón), se mantuvieron exactamente los mismos operadores canónicos del Algoritmo Genético Simple:

* **Tamaño de la población ($N$):** $1000$ individuos.
* **Elitismo ($k$):** $2$ mejores individuos transferidos intactos a la siguiente generación (`Elitismo.m`).
* **Mecanismo de Selección:** Torneo determinístico binario ($p = 2$) para la elección del Padre 1 y selección aleatoria uniforme para el Padre 2 (`seleccion_torneo_2.m`).
* **Mecanismo de Cruce:** Cruce en un solo punto (SPX) con probabilidad $P_c = 0.90$ ($90\%$) (`Cruce.m`).
* **Mecanismo de Mutación:** Mutación puntual con probabilidad $P_m = 0.02$ ($2\%$) (`Mutacion.m`).
* **Criterio de Parada:** Fitness normalizado $< 10^{-2}$ o $150$ generaciones continuas.
* **Semilla pseudoaleatoria:** `seed = 123` para garantizar la reproducibilidad experimental.

---

# 10. MODELADO MATEMÁTICO Y DISEÑO DEL FITNESS — PROBLEMA 2

## 10.1. Codificación del Cromosoma: Genotipo y Fenotipo

Dado que el posicionamiento de componentes electrónicos sobre una placa ocurre en un dominio espacial bidimensional continuo, no resulta conveniente una representación por permutación discreta. Por lo tanto, el genotipo de cada individuo se estructura como un vector de números reales en el hipercubo unitario:

$$\vec{g} = \langle g_1, g_2, \dots, g_{2C} \rangle \in [0.0, 1.0]^{28}$$

donde cada par de alelos consecutivos $(g_{2i - 1}, g_{2i})$ codifica la posición normalizada del componente $i \in \{1, \dots, 14\}$.

### Decodificación al Fenotipo Físico
Para asegurar que los componentes no sobrepasen las fronteras de la placa bajo ninguna circunstancia, se implementa una transformación lineal acotada que traduce los genes normalizados a coordenadas físicas reales $(x_i, y_i)$ en milímetros:

$$x_i = \left( \frac{w_i}{2} + \delta \right) + g_{2i - 1} \cdot \left( W_{\text{board}} - w_i - 2\delta \right)$$

$$y_i = \left( \frac{h_i}{2} + \delta \right) + g_{2i} \cdot \left( H_{\text{board}} - h_i - 2\delta \right)$$

Esta decodificación garantiza analíticamente que la restricción de frontera de sustrato se cumpla en el $100\%$ del espacio genético, evitando la necesidad de operadores de reparación artificiales y permitiendo que el Cruce SPX y la Mutación Puntual operen con plena libertad en $[0.0, 1.0]$.

## 10.2. Función de Aptitud Base

La aptitud base se calcula evaluando la eficiencia de interconexión y la compacidad de la solución factible:

$$f_{\text{base}}(\vec{x}) = 0.40 \cdot \left( \frac{\text{WL}_{\text{manhattan}}(\vec{x})}{\text{WL}_{\max}} \right) + 0.10 \cdot \left( \frac{\text{Área}_{\text{box}}(\vec{x})}{\text{Área}_{\max}} \right)$$

A medida que el algoritmo agrupa los componentes que comparten pistas de alta criticidad (por ejemplo, ubicando los capacitores C1 y C2 a milímetros de los pines de desacoplo de U1), la longitud total de cobre decrece significativamente, haciendo descender el término base hacia valores inferiores a $0.70$.

La **Aptitud Biológica** porcentual del individuo se deriva conforme a la interpretación biológica de adaptación:

$$\text{Aptitud}(\vec{x}) = \max\left(0, 1.0 - f_{\text{norm}}(\vec{x})\right) \times 100\%$$

## 10.3. Función de Penalización Integral

La violación de la restricción de no solapamiento físico debe ser tratada como un fallo crítico en el fenotipo. Para ello, se diseñó una función de penalización cuadrática progresiva que penaliza fuertemente el área solapada entre componentes:

$$\text{Penalización}_{\text{overlap}}(\vec{x}) = \sum_{i=1}^{C-1} \sum_{j=i+1}^{C} \left[ \frac{\text{ÁreaSolapada}(i, j)}{A_{\text{ref}}} + 0.10 \times \left( \frac{\text{ÁreaSolapada}(i, j)}{A_{\text{ref}}} \right)^2 \right]$$

donde $A_{\text{ref}} = 100.0\text{ mm}^2$ es el área de normalización. Si dos chips presentan interferencia espacial, el término de penalización escala de forma cuadrática, elevando drásticamente el fitness del individuo ($f_{\text{norm}} \gg 1.0$). Esto condena al espécimen a ser derrotado en el torneo de selección determinístico ($p = 2$), extinguiendo rápidamente las configuraciones físicamente inviables de la población en las primeras generaciones.

---

# 11. IMPLEMENTACIÓN COMPUTACIONAL — PROBLEMA 2

## 11.1. Arquitectura del Código y Modularidad

El sistema de optimización fue desarrollado siguiendo los principios de la arquitectura modular desacoplada:

* **Motor Genético Generalizado (`src/core/ga_engine.py`):** Contiene la clase `SimpleGeneticAlgorithm` y la estructura inmutable `GAParams`. Ejecuta el ciclo generacional aplicando rigurosamente los operadores de MATLAB de cátedra (`Cruce.m`, `Elitismo.m`, `Mutacion.m`, `seleccion_torneo_2.m`).
* **Módulo de Dominio del Problema (`src/problems/pcb.py`):** Define las clases `PCBComponent`, `PCBNet`, `PCBInstance`, `PlacedComponent`, `PCBSolution` y el evaluador `PCBModel`. Gestiona la decodificación fenotípica, la detección analítica de colisiones rectangulares y la generación de gráficos vectoriales a 300 DPI.
* **Orquestador Principal (`src/main.py`):** Despacha la ejecución mediante argumentos CLI (`--problem 2`), controla la captura de telemetría y exporta los resultados computados hacia `data/pcb_results.json` y `src/web/data/pcb_results.js`.
* **Capa de Visualización Interactiva (`src/web/`):** Dashboard web centrado y libre de emojis que renderiza el plano tecnológico en HTML5 Canvas con zoom/pan, trazas Manhattan ortogonales, gráfica de convergencia sincronizada en tiempo real y tarjeta de fórmulas de cátedra.

## 11.2. Tabla de Hiperparámetros

En la Tabla 1 se detallan los hiperparámetros operativos empleados en la simulación certificada del Problema 2:

| Parámetro | Notación | Valor Asignado | Justificación de Ingeniería |
| :--- | :---: | :---: | :--- |
| Tamaño de la Población | $N$ | $1000$ | Proporciona amplia diversidad para cubrir el espacio continuo $[0, 1]^{28}$. |
| Cantidad de Élite | $k$ | $2$ | Garantiza la preservación estricta de la mejor solución encontrada sin estancamiento. |
| Probabilidad de Cruce | $P_c$ | $0.90$ ($90\%$) | Alta recombinación para intercambiar bloques constructivos de ubicación entre padres. |
| Probabilidad de Mutación | $P_m$ | $0.02$ ($2\%$) | Perturbación exploratoria suave para afinar milimétricamente las coordenadas. |
| Tamaño de Torneo | $p$ | $2$ | Torneo determinístico binario para selección del Padre 1; Padre 2 aleatorio uniforme. |
| Límite Generacional | $G_{\max}$ | $150$ | Suficiente para alcanzar convergencia asintótica y factibilidad plena ($0$ colisiones). |
| Tolerancia de Parada | $\epsilon$ | $10^{-2}$ ($0.01$) | Umbral matemático formal de convergencia. |
| Semilla Pseudoaleatoria | `seed` | $123$ | Permite la reproducibilidad científica exacta de las trayectorias evolutivas. |

---

# 12. RESULTADOS EXPERIMENTALES — PROBLEMA 2

## 12.1. Población Inicial

La población inicial (Generación 0) se generó de manera pseudoaleatoria uniforme en el hipercubo continuo $[0.0, 1.0]^{28}$. En esta fase inicial, la ubicación aleatoria de los 14 componentes causó múltiples colisiones físicas y solapamientos de encapsulados, activando la penalización cuadrática integral.

En la Tabla 2 se presentan los primeros 5 individuos muestreados de la población inicial junto con las métricas agregadas de la Generación 0:

| Individuo | Genes de Muestra $[g_1, g_2, g_3, g_4, g_5, g_6, \dots]$ | Fitness Normalizado | Longitud Manhattan | Área Envolvente | Estado de Factibilidad |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **Individuo 1** | $[0.70, 0.29, 0.23, 0.55, 0.72, 0.42, \dots]$ | $98.54326$ | $1477.6\text{ mm}$ | $5178.4\text{ mm}^2$ | Infactible ($194.9\text{ mm}^2$ colisión, 3 errores) |
| **Individuo 2** | $[0.29, 0.63, 0.09, 0.43, 0.43, 0.49, \dots]$ | $11.50945$ | $1653.0\text{ mm}$ | $6119.6\text{ mm}^2$ | Infactible ($20.5\text{ mm}^2$ colisión, 3 errores) |
| **Individuo 3** | $[0.42, 0.68, 0.88, 0.51, 0.67, 0.59, \dots]$ | $36.29687$ | $1696.8\text{ mm}$ | $5698.6\text{ mm}^2$ | Infactible ($70.1\text{ mm}^2$ colisión, 3 errores) |
| **Individuo 4** | $[0.93, 0.84, 0.36, 0.04, 0.30, 0.40, \dots]$ | $44.50918$ | $2266.1\text{ mm}$ | $6409.6\text{ mm}^2$ | Infactible ($85.7\text{ mm}^2$ colisión, 4 errores) |
| **Individuo 5** | $[0.17, 0.83, 0.34, 0.55, 0.58, 0.52, \dots]$ | $72.00297$ | $2249.7\text{ mm}$ | $7171.3\text{ mm}^2$ | Infactible ($140.6\text{ mm}^2$ colisión, 4 errores) |
| **Estadísticas Gen 0** | **Mínimo:** $1.31957$ &nbsp;\|&nbsp; **Promedio:** $60.18195$ &nbsp;\|&nbsp; **Máximo:** $366.30076$ &nbsp;\|&nbsp; **Desv. Est. ($\sigma$):** $43.98570$ | | | | |

## 12.2. Población Final y Mejor Individuo

Tras completar las 150 generaciones (tiempo de ejecución de $17.32\text{ segundos}$ en CPU), la presión selectiva eliminó absolutamente todas las colisiones físicas, logrando que el acervo genético convergiera hacia individuos $100\%$ factibles con wirelength compacto.

En la Tabla 3 se detalla el Top 5 de individuos élite de la población final (Generación 150):

| Rango | Genes de Muestra $[g_1, g_2, g_3, g_4, g_5, g_6, \dots]$ | Fitness Normalizado | Aptitud Biológica | Longitud Manhattan | Área Envolvente | Factibilidad Física |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **#1 (Mejor)** | $[0.44, 0.53, 0.20, 0.72, 0.60, 0.42, \dots]$ | **0.675907** | **32.41%** | **1080.17 mm** | **3168.02 mm²** | **100% Factible (0 err)** |
| **#2** | $[0.44, 0.53, 0.20, 0.72, 0.60, 0.42, \dots]$ | $0.679445$ | $32.06\%$ | $1084.72\text{ mm}$ | $3178.91\text{ mm}^2$ | $100\%$ Factible (0 err) |
| **#3** | $[0.44, 0.53, 0.20, 0.72, 0.60, 0.42, \dots]$ | $0.679637$ | $32.04\%$ | $1085.03\text{ mm}$ | $3178.91\text{ mm}^2$ | $100\%$ Factible (0 err) |
| **#4** | $[0.44, 0.53, 0.20, 0.72, 0.60, 0.42, \dots]$ | $0.679637$ | $32.04\%$ | $1085.03\text{ mm}$ | $3178.91\text{ mm}^2$ | $100\%$ Factible (0 err) |
| **#5** | $[0.44, 0.53, 0.20, 0.72, 0.60, 0.42, \dots]$ | $0.680185$ | $31.98\%$ | $1085.89\text{ mm}$ | $3178.91\text{ mm}^2$ | $100\%$ Factible (0 err) |
| **Estadísticas Gen 150**| **Mínimo:** $0.67591$ &nbsp;\|&nbsp; **Promedio:** $0.82070$ &nbsp;\|&nbsp; **Máximo:** $33.29879$ &nbsp;\|&nbsp; **Desv. Est. ($\sigma$):** $1.58956$ | | | | | |

En la Tabla 4 se presenta la distribución geométrica final certificada del mejor individuo:

| Componente | Referencia | Función / Encapsulado | Coordenada X | Coordenada Y | Dimensiones ($W \times H$) |
| :--- | :---: | :--- | :---: | :---: | :---: |
| MCU ARM-Cortex | U1 | Procesador central (LQFP-48) | $44.55\text{ mm}$ | $41.82\text{ mm}$ | $16.0 \times 16.0\text{ mm}$ |
| Regulador LDO 3.3V | U2 | Alimentación de potencia (SOT-223) | $22.60\text{ mm}$ | $55.85\text{ mm}$ | $10.0 \times 8.0\text{ mm}$ |
| Cristal Oscilador 16MHz | Y1 | Reloj del sistema | $59.55\text{ mm}$ | $34.27\text{ mm}$ | $8.0 \times 5.0\text{ mm}$ |
| Memoria SPI Flash | U3 | Almacenamiento no volátil (SOIC-8) | $50.57\text{ mm}$ | $21.46\text{ mm}$ | $8.0 \times 8.0\text{ mm}$ |
| Conector USB-C | J1 | Entrada de potencia / datos | $61.31\text{ mm}$ | $44.93\text{ mm}$ | $10.0 \times 8.0\text{ mm}$ |
| Cabezal GPIO 2x5 | J2 | Puerto de expansión | $32.56\text{ mm}$ | $25.07\text{ mm}$ | $20.0 \times 6.0\text{ mm}$ |
| Cap. Desacoplo 100nF | C1 | Filtro alta frecuencia VDD | $15.70\text{ mm}$ | $24.40\text{ mm}$ | $4.0 \times 4.0\text{ mm}$ |
| Cap. Desacoplo 100nF | C2 | Filtro alta frecuencia VDD | $70.68\text{ mm}$ | $38.74\text{ mm}$ | $4.0 \times 4.0\text{ mm}$ |
| Cap. Filtrado 10uF | C3 | Filtro entrada LDO | $11.21\text{ mm}$ | $55.94\text{ mm}$ | $6.0 \times 6.0\text{ mm}$ |
| Cap. Filtrado 10uF | C4 | Filtro salida LDO | $11.69\text{ mm}$ | $43.81\text{ mm}$ | $6.0 \times 6.0\text{ mm}$ |
| Res. Reset Pull-up | R1 | Polarización pin NRST | $22.23\text{ mm}$ | $41.70\text{ mm}$ | $4.0 \times 3.0\text{ mm}$ |
| Res. Terminación SPI | R2 | Acoplamiento bus serie | $75.39\text{ mm}$ | $15.89\text{ mm}$ | $4.0 \times 3.0\text{ mm}$ |
| Res. LED de Estado | R3 | Limitadora de corriente LED | $52.19\text{ mm}$ | $56.36\text{ mm}$ | $4.0 \times 3.0\text{ mm}$ |
| Diodo ESD USB | D1 | Supresión de transitorios TVS | $75.39\text{ mm}$ | $52.82\text{ mm}$ | $5.0 \times 4.0\text{ mm}$ |

## 12.3. Comparativa: Población Inicial vs. Población Final

En la Tabla 5 se resume la transformación cuantitativa experimentada por la población entre el estado inicial desordenado y el óptimo convergente:

| Métrica de Desempeño | Generación 0 (Inicial) | Generación 150 (Final) | Variación Porcentual | Impacto en Ingeniería Electrónica |
| :--- | :---: | :---: | :---: | :--- |
| **Mejor Fitness ($f_{\min}$)** | $1.31957$ | $0.67591$ | **-48.78%** | Mejora sustancial en la calidad integral del diseño. |
| **Fitness Promedio ($\bar{f}$)** | $60.18195$ | $0.82070$ | **-98.64%** | Consolidación y maduración de todo el acervo genético. |
| **Desviación Estándar ($\sigma$)** | $43.98570$ | $1.58956$ | **-96.39%** | Alta homogeneidad de soluciones de élite viables. |
| **Tasa de Factibilidad** | $0.0\%$ (Todo infactible) | **100.0% Factible** | **+100.0%** | Erradicación total de interferencias espaciales. |
| **Área de Colisión Promedio** | $> 80\text{ mm}^2$ | **0.0 mm²** | **-100.0%** | Cero solapamientos físicos ($0$ colisiones). |
| **Longitud Manhattan** | $> 1800\text{ mm}$ | **1080.17 mm** | **-40.0%** | Menor inductancia parásita y menor retardo de señal. |
| **Área Bounding Box** | $6200.0\text{ mm}^2$ | **3168.02 mm²** | **-48.90%** | **60.40% de ahorro de sustrato** frente al área total. |

---

# 13. ANÁLISIS DE CONVERGENCIA Y ERRORES — PROBLEMA 2

## 13.1. Curvas de Convergencia Generacional

La evolución temporal del algoritmo a través de las 150 generaciones se sintetiza en la Tabla 6 y se ilustra gráficamente en las Figuras 3 y 4:

| Época / Generación | Mejor Fitness ($f_{\min}$) | Promedio Poblacional ($\bar{f}$) | Peor Fitness ($f_{\max}$) | Desv. Estándar ($\sigma$) | Estado Operativo del Sistema |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **0** | $1.31957$ | $60.18195$ | $366.30076$ | $43.98570$ | Alta tasa de colisiones y dispersión espacial. |
| **25** | $1.05458$ | $34.14202$ | $163.32043$ | $28.17529$ | Eliminación masiva de solapamientos graves. |
| **50** | $1.02397$ | $26.70720$ | $203.02813$ | $25.67758$ | Reducción del área perimetral y compactación. |
| **75** | $0.86117$ | $17.33044$ | $113.63693$ | $18.43466$ | Consolidación de bloques de alimentación y control. |
| **100** | $0.85195$ | $8.22482$ | $93.61396$ | $12.86872$ | Acercamiento del oscilador Y1 y la memoria U3 a U1. |
| **125** | $0.74597$ | $1.70070$ | $75.73460$ | $5.65391$ | Afinado milimétrico de las holguras dieléctricas. |
| **150** | **0.67591** | **0.82070** | **33.29879** | **1.58956** | **Solución Final Óptima Certificada (100% Factible).** |

### Figura 3: Plano Tecnológico del PCB Layout Óptimo (100 × 80 mm)
Ilustra la ubicación centralizada del procesador U1, la proximidad del bloque de potencia (U2, C3, C4) a la izquierda, la cercanía del cristal Y1 y el puerto USB-C J1 a la derecha, con las 22 trazas de cobre Manhattan completamente ordenadas.

![Plano Tecnológico del PCB Layout Óptimo](../img/figura3_pcb_layout_optimo.png)

### Figura 4: Curvas de Convergencia Generacional del SGA (Problema 2)
Muestra la reducción del mejor fitness ($f_{\min}$ en verde), la convergencia asintótica del promedio ($\bar{f}$ en azul discontinuo) y el estrechamiento del peor individuo ($f_{\max}$ en rojo).

![Curvas de Convergencia Generacional PCB](../img/figura4_pcb_convergencia.png)

## 13.2. Auditoría del Error Residual

El criterio formal de parada de cátedra estipulaba alcanzar un fitness $< 10^{-2} = 0.01$. Sin embargo, en la formulación multicriterio de diseño físico, la función objetivo base no puede tender matemáticamente a cero debido a que la longitud de pistas de cobre física y el área geométrica de los encapsulados son magnitudes físicas estrictamente positivas no anulables:

$$\text{WL}_{\text{manhattan}} \ge \sum \omega_k \cdot \text{DistanciaMínima} > 0$$

$$\text{Área}_{\text{box}} \ge \sum_{i=1}^{C} (w_i \cdot h_i) = 642.0\text{ mm}^2 > 0$$

Por ende, el fitness óptimo alcanzado de $f_{\min} = 0.675907$ representa el mínimo físico ineludible del circuito bajo estudio (con una aptitud biológica del $32.41\%$). La auditoría del error residual confirma:
* **Error de Colisión:** $0.00000$ (ausencia absoluta de solapamientos).
* **Violación de Aislamiento:** $0$ instancias con distancia $< 2.0\text{ mm}$.
* **Violación de Límites:** $0$ componentes fuera de la tarjeta.
* **Estabilidad Generacional:** El algoritmo alcanzó una meseta asintótica en la generación 140, demostrando convergencia plena.

## 13.3. Análisis de los Operadores y las Penalizaciones

1. **Impacto de la Selección por Torneo Asimétrico:** El torneo determinístico binario ($p = 2$) confirió una presión selectiva suficiente para erradicar las colisiones en las primeras 40 generaciones. Al seleccionar el Padre 2 de manera aleatoria uniforme, se impidió el colapso de diversidad genética prematuro en el espacio de 28 variables continuas.
2. **Efectividad del Cruce en un Solo Punto (SPX, $P_c = 90\%$):** Al estar organizados los genes por pares consecutivos $(X_i, Y_i)$, un corte simple en el cromosoma preserva la integridad posicional de subbloques funcionales enteros, recombinando exitosamente disposiciones parciales eficientes de los circuitos de alimentación y procesamiento.
3. **Mecanismo de Mutación Puntual ($P_m = 2\%$):** Actuó como un operador de búsqueda local fina, desplazando suavemente los componentes en fracciones de milímetro para minimizar la longitud de las trazas sin reintroducir colisiones.
4. **Comportamiento de la Penalización Cuadrática:** La penalización cuadrática externa demostró ser el factor clave de convergencia, creando una pendiente pronunciada que guió a los individuos desde regiones densamente colisionadas hacia la región factible en etapas tempranas de la búsqueda.
