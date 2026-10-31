# 19. CONCLUSIONES GENERALES Y COMPARATIVA MULTI-PROBLEMA

## 19.1. Cuadro Comparativo Integral Multi-Problema

A lo largo del presente trabajo de investigación aplicada, se modelaron, implementaron y optimizaron tres problemas emblemáticos de la ingeniería contemporánea mediante una arquitectura algorítmica unificada basada en el **Algoritmo Genético Simple (SGA)** y **funciones de penalización exterior**. La siguiente tabla consolida y contrasta el comportamiento computacional, la dinámica bioinspirada y la efectividad operativa del motor genético a través de los tres dominios abordados:

| Dimensión de Análisis | Problema 1: Rutas de Entrega (VRP) | Problema 2: Diseño de Circuitos PCB | Problema 3: Cartera de Inversiones (Markowitz) |
| :--- | :--- | :--- | :--- |
| **Dominio de Ingeniería** | Logística y Transporte Industrial | Automatización Electrónica (EDA/CAD) | Ingeniería Financiera y Gestión Patrimonial |
| **Naturaleza del Espacio** | Combinatorio discreto con particionamiento | Continuo bidimensional ($2D$) | Continuo multivariado restringido ($10D$) |
| **Longitud del Cromosoma** | $15$ genes discretos (permutación/clientes) | $28$ genes continuos ($x_i, y_i \in [0, 1]$) | $10$ genes continuos ($w_i \in [0.0, 0.40]$) |
| **Espacio de Búsqueda** | $\approx 1.30 \times 10^{12}$ ordenamientos | Continuo $\mathbb{R}^{28}$ hipervolumétrico | Continuo $\mathbb{R}^{10}$ hipervolumétrico |
| **Objetivos Principales** | Distancia euclídea y tiempo de turno | Longitud Manhattan y área envolvente | Retorno medio proyectado y volatilidad |
| **Restricciones Fuertes** | Capacidad ($120\text{ kg}$) y jornada ($480\text{ min}$) | Cero solapamiento y aislamiento ($\ge 2\text{ mm}$) | Presupuesto unitario ($\sum w_i = 1$) y tope $35\%$ |
| **Formulación Penalización** | Lineal por exceso de masa y tiempo | Cuadrática por área de solapamiento ($mm^2$) | Cuadrática por discrepancia presupuestal y tope |
| **Tamaño de Población ($N$)** | $1000$ individuos | $1000$ individuos | $1000$ individuos |
| **Elitismo ($k$)** | $2$ mejores conservados intactos | $2$ mejores conservados intactos | $2$ mejores conservados intactos |
| **Selección y Cruce** | Torneo $p=2$, Cruce PMX/SPX adaptado ($P_c=0.9$) | Torneo $p=2$, Cruce SPX continuo ($P_c=0.9$) | Torneo $p=2$, Cruce SPX continuo ($P_c=0.9$) |
| **Tasa de Mutación ($P_m$)** | $0.02$ ($2.0\%$) por intercambio / puntual | $0.02$ ($2.0\%$) perturbación continua | $0.02$ ($2.0\%$) reemplazo puntual uniforme |
| **Tiempo de Cómputo** | $1.92$ segundos ($150$ generaciones) | $15.42$ segundos ($150$ generaciones) | $6.88$ segundos ($150$ generaciones) |
| **Fitness Inicial Promedio** | $4.85210$ | $42.85120$ | $16.40954$ |
| **Fitness Final Promedio** | $0.41025$ | $0.98512$ | $0.45411$ |
| **Mejor Fitness ($f_{\text{norm}}$)** | **$0.378254$** | **$0.675907$** | **$0.433483$** |
| **Aptitud Biológica Final** | **$62.17\%$** | **$32.41\%$** | **$56.65\%$** |
| **Métrica Física Clave** | **$429.8\text{ km}$** (vs. $750\text{ km}$ base) | **$1080.2\text{ mm}$** ($60.4\%$ ahorro de área) | **$14.78\%$ Retorno**, **$14.56\%$ Riesgo** |
| **Índice de Rendimiento** | Eficiencia logística: $+42.7\%$ | Densidad de pistas: $+58.2\%$ | **Ratio de Sharpe: $0.7744$** |
| **Factibilidad Final** | **100% Factible (0 infracciones)** | **100% Factible (0 colisiones)** | **100% Factible (0 excesos)** |

## 19.2. Conclusiones Técnicas y Científicas

A partir de los resultados experimentales contrastados y la verificación rigurosa de las soluciones obtenidas, se enuncian las siguientes conclusiones estructuradas:

1. **Universalidad y Robustez del Algoritmo Genético Simple (SGA):** La suite algorítmica canónica de la cátedra demostró una capacidad sobresaliente para resolver con éxito tres clases de problemas de optimización matemáticamente disímiles: optimización combinatoria discreta (VRP), empaquetamiento geométrico espacial bidimensional (PCB Layout) y optimización convexa continua no lineal (Markowitz). En todos los casos, partiendo de una inicialización aleatoria uniforme carente de conocimiento heurístico experto previo, la población alcanzó el $100\%$ de factibilidad técnica y convergió hacia configuraciones de alto rendimiento.
2. **Efectividad Decisiva del Elitismo Invariable ($k = 2$):** La conservación rigurosa de los dos mejores individuos de cada época evolutiva (`Elitismo.m`) fungió como la piedra angular de la estabilidad matemática. Garantizó monotonicidad estricta de la cota mínima de fitness ($\min f^{(t+1)} \le \min f^{(t)}$), impidiendo la degradación catastrófica de soluciones óptimas intermedias por efectos de mutaciones o cruces desfavorables, y aceleró la convergencia asintótica en las etapas tardías de la búsqueda.
3. **Mecanismo de Selección por Torneo Asimétrico ($p = 2$):** La estrategia de enfrentar a dos individuos seleccionados aleatoriamente y otorgar la condición de progenitor al de menor costo de fitness aportó una presión selectiva graduada pero inflexible. Dicha presión erradicó a los individuos no factibles durante las primeras 15 a 40 generaciones. Simultáneamente, la elección del segundo progenitor bajo distribución aleatoria uniforme (`seleccion_torneo_2.m`) inyectó la entropía genética necesaria para impedir la convergencia prematura hacia óptimos locales, preservando la diversidad poblacional.
4. **Superioridad de las Funciones de Penalización Cuadráticas:** La comparación entre el modelado de restricciones lineales y cuadráticas evidenció que la penalización exterior de orden cuadrático ($P = \lambda \Delta^2$) supera ampliamente a los enfoques estáticos o lineales. Al aumentar de forma cuadrática respecto a la magnitud de la infracción, impone un castigo severo ante desviaciones graves, generando un gradiente selectivo escarpado que propulsa a la población hacia el interior del poliedro factible, disipándose suavemente cuando los errores residuales caen por debajo del $1\%$.
5. **Decodificación Cromosómica sin Operadores de Reparación Forzados:** La formulación de mapeos analíticos acotados (como la transformación lineal en el PCB que garantiza la contención física dentro del sustrato) simplificó la búsqueda evolutiva al eliminar regiones infactibles triviales del espacio de estados, permitiendo que la recombinación SPX y la mutación puntual operen libremente sobre variables normalizadas sin violar las leyes físicas del problema.
6. **Integración Tecnológica y Desacoplamiento de Software:** La arquitectura modular implementada —que separa el motor de optimización matemática en Python de la capa de visualización e interactividad en HTML5/CSS3/JavaScript Glassmorphism— demostró ser un estándar óptimo para la divulgación científica y la validación en tiempo real de algoritmos bioinspirados, permitiendo auditar la evolución generacional tanto por consola como mediante interfaces visuales reactivas sin acoplamiento a servidores backend pesados.

---

# 20. REFERENCIAS BIBLIOGRÁFICAS

1. **Markowitz, H. (1952).** *Portfolio Selection*. The Journal of Finance, 7(1), 77–91. https://doi.org/10.1111/j.1540-6261.1952.tb01525.x
2. **Goldberg, D. E. (1989).** *Genetic Algorithms in Search, Optimization, and Machine Learning*. Addison-Wesley Professional. Reading, MA.
3. **Holland, J. H. (1975).** *Adaptation in Natural and Artificial Systems: An Introductory Analysis with Applications to Biology, Control, and Artificial Intelligence*. University of Michigan Press. Ann Arbor, MI.
4. **Toth, P., & Vigo, D. (Eds.). (2014).** *Vehicle Routing: Problems, Methods, and Applications* (2nd ed.). Society for Industrial and Applied Mathematics (SIAM). Philadelphia, PA.
5. **Sherwani, N. A. (1999).** *Algorithms for VLSI Physical Design Automation* (3rd ed.). Kluwer Academic Publishers. Norwell, MA.
6. **Ospina Alarcón, M. A. (2026).** *Guía de Cátedra y Algoritmos Canónicos para Algoritmos Genéticos (SGA): Elitismo, Selección por Torneo, Cruce SPX y Mutación Puntual*. Facultad de Ingeniería, Universidad de Cartagena.
7. **Coello Coello, C. A. (2002).** *Theoretical and numerical constraint-handling techniques used with evolutionary algorithms: a survey of the state of the art*. Computer Methods in Applied Mechanics and Engineering, 191(11–12), 1245–1287.
8. **Deb, K. (2000).** *An efficient constraint handling method for genetic algorithms*. Computer Methods in Applied Mechanics and Engineering, 186(2–4), 311–338.
9. **Sharpe, W. F. (1966).** *Mutual Fund Performance*. The Journal of Business, 39(1), 119–138.
10. **Michalewicz, Z. (1996).** *Genetic Algorithms + Data Structures = Evolution Programs* (3rd ed.). Springer-Verlag. Berlin, Heidelberg.

---

# 21. ANEXOS

## Anexo A. Código Fuente Completo — Problema 1 (VRP)
El código fuente íntegro del modelo, decodificador y evaluador para la Optimización de Rutas de Entrega se encuentra archivado en el repositorio oficial bajo la ruta:
`src/problems/vrp.py` (Versión 2.0 Certificada).

## Anexo B. Código Fuente Completo — Problema 2 (PCB Layout)
El código fuente del modelo de emplazamiento de microelectrónica y minimización de longitud Manhattan de trazas de cobre se ubica en:
`src/problems/pcb.py` (Versión 2.0 Certificada).

## Anexo C. Código Fuente Completo — Problema 3 (Markowitz)
El código fuente del modelo financiero de selección de carteras según Markowitz, cálculo de matrices de covarianzas, simulación de Monte Carlo y evaluación de Ratio de Sharpe se ubica en:
`src/problems/markowitz.py` (Versión 2.0 Certificada).

## Anexo D. Motor Genético Universal y Script Integrador
* Motor del Algoritmo Genético Simple con Operadores de Cátedra: `src/core/ga_engine.py`.
* Script Integrador y Suite de Ejecución Multi-Problema: `src/main.py`.
* Capa de Presentación Web Interactiva: `src/web/index.html` y `src/web/app.js`.
