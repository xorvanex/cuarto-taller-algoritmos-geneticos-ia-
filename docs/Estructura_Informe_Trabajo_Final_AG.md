# Estructura del Informe Final: Algoritmos Genéticos (AG)
## Asignatura: Inteligencia Artificial (Corte 3) — Universidad de Cartagena

> **Documento guía para copiar directamente en Microsoft Word (.docx)**  
> **Estilo recomendado para estilos de Word:**
> * `#` o Números Romanos/Enteros $\rightarrow$ **Título 1**
> * `##` o 1.1, 1.2 $\rightarrow$ **Título 2**
> * `###` o 1.1.1, 1.1.2 $\rightarrow$ **Título 3**
> * Viñetas $\rightarrow$ **Texto normal / Lista con viñetas**

---

# PORTADA
* **Título del Proyecto:** Optimización de Problemas Complejos de Ingeniería Mediante Algoritmos Genéticos Simples (SGA) con Manejo de Restricciones y Funciones de Penalización
* **Subtítulo:** Actividad de Trabajo Colaborativo — Computación Evolutiva y Bioinspirada
* **Asignatura:** Inteligencia Artificial (Tercer Corte)
* **Docente:** Ing. Manuel Alejandro Ospina Alarcón
* **Integrantes:** Dago David Palmera Navarro (Cód. 0222321003) & [Nombre Segundo Integrante]
* **Institución:** Universidad de Cartagena — Facultad de Ingeniería — Programa de Ingeniería de Sistemas
* **Ciudad y Fecha:** Cartagena de Indias, D. T. y C., Noviembre de 2026

---

# ÍNDICE GENERAL / TABLA DE CONTENIDO
*(Generar automáticamente en Word mediante Referencias -> Tabla de contenido)*

---

# 1. INTRODUCCIÓN
* **1.1. Contextualización del Paradigma Evolutivo:**
  * Fundamentos de la Computación Blanda (*Soft Computing*) y computación bioinspirada.
  * Diferencias clave entre métodos de búsqueda exacta/analítica y heurísticas adaptativas estocásticas.
* **1.2. Relevancia y Justificación en Problemas de Ingeniería:**
  * Justificación del uso de Algoritmos Genéticos en espacios de búsqueda de alta dimensionalidad y problemas NP-difíciles.
* **1.3. Objetivos del Trabajo:**
  * **1.3.1. Objetivo General:**
    * Diseñar, implementar computacionalmente y validar un Algoritmo Genético Simple (SGA) modular con mecanismos de penalización para la resolución óptima de problemas multivariables con restricciones.
  * **1.3.2. Objetivos Específicos:**
    * Definir el esquema de codificación (genotipo y fenotipo) adaptado al espacio de estados del problema seleccionado.
    * Formular una función de aptitud (*fitness*) matemáticamente rigurosa que penalice soluciones infactibles (restricciones fuertes y débiles).
    * Implementar los operadores evolutivos canónicos: selección por torneo ($p=2$), elitismo estricto (2 individuos), cruce en 1 punto (SPX, $P_c=90\%$) y mutación puntual ($P_m=2\%$).
    * Monitorear y registrar la población inicial, la población final y las curvas de convergencia generacional.
    * Realizar auditoría estadística de errores residuales y analizar la estabilidad del algoritmo frente al óptimo global.
* **1.4. Alcance y Organización del Documento:**
  * Resumen descriptivo de los capítulos que componen el informe técnico.

---

# 2. FUNDAMENTACIÓN TEÓRICA Y ESTADO DEL ARTE
* **2.1. Biología Evolutiva y su Correspondencia Computacional:**
  * Selección natural darwiniana y transmisión genética mendeliana.
  * Tabla de correspondencia entre el dominio biológico y el dominio de algoritmos genéticos:
    * Medio ambiente $\leftrightarrow$ Función objetivo / Espacio del problema.
    * Organismo / Individuo $\leftrightarrow$ Solución candidata.
    * Cromosoma $\leftrightarrow$ Vector de variables de decisión.
    * Gen $\leftrightarrow$ Variable / Parámetro particular.
    * Alelo $\leftrightarrow$ Valor asignado a la variable.
    * Fitness / Aptitud $\leftrightarrow$ Bondad o calidad de la solución.
* **2.2. Arquitectura del Algoritmo Genético Simple (SGA de Holland):**
  * Flujo generacional iterativo.
  * Teorema del Esquema (*Schema Theorem*) e Hipótesis de los Bloques Constructores (*Building Blocks*).
* **2.3. Operadores Genéticos y Parámetros Operativos:**
  * **2.3.1. Operadores de Selección:**
    * Selección por Torneo Determinístico ($p=2$): presión selectiva y conservación de diversidad.
    * Elitismo: fundamentación matemática para garantizar monotonía no decreciente en la mejor aptitud.
  * **2.3.2. Operador de Cruce (Crossover):**
    * Cruce en un punto (SPX): selección estocástica del locus de corte y tasa de cruce ($P_c = 0.90$).
    * Estrategia reproductiva destructiva vs. no destructiva y operador de copia asexual.
  * **2.3.3. Operador de Mutación:**
    * Perturbación puntual con alelo aleatorio y tasa controlada ($P_m = 0.02$).
    * Equilibrio entre exploración global y explotación local para evitar óptimos locales.
* **2.4. Manejo Formal de Restricciones en Algoritmos Genéticos:**
  * Restricciones Fuertes (*Hard Constraints*) vs. Restricciones Débiles (*Soft Constraints*).
  * **2.4.1. Funciones de Penalización:**
    * Penalizaciones Estáticas: factor de castigo constante.
    * Penalizaciones Dinámicas: castigo progresivo según el número de generación $t$.
    * Penalizaciones Adaptativas: castigo modulado por el porcentaje de soluciones factibles.
  * **2.4.2. Métodos Alternativos:**
    * Búsqueda en Región Factible (*Feasible Space*).
    * Métodos y operadores de Reparación Heurística.

---

# 3. DEFINICIÓN DEL PROBLEMA Y CASO DE ESTUDIO
* **3.1. Marco General y Parámetros Asignados por la Cátedra:**
  * Tamaño de población: $N = 1000$ individuos.
  * Elitismo: 2 mejores individuos.
  * Algoritmo de selección: Torneo determinístico ($p = 2$).
  * Algoritmo de cruce: 1 punto (SPX) con $P_c = 90\%$.
  * Algoritmo de mutación: Puntual con $P_m = 2\%$.
  * Criterio de parada: Individuo con fitness $< 10^{-2}$ o límite generacional.
* **3.2. Formulación del Problema Seleccionado (del Banco Oficial de la Prueba):**
  *(Completar con la opción seleccionada: Rutas de Entrega, PCB Layout o Cartera de Inversiones)*
  * **3.2.1. Contexto y Descripción del Escenario de Ingeniería:**
    * Descripción del sistema, entidades participantes y variables de entorno.
  * **3.2.2. Función Objetivo:**
    * Formulación analítica de minimización o maximización.
  * **3.2.3. Catálogo de Restricciones del Sistema:**
    * Tabla clasificatoria de restricciones (fuertes y débiles).
    * Ecuaciones analíticas de cumplimiento e intervalos admisibles.

---

# 4. MODELADO MATEMÁTICO, CODIFICACIÓN Y DISEÑO DEL FITNESS
* **4.1. Diseño del Genotipo y Fenotipo (Cromosoma):**
  * Estructura de datos del cromosoma (vector numérico continuo / entero / permutacional).
  * Función de decodificación y mapeo del genotipo hacia la solución física (fenotipo).
* **4.2. Formulación de la Función de Aptitud Base ($f_{\text{base}}$):**
  * Expresión analítica y cálculo del costo/rendimiento directo.
* **4.3. Diseño de la Función de Penalización Integral:**
  * Formulación matemática de las penalizaciones aplicables ante transgresión de cotas:
    $$\text{Fitness}(\vec{x}) = f_{\text{base}}(\vec{x}) + \sum_{k=1}^m \lambda_k \cdot R_k(\vec{x})$$
  * Justificación cuantitativa de los coeficientes de castigo $\lambda_k$.
* **4.4. Diagrama de Flujo del Proceso de Evaluación:**
  * Diagrama de bloques que ilustra la recepción del cromosoma, decodificación, evaluación base, cálculo de castigos y retorno de la aptitud consolidada.

---

# 5. IMPLEMENTACIÓN COMPUTACIONAL Y ARQUITECTURA DEL SOFTWARE
* **5.1. Entorno Tecnológico y Herramientas Utilizadas:**
  * Lenguaje de programación (Python 3.x / MATLAB), librerías auxiliares (NumPy, Matplotlib, SciPy, Pandas).
* **5.2. Arquitectura y Modularidad del Código:**
  * Descripción de módulos, funciones y scripts del repositorio:
    * Módulo de inicialización poblacional.
    * Módulo de cálculo de fitness y restricciones.
    * Módulo de selección (Torneo y Elitismo).
    * Módulo de reproducción (Cruce SPX y Mutación Puntual).
    * Bucle generacional y criterio de parada.
* **5.3. Tabla de Hiperparámetros Calibrados:**
  * Resumen tabular de todos los parámetros de ejecución.

---

# 6. RESULTADOS EXPERIMENTALES Y AUDITORÍA DE POBLACIONES
* **6.1. Análisis y Registro de la Población Inicial ($t = 0$):**
  * Metodología de inicialización uniforme aleatoria.
  * Tabla con muestra representativa de individuos de la Generación 0 (cromosomas y aptitudes).
  * Estadísticas de arranque: mejor fitness inicial, fitness promedio inicial, peor fitness inicial y desviación estándar.
  * Análisis de la diversidad genética y cobertura del espacio de búsqueda al inicio.
* **6.2. Análisis y Registro de la Población Final ($t = t_{\text{convergencia}}$):**
  * Criterio de parada satisfecho (generación exacta y condición de corte).
  * Tabla con el Top de mejores soluciones convergidas (Generación final).
  * Especificación detallada del **Mejor Individuo Absoluto** ($\vec{x}^*$): genes decodificados y valor de fitness óptimo.
  * Comprobación matemática del cumplimiento del 100% de las restricciones en la solución final.
* **6.3. Tabla Comparativa: Población Inicial vs. Población Final:**
  * Métricas cuantitativas de evolución (reducción de varianza, homogeneidad alélica y ganancia de aptitud).

---

# 7. ANÁLISIS DE CONVERGENCIA, ERRORES Y GRÁFICAS
* **7.1. Curvas de Evolución Generacional (Curvas de Convergencia):**
  * Gráfica longitudinal por generación:
    * Curva del Mejor Fitness (evolución del líder elitista).
    * Curva del Fitness Promedio (salud global de la población).
    * Curva del Peor Fitness (cota inferior poblacional).
  * Identificación de las fases del proceso: fase exploratoria rápida, fase de transición y fase de explotación/refinamiento asintótico.
* **7.2. Auditoría y Análisis del Error:**
  * Evolución de la discrepancia residual absoluta generación tras generación.
  * Análisis de estabilidad y ausencia de oscilaciones destructivas.
* **7.3. Análisis de Sensibilidad e Impacto de Parámetros:**
  * Evaluación del efecto del operador de elitismo ($2$ individuos) frente a una variante sin elitismo.
  * Análisis comparativo variando la tasa de mutación ($P_m = 1\%$, $2\%$ y $5\%$).
* **7.4. Validación de la Solución Óptima:**
  * Comparación de la solución heurística obtenida frente a valores teóricos o métodos analíticos de referencia.

---

# 8. CONCLUSIONES
* Conclusión 1: Eficiencia y capacidad de búsqueda global del Algoritmo Genético Simple frente a espacios multivariables complejos.
* Conclusión 2: Efectividad y rigor de las funciones de penalización en la exclusión de regiones infactibles sin destruir la diversidad genética.
* Conclusión 3: Justificación del equilibrio óptimo entre la presión selectiva del torneo determinístico ($p=2$) y la tasa controlada de mutación ($2\%$).
* Conclusión 4: Comportamiento asintótico del elitismo estricto como garante de convergencia robusta y monótona.
* Conclusión 5: Recomendaciones y extensiones futuras (algoritmos genéticos multiobjetivo NSGA-II, codificación adaptativa y paralelización).

---

# 9. REFERENCIAS BIBLIOGRÁFICAS
*(Formato IEEE o APA 7.ª edición)*
* Holland, J. H. (1975). *Adaptation in Natural and Artificial Systems*. University of Michigan Press.
* Goldberg, D. E. (1989). *Genetic Algorithms in Search, Optimization, and Machine Learning*. Addison-Wesley.
* Michalewicz, Z. (1996). *Genetic Algorithms + Data Structures = Evolution Programs*. Springer-Verlag.
* Eiben, A. E., & Smith, J. E. (2015). *Introduction to Evolutionary Computing*. Springer.
* Ospina Alarcón, M. A. (2026). *Material de clase y diapositivas: Algoritmos Genéticos y Computación Evolutiva*. Universidad de Cartagena.

---

# 10. ANEXOS
* **Anexo A. Código Fuente Modular Completo:**
  * Scripts comentados y formateados con sintaxis limpia.
* **Anexo B. Tabla Numérica de Convergencia Generacional:**
  * Muestreo tabular de aptitudes (época a época o intervalos de 25 generaciones).
* **Anexo C. Registro de Auditoría de Cumplimiento de Restricciones:**
  * Tabla de comprobación numérica de cada restricción frente a la solución óptima.
