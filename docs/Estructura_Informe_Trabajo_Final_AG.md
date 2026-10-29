# Estructura Limpia del Informe Final: Algoritmos Genéticos (AG)
## Asignatura: Inteligencia Artificial (Corte 3) — Universidad de Cartagena

> **Guía directa para estructurar el documento en Word (.docx)**  
> Sin saturación de viñetas, con prosa fluida y la misma jerarquía académica utilizada en los informes de Corte 1 y Corte 2.

---

# PORTADA
* **Título:** CUARTO TALLER: ALGORITMOS GENÉTICOS (AG)
* **Subtítulo:** Optimización Heurística y Solución de Problemas Complejos de Ingeniería Mediante Algoritmo Genético Simple (SGA) con Manejo de Restricciones y Funciones de Penalización
* **Institución:** Universidad de Cartagena — Facultad de Ingeniería — Programa de Ingeniería de Sistemas
* **Asignatura:** Inteligencia Artificial (Tercer Corte)
* **Docente:** Ing. Manuel Alejandro Ospina Alarcón
* **Autores:** Dago David Palmera Navarro (Cód. 0222321003) & Julián David Camargo Padilla (Cód. 0222320016)
* **Lugar y Fecha:** Cartagena de Indias, D. T. y C., Noviembre de 2026

---

# ÍNDICE GENERAL / TABLA DE CONTENIDO
*(Generado automáticamente desde Word en Referencias $\rightarrow$ Tabla de contenido)*

---

# 1. INTRODUCCIÓN

*(Sección en prosa continua y fluida de 3 a 4 párrafos completos, sin viñetas ni subdivisiones 1.1 o 1.2)*

* **Párrafo 1 (Fundamento del paradigma):** Apertura sobre el Soft Computing y el origen de la Computación Evolutiva. Contraste entre la rigidez de los métodos analíticos exactos y la necesidad de heurísticas bioinspiradas para explorar espacios de búsqueda no lineales, multimodales y NP-difíciles donde la búsqueda exhaustiva resulta inviable.
* **Párrafo 2 (El Algoritmo Genético Simple):** Fundamentos del marco de John Henry Holland (1975) y el principio darwiniano de supervivencia del más apto. Explicación de cómo la población evoluciona mediante operadores de selección, cruce, mutación y elitismo, adaptándose de forma estocástica pero guiada hacia regiones óptimas.
* **Párrafo 3 (Aplicación al problema del trabajo y restricciones):** Presentación del caso de estudio de ingeniería abordado en este informe (del banco de la prueba oficial). Planteamiento de la necesidad de manejar restricciones del mundo real mediante funciones de penalización para dirigir la búsqueda hacia el espacio de soluciones factibles.
* **Párrafo 4 (Propósito y estructura del informe):** Declaración explícita del alcance del trabajo, los parámetros canónicos evaluados ($N=1000$, Elitismo=2, $p=2$, $P_c=90\%$, $P_m=2\%$, Parada $< 10^{-2}$) y síntesis del contenido presentado a lo largo de los capítulos.

---

# 2. OBJETIVOS

## 2.1. Objetivo General
Diseñar, implementar y evaluar computacionalmente un Algoritmo Genético Simple (SGA) robusto y modular con manejo de restricciones y funciones de penalización, con el propósito de optimizar un problema de ingeniería de alta dimensionalidad y analizar la dinámica de convergencia poblacional.

## 2.2. Objetivos Específicos
1. Analizar los fundamentos matemáticos y bioinspirados del Algoritmo Genético Simple, formalizando el mapeo de genotipo a fenotipo y la formulación analítica de la función de aptitud (*fitness*).
2. Modelar e integrar formalmente las restricciones operativas del problema (fuertes y débiles) mediante funciones de penalización para garantizar la factibilidad de la solución óptima.
3. Implementar de forma modular los operadores genéticos canónicos: selección por torneo determinístico ($p=2$), elitismo estricto (2 individuos), cruce en un punto (SPX con $P_c=0.90$) y mutación puntual ($P_m=0.02$).
4. Monitorear y registrar la evolución de la población, documentando cuantitativamente la población inicial aleatoria ($t=0$) y la población final convergida ($t=t_{\text{fin}}$).
5. Evaluar el desempeño del algoritmo a través de curvas de convergencia generacional (mejor, promedio y peor fitness), auditando los errores residuales y formulando conclusiones técnicas basadas en la experimentación.

---

# 3. FUNDAMENTACIÓN TEÓRICA

## 3.1. Analogía Biológico-Computacional y Terminología
Mapeo entre las estructuras biológicas de la genética natural (individuo, cromosoma, gen, alelo, población y aptitud) y sus equivalentes en optimización matemática (solución candidata, vector de variables, variable, valor del parámetro, conjunto de soluciones y función objetivo).

## 3.2. Estructura y Flujo del Algoritmo Genético Simple (SGA)
Ciclo evolutivo canónico, Teorema del Esquema de Holland e Hipótesis de los Bloques Constructores.

## 3.3. Operadores Genéticos y Dinámica Poblacional
* **Selección por Torneo y Elitismo:** Mecanismo de torneo binario determinístico ($p=2$), presión selectiva y demostración de cómo el elitismo garantiza la no degeneración del mejor fitness.
* **Cruce (Crossover) y Copia Asexual:** Recombinación en un punto (SPX) con $P_c=90\%$, locus estocástico de corte y justificación de estrategias reproductivas destructivas vs. no destructivas.
* **Mutación Puntual:** Perturbación estocástica de genes con $P_m=2\%$, función como seguro contra el estancamiento prematuro y preservación de la diversidad alélica.

## 3.4. Manejo de Restricciones en Optimización Bioinspirada
Definición de restricciones fuertes (*hard constraints*) y débiles (*soft constraints*). Análisis comparativo de los métodos de tratamiento: funciones de penalización (estáticas, dinámicas y adaptativas), métodos de región factible y métodos de reparación heurística.

---

# 4. DEFINICIÓN DEL PROBLEMA Y CASO DE ESTUDIO

## 4.1. Parámetros Generales Asignados por la Cátedra
Contextualización de las directrices oficiales: tamaño de población ($1000$ individuos), elitismo ($2$ individuos), torneo ($p=2$), cruce SPX ($90\%$), mutación puntual ($2\%$) y criterio de parada (fitness $< 10^{-2}$ o límite generacional).

## 4.2. Formulación del Problema Seleccionado
*(Se desarrolla el problema elegido de la prueba oficial)*
* **Contexto físico y variables de decisión:** Descripción de las variables continuas o discretas que definen el sistema.
* **Función Objetivo:** Expresión matemática que se persigue minimizar o maximizar.
* **Restricciones del Problema:** Catálogo formal de cotas, límites operativos y condiciones de viabilidad.

---

# 5. MODELADO MATEMÁTICO, CODIFICACIÓN Y DISEÑO DEL FITNESS

## 5.1. Codificación del Cromosoma (Genotipo $\rightarrow$ Fenotipo)
Definición del vector cromosómico, rango de definición de los alelos y función de decodificación hacia el espacio físico de soluciones.

## 5.2. Función de Aptitud Base y Función de Penalización
* Expresión analítica del fitness base ($f_{\text{base}}$).
* Formulación de las penalizaciones aditivas por violación de restricciones:
  $$\text{Fitness}(\vec{x}) = f_{\text{base}}(\vec{x}) + \sum_{k=1}^m \lambda_k \cdot R_k(\vec{x})$$
* Calibración y justificación técnica de los coeficientes de penalización $\lambda_k$.

---

# 6. IMPLEMENTACIÓN COMPUTACIONAL Y RESULTADOS EXPERIMENTALES

## 6.1. Arquitectura del Código y Modularidad
Descripción de las funciones implementadas (`Fitness`, `Elitismo`, `seleccion_torneo_2`, `Cruce`, `Mutacion` y script driver principal).

## 6.2. Registro y Análisis de la Población Inicial ($t = 0$)
Muestra representativa de la población inicial aleatoria, tabla de métricas (mejor fitness, promedio, peor fitness y desviación estándar) y análisis de diversidad genética inicial.

## 6.3. Registro y Análisis de la Población Final ($t = t_{\text{convergencia}}$)
Condición de parada alcanzada, tabla de los mejores individuos de la generación final, valores decodificados del **Mejor Individuo Absoluto** ($\vec{x}^*$) y comprobación matemática del 100% de cumplimiento de restricciones.

## 6.4. Comparativa Cuantitativa: Población Inicial vs. Población Final
Tabla resumen comparativa de la evolución poblacional (ganancia de aptitud, reducción de dispersión y refinamiento de parámetros).

---

# 7. ANÁLISIS DE CONVERGENCIA, ERRORES Y GRÁFICAS

## 7.1. Curvas de Convergencia Generacional
Análisis de la gráfica de evolución por épocas: curva del mejor fitness (élite), curva del promedio poblacional y curva del peor fitness. Identificación de las fases de exploración rápida y explotación asintótica.

## 7.2. Auditoría del Error Residual y Estabilidad
Comportamiento numérico del error a lo largo de las generaciones, tasa de mejora y ausencia de oscilaciones divergentes.

## 7.3. Análisis del Comportamiento de los Operadores y Penalizaciones
Evaluación del impacto del elitismo en la estabilidad y efectividad de los términos de penalización para expulsar soluciones infactibles hacia la región válida.

---

# 8. CONCLUSIONES

*(Redactadas en párrafos sólidos y numerados, destacando el concepto clave en negrita al inicio de cada conclusión)*

1. **Efectividad del Algoritmo Genético Simple en Espacios Complejos:** [Párrafo explicativo analizando la convergencia lograda frente a la dimensionalidad del problema].
2. **Impacto Decisivo de las Funciones de Penalización:** [Párrafo sobre cómo las penalizaciones mantuvieron la solución dentro del espacio factible sin comprometer la exploración].
3. **Rol del Elitismo en la Monotonía de la Convergencia:** [Párrafo justificando que la copia de los 2 mejores individuos garantizó que no hubiese pérdida de aptitud acumulada].
4. **Equilibrio entre Presión Selectiva y Diversidad Genética:** [Párrafo evaluando el comportamiento conjunto del torneo $p=2$, cruce al $90\%$ y mutación al $2\%$].
5. **Comprobación de Cumplimiento y Proyección Técnica:** [Párrafo final sobre la calidad del óptimo alcanzado y recomendaciones de escalabilidad].

---

# 9. REFERENCIAS BIBLIOGRÁFICAS
*(Formato IEEE o APA 7.ª edición, 5 a 8 referencias académicas formales)*
* Holland, J. H. (1975). *Adaptation in Natural and Artificial Systems*. University of Michigan Press.
* Goldberg, D. E. (1989). *Genetic Algorithms in Search, Optimization, and Machine Learning*. Addison-Wesley.
* Michalewicz, Z. (1996). *Genetic Algorithms + Data Structures = Evolution Programs*. Springer-Verlag.
* Eiben, A. E., & Smith, J. E. (2015). *Introduction to Evolutionary Computing*. Springer.
* Ospina Alarcón, M. A. (2026). *Material de clase y diapositivas de Inteligencia Artificial*. Universidad de Cartagena.

---

# 10. ANEXOS
* **Anexo A.** Código Fuente Completo y Comentado del Algoritmo Genético.
* **Anexo B.** Tabla Numérica Extendida de Convergencia por Generaciones.
