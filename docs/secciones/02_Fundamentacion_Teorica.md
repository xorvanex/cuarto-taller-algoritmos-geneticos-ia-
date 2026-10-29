# 3. FUNDAMENTACIÓN TEÓRICA

## 3.1. Analogía Biológico-Computacional y Terminología en Algoritmos Genéticos

Los Algoritmos Genéticos constituyen una familia de métodos de búsqueda y optimización estocástica fundamentados en la teoría de la evolución biológica descrita por Charles Darwin y formalizada computacionalmente por John Henry Holland en la Universidad de Michigan durante la década de 1970. Su premisa central reside en la analogía directa entre el proceso de adaptación de las especies al entorno natural y el proceso de refinamiento iterativo de un conjunto de soluciones candidatas frente a una función de evaluación cuantitativa. Bajo esta correspondencia, el término *individuo* o *cromosoma* hace referencia a una solución candidata codificada como una cadena de valores (genotipo), mientras que su manifestación en el espacio del problema real —las variables de decisión concretas que representa— constituye el *fenotipo*. Cada posición discreta dentro del cromosoma recibe el nombre de *gen*, y los valores posibles que dicho gen puede asumir se denominan *alelos*.

El conjunto total de individuos que el algoritmo mantiene y evoluciona simultáneamente en cada iteración constituye la *población*, y cada iteración completa del ciclo evolutivo recibe el nombre de *generación*. La aptitud de un individuo para sobrevivir y reproducirse en el entorno del problema se cuantifica mediante la *función de aptitud* o *fitness*, cuyo valor escalar sintetiza la calidad de la solución representada por ese cromosoma respecto a la función objetivo del problema. Un individuo con bajo valor de fitness en problemas de minimización —o alto valor en problemas de maximización— representa una solución de alta calidad y, en consecuencia, posee mayor probabilidad de selección para la reproducción y de transferir sus características genéticas a la siguiente generación. Esta presión selectiva, combinada con mecanismos de recombinación y perturbación aleatoria, genera la dinámica que impulsa a la población hacia regiones del espacio de búsqueda progresivamente más prometedoras.

## 3.2. Estructura y Flujo del Algoritmo Genético Simple (SGA)

El Algoritmo Genético Simple opera mediante un ciclo generacional bien definido, cuya correcta implementación garantiza que la población evolucione de forma coherente hacia soluciones de calidad creciente. El proceso se inicia con la *inicialización aleatoria* de una población de N individuos, generados de manera uniforme en el espacio de búsqueda para garantizar diversidad fenotípica máxima en la primera generación. A continuación, cada individuo es evaluado mediante la función de aptitud, que asigna a cada cromosoma un valor escalar representativo de su calidad como solución al problema planteado.

Con los valores de fitness calculados, el ciclo evolutivo principal comienza por la fase de *selección por elitismo*, mediante la cual los mejores individuos de la generación actual son identificados y preservados de forma directa e incondicional en la siguiente generación, sin pasar por los operadores de cruce ni de mutación. Este mecanismo garantiza que la calidad del mejor individuo jamás disminuya a lo largo del proceso evolutivo, actuando como memoria histórica del óptimo encontrado. Sobre el resto de la población se aplican en secuencia los operadores de *selección por torneo*, *cruce en un punto* y *mutación puntual* para construir la nueva generación, la cual reemplaza íntegramente a la anterior. El ciclo se repite hasta que se satisface algún criterio de parada: la función de aptitud del mejor individuo alcanza un umbral de convergencia previamente establecido (fitness < 10⁻²), o bien se agota el número máximo de generaciones configurado. El pseudocódigo del SGA implementado en el presente trabajo puede expresarse de forma esquemática como sigue:

```
Inicializar población P(0) de tamaño N = 1000 aleatoriamente
Evaluar fitness de cada individuo en P(0)
t ← 0
Mientras fitness_min(P(t)) ≥ 10⁻² hacer:
    [elite_1, elite_2] ← Elitismo(P(t), fitness(t))       // 2 mejores pasan directos
    Para i = 1 hasta (N/2 - 1):
        [p1, p2]   ← Torneo(P(t))                          // Selección
        [h1, h2]   ← Cruce_SPX(p1, p2, Pc = 0.90)        // Recombinación
        [m1, m2]   ← Mutación(h1, h2, Pm = 0.02)          // Perturbación
        Agregar m1, m2 a P_nueva
    P(t+1) ← [elite_1, elite_2] ∪ P_nueva
    Evaluar fitness de cada individuo en P(t+1)
    t ← t + 1
Retornar mejor individuo x* con fitness mínimo
```

## 3.3. Operadores Genéticos

### 3.3.1. Selección por Torneo y Elitismo

La selección determina qué individuos de la generación actual tendrán el privilegio de reproducirse y contribuir con su material genético a la generación siguiente. En el SGA implementado se combinan dos estrategias de selección complementarias: el torneo determinístico de tamaño p = 2 y el elitismo estricto de 2 individuos.

El torneo determinístico de tamaño p = 2 opera del siguiente modo: se extraen aleatoriamente dos individuos de la población y se comparan sus valores de fitness; el individuo con menor fitness (en minimización) gana el torneo y es designado Padre 1. El Padre 2 se selecciona de forma completamente aleatoria sobre el conjunto poblacional, sin competencia, garantizando así que la presión selectiva no elimine completamente la diversidad genética necesaria para la exploración. Esta asimetría intencional entre la selección del primer y segundo padre distingue al esquema de torneo implementado en la cátedra del torneo simétrico clásico, introduciendo un equilibrio deliberado entre presión selectiva y aleatoriedad. Formalmente, si se denota el fitness del individuo i como f(i), entonces:

    Padre1 = argmin{f(a), f(b)},  donde a, b ∈ P son seleccionados aleatoriamente
    Padre2 = individuo c ∈ P seleccionado uniformemente al azar

El elitismo actúa como complemento determinístico de la selección estocástica. Al inicio de cada ciclo generacional, la población es ordenada ascendentemente por su valor de fitness, y los dos individuos con menor fitness (Elites 1 y 2) son copiados directamente a la nueva generación sin modificación alguna. Esta estrategia garantiza la propiedad de monotonía del mejor fitness, es decir, que el óptimo histórico nunca se degrada entre generaciones consecutivas, lo que otorga al algoritmo una garantía asintótica de convergencia hacia el mejor individuo descubierto en cualquier generación pasada.

### 3.3.2. Cruce en un Punto (SPX)

El cruce o recombinación es el operador genético primario de *explotación*, responsable de combinar el material genético de dos individuos parentales para producir descendencia que hereda características de ambos progenitores. El operador implementado es el cruce en un punto, denominado en la literatura SPX (*Single-Point Crossover*), cuyo mecanismo opera de la siguiente forma: dado un par de padres P1 y P2, cada uno de longitud L genes, se genera un número aleatorio uniforme r ∈ [0, 1]; si r ≤ Pc = 0.90, el cruce tiene lugar y se selecciona aleatoriamente un punto de corte k ∈ {1, 2, ..., L−1}. Los hijos resultantes se construyen intercambiando los segmentos genotípicos a partir del punto de corte:

    Hijo1 = [P1(1:k) | P2(k+1:L)]
    Hijo2 = [P2(1:k) | P1(k+1:L)]

Si el número aleatorio r > Pc (probabilidad del 10%), el operador de copia se aplica en su lugar, transfiriendo los padres directamente como hijos sin modificación. Este mecanismo de copia garantiza que no toda la descendencia sea producto de recombinación, preservando en cierta medida los esquemas genotípicos intactos de los mejores individuos seleccionados. La probabilidad de cruce Pc = 0.90 fue establecida por la cátedra y refleja un sesgo pronunciado hacia la explotación por recombinación, privilegiando la mezcla de esquemas prometedores en la vasta mayoría de los apareamientos.

### 3.3.3. Mutación Puntual

La mutación es el operador genético de *exploración* del SGA, encargado de introducir perturbaciones aleatorias en el genotipo de los individuos para diversificar el acervo genético poblacional y permitir al algoritmo escapar de cuencas de atracción de óptimos locales en las que la recombinación por sí sola no puede progresar. El operador de mutación puntual implementado actúa sobre cada par de hijos generados por el cruce de la siguiente manera: se genera un número aleatorio r ∈ [0, 1]; si r ≤ Pm = 0.02, la mutación se activa; en caso contrario, los hijos permanecen inalterados (operador de copia). Cuando la mutación es activada, se selecciona aleatoriamente uno de los dos hijos y dentro de él se selecciona aleatoriamente un gen k ∈ {1, 2, ..., L} cuyo valor es reemplazado por un nuevo alelo generado uniformemente en el dominio de variación correspondiente a esa variable de decisión.

La probabilidad de mutación Pm = 0.02 es deliberadamente baja en comparación con la probabilidad de cruce, lo que refleja el rol secundario de la mutación como operador de diversificación de fondo, frente al rol primario del cruce como motor principal de convergencia. Sin embargo, su contribución resulta crítica en etapas tardías del proceso evolutivo, cuando la diversidad poblacional ha disminuido significativamente por efecto de la selección y la recombinación, y la exploración del vecindario del óptimo actual se vuelve necesaria para refinar la solución o escapar de óptimos locales.

### 3.3.4. Manejo de Restricciones y Funciones de Penalización

En los problemas de optimización con restricciones de ingeniería —como el VRP, el PCB y la cartera de Markowitz— no todas las soluciones generadas durante la inicialización o la evolución del AG pertenecen necesariamente a la región factible del espacio de búsqueda. La gestión de estas restricciones en el marco de un algoritmo genético se realiza mediante la técnica de *penalización de la función de aptitud*, que transforma el problema de optimización con restricciones en un problema de optimización sin restricciones aumentando artificialmente el fitness de los individuos infactibles para desfavorecerlos en la selección.

La función de aptitud penalizada adopta la forma general:

    f_penalizada(x) = f_base(x) + λ · Σ max(0, gᵢ(x))²

donde f_base(x) es la evaluación de la función objetivo sobre el individuo x, gᵢ(x) ≥ 0 representa la violación de la restricción i-ésima (un valor positivo indica infactibilidad), y λ es el coeficiente de penalización que regula la severidad con la que se castiga la violación. Un valor de λ excesivamente pequeño permite que individuos infactibles proliferen en la población, contaminando la búsqueda; un valor excesivamente grande genera una frontera de penalización tan abrupta que impide la exploración del límite de la región factible. La calibración adecuada de λ para cada problema específico constituye en sí misma un aspecto técnico crítico del diseño del AG, y será documentada individualmente para cada uno de los tres problemas abordados en el presente trabajo. El mecanismo de penalización asegura que, conforme la población converge, los individuos infactibles sean sistemáticamente desplazados por individuos factibles de mayor calidad, guiando la trayectoria evolutiva hacia el óptimo global dentro de la región factible del espacio de decisión.

