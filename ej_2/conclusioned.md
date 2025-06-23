# Conclusiones

## QAgent

### Ingeniería de Características
Se optó por utilizar la posición y la velocidad del pájaro como features para el QAgent. 

- Discretización de la posición del pájaro:
    Tenemos en cuenta la posición relativa del jugador respecto al tubo superior y inferior, y la posición relativa al centro del tubo. Se divide en 4 casos:
    - Por encima del tubo superior: 0
    - Entre los tubos, más cerca del tubo superior: 2
    - Entre los tubos, más cerca del tubo inferior: 3
    - Por debajo del tubo inferior: 1

- Discretización de la velocidad del jugador:
    Tenemos en cuenta la velcidad del pájaro con respecto al eje de coordenadas Y. Se divide en 5 casos:
    - Muy rápido descendiendo: -2
    - Rápido descendiendo: -1
    - Estable: 0
    - Rápido ascendiendo: 1
    - Muy rápido ascendiendo: 2

En un principio se optó por agregar la discretización de la distancia con respecto a los tubos pero no alteró los resultados de forma significativa. Sin embargo, se veía afectado el tiempo de entrenamiento. Por esto despreciamos la implementación de las demás features. 

### Rendimiento QAgent
El modelo logra un correcto balance entre rendimiento y tiempo de entrenamiento. Resultando este efectivo a la hora de sobrepasar tubos. Se nota una leve oscilación en el pájaro debido a que intenta mantenerse en el centro del gap de los dos tubos. 


## Red neuronal
En el desarrollo del modelo, se utilizó una arquitectura de red neuronal de 9 capas, incluyendo normalización y dropout. Anteriormente, se intentó que esta sea menos compleja, pero los resultados si bien no eran tan diferentes (aprox un mae en el set de validación de 0.2), se mantuvo el mejor resultado obtenido. 

### Rendimiento NN
Se observa un rendimiento similar al del QAgent, con la diferencia que el pájaro no oscila de la misma forma.

## Comparacion y conclusiones finales
En base al rendimiento de ambos agentes, y debido a la simplicidad del problema, creemos que es mejor utilizar el QAgent, si no nos interesa la parte de la oscilación con un fin visual. 



