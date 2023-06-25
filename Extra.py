# Librerias por defecto
import numpy as np

## -------------------------- Funcion extender ------------------------- #
def extender(Dim, Punto_1, Punto_2):
    # Revisar formato de puntos (np.array conviene)

    # Alto, Ancho = Dim[:2]      # Obtengo las dimensiones
    Dim = np.flip(Dim[:2])          # Queda en formato (Ancho, Alto)

    Puntos = np.array([Punto_1,Punto_2])
    Dir = Puntos[1] - Puntos[0]

    # Dir = Punto_2 - Punto_1
    Dir = Dir / np.max(abs(Dir))    # Normalizo direccion
    Extremos = np.zeros( (2,2) )

    for pt in range(2): # For para los puntos
        # Extremo 1
        Signo = np.sign(Dir)    # Esto sirve para reconocer el limite
        t = np.zeros( (1,2) )   # t sirve para vectorizar conjuntos de puntos
        
        for i in range(2):  # For para coordenadas
            if (Signo[0,i] == 1):
                Lim = Dim[i]
            else: Lim = 0

            t[i] = (Lim - Puntos[pt,i])/Dir[0,i]

        t_final = np.max( abs(t) )

        Extremos[pt] = Puntos[pt] + Dir*t_final

        Dir = -Dir  # Cambia la direccion para encontrar el otro extremo

    return Extremos

## --------------------------------------------------------------------- #


P1 = np.array([(2896, 263)])
P2 = np.array([(2877, 250)])

Dimensiones = np.array([(2322, 4128)])

Extremos = extender(Dimensiones, P1, P2)
print(Extremos)