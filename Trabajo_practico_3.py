# Librerias por defecto
import cv2
import numpy as np
import matlab       # Funciones homologas a matlab

## ------------------------------- Tratamiento de imagen inicial -------------------------------- #
# Abrir Imagen
Img_path = 'foto_7.jpg'
Img = cv2.imread(Img_path)

# Conversion a escala de grises
Img_gris = cv2.cvtColor(Img,cv2.COLOR_BGR2GRAY)

# Umbralado
# ret,Img_BW1 = cv2.threshold(Img_gris,120,255,cv2.THRESH_BINARY)
ret,Img_BW = cv2.threshold(Img_gris,25,255,cv2.THRESH_OTSU)

# Los bordes son negros y el fondo (la hoja) es blanco. Para cambiar ésto
# se invierte la imagen
Img_BW = cv2.bitwise_not(Img_BW)  # Negativo

# Dilatacion
# Al principio se ejerce una dilatación pequeña para evitar que las letras tengan sus contornos
# mal cerrados

# Para dilatar, hay que crear una estructura llamada kernel
kernel = np.ones((3,3),np.uint8)
Img_BW = cv2.dilate(Img_BW, kernel, iterations = 1)
# El tercer argumento es opcional y por defecto es 1

# Limpiar bordes
Img_BW = matlab.imclearborder(Img_BW)
matlab.imshow(Img_BW, title = 'Sin bordes')

# Dimensiones de la imagen
Alto,Ancho = reversed(Img_BW.shape[:2])      # Obtengo las dimensiones
# print("Dimensiones de la imagen: ", Ancho, Alto, "\n")

## ---------------------------------------------------------------------------------------------- #

## -------------------------------------- Aperture science -------------------------------------- #
# Al principio se ejerce una dilatación pequeña para evitar que las
# letras tengan sus contornos mal cerrados

# Para dilatar, hay que crear una estructura llamada kernel
kernel = np.ones((3,3),np.uint8)
Img_BW = cv2.morphologyEx(Img_BW, cv2.MORPH_OPEN, kernel)
# Img_BW = cv2.dilate(Img_BW, kernel, iterations = 1)

# El tercer argumento es opcional y por defecto es 1

## ---------------------------------------------------------------------------------------------- #


## ------------------------------ Alto y ancho promedio de letras ------------------------------- #
Output = cv2.connectedComponentsWithStats(Img_BW, 8, cv2.CV_32S)
Ancho_prom,Alto_prom = matlab.dim_promedio(Output)

print("Dimensiones promedio de las letras")
print("Ancho: {:.2f},\tAlto: {:.2f}".format(Ancho_prom, Alto_prom))
# Se formatean los float para q se muestren sólo dos decimales, una porqueria la verdad.

## ---------------------------------------------------------------------------------------------- #


## ------------------------------------ Deteccion de Texto -------------------------------------- #
# Se usa dilatación con distintos kernel para resaltar palabras o párrafos
dil_text = 4;  # dil_pal = 0.3

# Img_palabra = matlab.dilatacion_especial(Img_BW, Ancho_prom, Alto_prom, dil_pal)
# matlab.imshow(Img_palabra, title = 'Palabras detectadas 1')

Img_texto = matlab.dilatacion_especial(Img_BW, Ancho_prom, Alto_prom, dil_text)
matlab.imshow(Img_texto, title = 'Texto detectado')
## ---------------------------------------------------------------------------------------------- #


## ------------------------------------ Aproximar contorno -------------------------------------- #
Img_color = cv2.cvtColor(Img_BW,cv2.COLOR_GRAY2BGR) # Funciona de B&W a color tambien

Contours, _ = cv2.findContours(Img_texto,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for cnt in Contours:
    area = cv2.contourArea(cnt)
    # Calcula el area de cada contorno
    if area > 0.2*Ancho*Alto:
        # Si el area elegida abarca un 20% de la imagen

        # Envolvente convexa
        Hull = cv2.convexHull(cnt)
        Hull = np.resize(Hull,( int( np.size(Hull)/2 ), 2))   # Formateo
        cv2.drawContours(Img_color,[Hull],0,(255,0,255),5 )
        for points in Hull:
            cv2.circle(Img_color,(points[0],points[1]),5,(255,0,0),10)

        # Minimo Rectangulo contenedor de la figura
        Rect = cv2.minAreaRect(cnt)
        points = cv2.boxPoints(Rect)
        Box = np.int32(points)
        # Dibujar Rectangulo de minima area que abarca al texto
        cv2.drawContours(Img_color,[Box],0,(0,0,255),5 )

# matlab.imshow(Img_color, title = 'Imagen a color bla bla')
# Imprimir resultados para la caja, funca relativamente bien, faltan ajustes
print(Box, Box.shape)

## ---------------------------------------------------------------------------------------------- #


## -------------------------------------- Ajustar imagen ---------------------------------------- #
# Lo mejor de los ensayos anteriores es el mínimo rectángulo contenedor en
# la figura.

Img_color = matlab.expand(Img_color,Box)
Img_BW = matlab.expand(Img_BW,Box)
# En realidad no es de matlab pero la puse ahí para tener todo en el mismo archivo

# matlab.imshow(Img_color, title = 'Imagen Expandida')

Box = np.float32(Box)
# Se convierte P_1 a float32

match Img_path:
    # Se usan valores obtenidos a mano para hacer la transformación de perspectiva.
    case "foto_1.jpg":
        # Se elige como primer punto el que tiene menor coordenada 'y' y a partir de ahí se
        # recorren los puntos en sentido antihorario
        Box = np.float32([ [2713,360],[829,1170],[1508,2386],[3683,1185] ]) # Hecho
        # Box = np.float32([ Sup_Der, Sup_Izq, Inf_Izq,Inf_Der ])

    case "foto_2.jpg":
        Box = np.float32([ [2219,447],[3,792],[196,2067],[2416,1722] ]) # Hecho

    case "foto_3.jpg":
        Box = np.float32([ [3515,285],[895,398],[576,1750],[4036,1673] ]) # Hecho
    
    case "foto_4.jpg":
        Box = np.float32([ [3344,253],[742,240],[405,1647],[3780,1638] ]) # Hecho

    case "foto_5.jpg":
        Box = np.float32([ [940,105],[962,2136],[2081,2285],[2068,0] ]) # Hecho

    case "foto_6.jpg":
        Box = np.float32([ [3546,1244],[1780,158],[883,933],[2741,2309] ]) # Hecho

    case "foto_7.jpg":
        Box = np.float32([ [2426,269],[873,1673],[1943,2686],[3469,879] ]) # Hecho

Ancho,Alto = 2000,800    # Ancho y alto de la nueva imagen

P_2 = np.float32([ [Ancho,0],[0,0],[0,Alto],[Ancho,Alto] ])
# Hay que tener cuidado con el orden de los puntos en P_2, deben
# corresponder a las mismas esquinas que los de la imagen original,
# en el mismo orden

Matriz = cv2.getPerspectiveTransform(Box,P_2)
# Este comando crea la matriz para girar la imagen
# Ambos conjuntos deben ser de tipo np.float32


Transformada = cv2.warpPerspective(Img_BW,Matriz,(Ancho,Alto))
# Transformada = cv2.warpPerspective(Img_color,Matriz,(Ancho,Alto))
# Este comando gira la imagen a partir de la matriz obtenida

matlab.imshow(Transformada, title = 'Imagen Transformada')

## ---------------------------------------------------------------------------------------------- #


## ----------------------------------- Deteccion de palabras ------------------------------------ #
# Se usa dilatación con distintos kernel para resaltar palabras o párrafos
# No se dilata para obtener los caracteres

kernel = np.ones((3,3),np.uint8)
Img_BW = cv2.morphologyEx(Transformada, cv2.MORPH_OPEN, kernel)
Output = cv2.connectedComponentsWithStats(Img_BW, 8, cv2.CV_32S)
Img_color,num_labels = matlab.resultados(Output)
print("Número de caracteres: ", num_labels)
matlab.imshow(Img_color, title = 'Caracteres detectados')

dil_palabra = 0.25
Img_BW = matlab.dilatacion_especial(Transformada, Ancho_prom, Alto_prom, dil_palabra)
Output = cv2.connectedComponentsWithStats(Img_BW, 8, cv2.CV_32S)
Img_color,num_labels = matlab.resultados(Output)
print("Número de palabras: ", num_labels)
matlab.imshow(Img_color, title = 'Palabras detectadas')

dil_parrafos = 0.7
Img_BW = matlab.dilatacion_especial(Transformada, Ancho_prom, Alto_prom, dil_parrafos)
Output = cv2.connectedComponentsWithStats(Img_BW, 8, cv2.CV_32S)
Img_color,num_labels = matlab.resultados(Output)
print("Número de párrafos: ", num_labels)
matlab.imshow(Img_color, title = 'Parrafos detectados')

matlab.resultados_finales()
# Los resultados con la imagen transformada son mejores que con la imagen
# original. Se tiene mejor diferenciación de palabras y menos errores
# (aunque podría decirse que son la misma cosa)

## ---------------------------------------------------------------------------------------------- #