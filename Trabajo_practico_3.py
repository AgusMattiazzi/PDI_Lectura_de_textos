# Librerias por defecto
import cv2
import numpy as np
import matlab       # Funciones homologas a matlab

## --------------------- Comandos basicos de Python -------------------- #
# Abrir Imagen
Img = cv2.imread('foto_1.jpg')

# Conversion a escala de grises
Img_gris = cv2.cvtColor(Img,cv2.COLOR_BGR2GRAY)

# Umbralado
# ret,Img_BW1 = cv2.threshold(Img_gris,120,255,cv2.THRESH_BINARY)
ret,Img_BW1 = cv2.threshold(Img_gris,25,255,cv2.THRESH_OTSU)
# matlab.imshow(Img_BW1, title = 'Imagen binaria')
# Los bordes son negros y el fondo (la hoja) es blanco. Para cambiar ésto
# simplemente se invierte la imagen
Img_BW1 = cv2.bitwise_not(Img_BW1)  # Negativo

# Limpiar bordes
Img_BW = matlab.imclearborder(Img_BW1)
# matlab.imshow(Img_BW, title = 'Sin bordes')

# Dimensiones de la imagen
Alto,Ancho = reversed(Img_BW.shape[:2])      # Obtengo las dimensiones
# print("Dimensiones de la imagen: ", Ancho, Alto, "\n")

## --------------------------------------------------------------------- #

## ---------------------------- Dilatacion ----------------------------- #
# Al principio se ejerce una dilatación pequeña para evitar que las
# letras tengan sus contornos mal cerrados

# Para dilatar, hay que crear una estructura llamada kernel
kernel = np.ones((3,3),np.uint8)
Img_BW = cv2.dilate(Img_BW, kernel, iterations = 1)
# El tercer argumento es opcional y por defecto es 1

## --------------------------------------------------------------------- #

## ---------------------- Componentes conectadas ----------------------- #
# img = cv2.imread('Letras y Figuras.tif', cv2.IMREAD_GRAYSCALE)
Output = cv2.connectedComponentsWithStats(Img_BW, 8, cv2.CV_32S)
# https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#ga107a78bf7cd25dec05fb4dfc5c9e765f

# Resultados
num_labels =    Output[0]   # Cantidad de elementos
labels =        Output[1]   # Matriz con etiquetas
stats =         Output[2]   # Matriz de stats
centroids =     Output[3]   # Centroides de elementos

# Coloreamos los elementos
Img_color = matlab.label2rgb(labels,num_labels)

Suma_Alto = 0;    Suma_Ancho = 0

Palabras = Img_BW.copy()

# Agregamos Bounding Box
for i in range(1,num_labels):
    Bbox = stats[i,]
    # Bbox[0] : Coordenada x punto superior izquierdo
    # Bbox[1] : Coordenada y punto superior izquierdo
    # Bbox[2] : Ancho
    # Bbox[3] : Alto

    # Dibujar rectangulos
    cv2.rectangle(Img_color, (Bbox[0], Bbox[1]), (Bbox[0]+Bbox[2],
        Bbox[1]+Bbox[3]), (255,255,255), 2)
    
    cv2.rectangle(Palabras, (Bbox[0], Bbox[1]), (Bbox[0]+Bbox[2],
        Bbox[1]+Bbox[3]), (255,255,255), 2)

    Suma_Ancho = Suma_Ancho + Bbox[2]
    Suma_Alto = Suma_Alto + Bbox[3]

# matlab.imshow(Img_color, title = 'Imagen etiquetada')

# Ancho y alto promedio de letras.
Ancho_prom = Suma_Ancho/num_labels
Alto_prom = Suma_Alto/num_labels

print("Dimensiones promedio de las letras")
print("Ancho: {:.2f},\tAlto: {:.2f}".format(Ancho_prom, Alto_prom))
# Se formatean los float para q se muestren sólo dos decimales, una porqueria la verdad.

# matlab.imshow(Img_color, title = 'Matriz Etiqueta RGB')

# Se usa una dilatación para unir las letras de cada palabra
Palabras = matlab.imfill(Palabras)
kernel = np.ones((8,8),np.uint8)    # square (5,5)
Palabras = cv2.dilate(Palabras, kernel, iterations = 1)
# matlab.imshow(Palabras, title = 'Palabras 1')  # Parece prometedor

## ------------------------ Deteccion de Texto ------------------------- #
# Se usa dilatación con distintos kernel para resaltar palabras o párrafos
dil_palabra = 0.3;  dil_texto = 4

Img_palabra = matlab.dilatacion(Img_BW, Ancho_prom, Alto_prom, dil_palabra)
# matlab.imshow(Img_palabra, title = 'Palabras detectadas')

Img_texto = matlab.dilatacion(Img_BW, Ancho_prom, Alto_prom, dil_texto)
# matlab.imshow(Img_texto, title = 'Texto detectado')

## ------------------------ Aproximar contorno ------------------------- #
Img_gris2 = cv2.cvtColor(Img_texto,cv2.COLOR_BGR2GRAY)

# Umbralado
# ret,Img_BW1 = cv2.threshold(Img_gris,120,255,cv2.THRESH_BINARY)
ret,Area_Texto = cv2.threshold(Img_gris2,25,255,cv2.THRESH_OTSU)
Ancho,Alto = Area_Texto.shape

Contours, _ = cv2.findContours(Area_Texto,cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)

for cnt in Contours:
    area = cv2.contourArea(cnt)
    # Calcula el area de cada contorno
    if area > 0.2*Ancho*Alto:
        # Si el area elegida abarca un 20% de la imagen

        # Envolvente convexa
        Hull = cv2.convexHull(cnt)
        Hull = np.resize(Hull,( int( np.size(Hull)/2 ), 2))   # Formateo
        # cv2.drawContours(Img_color,[Hull],0,(255,0,255),5 )
        # for points in Hull:
        #     cv2.circle(Img_color,(points[0],points[1]),5,(255,0,0),10)

        # Minimo Rectangulo contenedor de la figura
        Rect = cv2.minAreaRect(cnt)
        points = cv2.boxPoints(Rect)
        Box = np.int32(points)
        # Dibujar Rectangulo de minima area que abarca al texto
        cv2.drawContours(Img_color,[Box],0,(0,0,255),5 )

# Imprimir resultados para la caja, funca relativamente bien, faltan ajustes
print(Box, Box.shape)
# print(Bbox)
# Hull = np.resize(Hull,(np.int( np.size(Hull)/2 ), 2))
# print(Hull, Hull.shape )
# matlab.imshow(Img_color, title = 'Figura')
# matlab.imshow(Img_parrafo, title = 'Figura')

## --------------------------------------------------------------------- #

## ------------------------ Adaptar contorno --------------------------- #
# Lo mejor de los ensayos anteriores es el mínimo rectángulo contenedor en
# la figura.

top,bottom,left,right = matlab.expand(Img_gris2,Box) # En realidad no es de
# matlab pero la puse ahí para tener todo en el mismo archivo

Img_texto = cv2.copyMakeBorder(Img_texto, top, bottom, left,
                                 right, cv2.BORDER_CONSTANT, value = 0)
Img_gris2 = cv2.copyMakeBorder(Img_gris2, top, bottom, left,
                                 right, cv2.BORDER_CONSTANT, value = 0)

# Umbralado
ret,Area_Texto = cv2.threshold(Img_gris2,25,255,cv2.THRESH_OTSU)
Ancho,Alto = Area_Texto.shape

Contours, _ = cv2.findContours(Area_Texto,cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)

for cnt in Contours:
    area = cv2.contourArea(cnt)
    # Calcula el area de cada contorno
    if area > 0.2*Ancho*Alto:
        # Si el area elegida abarca un 20% de la imagen

        # Envolvente convexa
        Hull = cv2.convexHull(cnt)
        Hull = np.resize(Hull,( int( np.size(Hull)/2 ), 2))   # Formateo
        cv2.drawContours(Img_texto,[Hull],0,(255,0,255),5 )
        for points in Hull:
            cv2.circle(Img_texto,(points[0],points[1]),5,(255,0,0),10)

        # Minimo Rectangulo contenedor de la figura
        Rect = cv2.minAreaRect(cnt)
        points = cv2.boxPoints(Rect)
        Box = np.int32(points)
        print(Box, Box.shape)
        cv2.drawContours(Img_texto,[Box],0,(0,0,255),5 )
        # Pasar Box como una lista

# Imprimir resultados para la caja, funca relativamente bien, faltan ajustes
# print(Bbox)
# Hull = np.resize(Hull,(np.int( np.size(Hull)/2 ), 2))
# print(Hull, Hull.shape )
# matlab.imshow(Img_texto, title = 'Figura ajustada')

# P_1 = get_corners(Img,100,100)
# A traves de get_corners se obtienen las esquinas automaticamente

# for points in Hull:
#     cv2.circle(Img_color,(points[0],points[1]),5,(255,0,0),10)

Img_esq = Img_gris2.copy()
# Img_esq = cv2.drawContours(Img_esq,Box,-1,(255,0,0),25)
# cv2.polylines(Img_esq, Box, True, (0,0,255),25)
# matlab.imshow(Img_esq, title = 'Esquinas detectadas')

Box = np.float32(Box)
# Se convierte P_1 a float32
Box = np.float32([ [2723,334],[832,1155],[1511,2386],[3749,1155] ]) # Valor obtenido a mano

Ancho,Alto = 2000,800    # Ancho y alto de la nueva imagen

P_2 = np.float32([ [Ancho,0],[0,0],[0,Alto],[Ancho,Alto] ])
# Hay que tener cuidado con el orden de los puntos en P_2, deben
# corresponder a las mismas esquinas que los de la imagen original,
# en el mismo orden

Matriz = cv2.getPerspectiveTransform(Box,P_2)
# Este comando crea la matriz para girar la imagen
# Ambos conjuntos deben ser de tipo np.float32

Img_BW = cv2.copyMakeBorder(Img_BW, top, bottom, left, right, cv2.BORDER_CONSTANT, value = 0)
Transformada = cv2.warpPerspective(Img_BW,Matriz,(Ancho,Alto))
# Transformada = cv2.warpPerspective(Img_color,Matriz,(Ancho,Alto))
# Este comando gira la imagen a partir de la matriz obtenida

matlab.imshow(Transformada, title = 'Imagen Transformada')

## --------------------------------------------------------------------- #