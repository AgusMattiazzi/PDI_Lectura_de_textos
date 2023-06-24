# Librerias por defecto
import cv2
import numpy as np

# Archivo donde voy a poner funciones homologas a matlab
import matlab

## --------------------- Comandos basicos de Python -------------------- #
# Abrir Imagen
Img = cv2.imread('foto_1.jpg')

# Conversion a escala de grises
Img_gris = cv2.cvtColor(Img,cv2.COLOR_BGR2GRAY)

# Umbralado
# ret,Img_BW1 = cv2.threshold(Img_gris,120,255,cv2.THRESH_BINARY)
ret,Img_BW1 = cv2.threshold(Img_gris,25,255,cv2.THRESH_OTSU)
Img_BW1 = cv2.bitwise_not(Img_BW1)  # Negativo

# Limpiar bordes
Img_BW = matlab.imclearborder(Img_BW1)

# Dimensiones de la imagen
Alto,Ancho = Img_BW.shape[:2]      # Obtengo las dimensiones
print("Dimensiones: ", Ancho, Alto)

# Mostrar Imagen
# imshow(Img_BW, title = 'Bordes Limpios')

## --------------------------------------------------------------------- #

## ---------------------------- Dilatacion ----------------------------- #
# Para dilatar, hay que crear una estructura llamada kernel
kernel = np.ones((3,3),np.uint8)    # square (5,5)
Img_BW = cv2.dilate(Img_BW, kernel, iterations = 1)
# El tercer argumento es opcional y por defecto es 1

# imshow(Img_BW, title = 'Canny + Dilatacion')

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
    # print (Bbox)
    # Bbox[0] : Coordenada x punto superior izquierdo
    # Bbox[1] : Coordenada y punto superior izquierdo
    # Bbox[2] : Ancho
    # Bbox[3] : Alto

    cv2.rectangle(Img_color, (Bbox[0], Bbox[1]), (Bbox[0]+Bbox[2],
        Bbox[1]+Bbox[3]), (255,255,255), 2)
    
    cv2.rectangle(Palabras, (Bbox[0], Bbox[1]), (Bbox[0]+Bbox[2],
        Bbox[1]+Bbox[3]), (255,255,255), 2)

    Suma_Ancho = Suma_Ancho + Bbox[2]
    Suma_Alto = Suma_Alto + Bbox[3]

# Bbox = stats[45,]
# print (Bbox)
# Crop = Img_color2[ Bbox[1]:Bbox[1]+Bbox[3], Bbox[0]:Bbox[0]+Bbox[2] ]
# imshow(Crop, title = 'Imagen Recortada')

Ancho_prom = Suma_Ancho/num_labels
Alto_prom = Suma_Alto/num_labels

print(Ancho_prom, Alto_prom)    # Valor preliminar

## Esto se puede modificar para detectar los puntos, CONSERVAR
# Suma_Alto = 0;    Suma_Ancho = 0
# # Segundo ciclo
# for i in range(1,num_labels):
#     Bbox = stats[i,]
#     if (Bbox[2] < 2*Ancho_prom and Bbox[2] > 0.5*Ancho_prom):
#         if (Bbox[3] < 2*Alto_prom and Bbox[3] > 0.5*Alto_prom):
#             # print (Bbox)
#             Suma_Ancho = Suma_Ancho + Bbox[2] 
#             Suma_Alto = Suma_Alto + Bbox[3]
#     # Si el ancho o alto estan muy alejados del promedio, se ignoran

# Ancho_prom = Suma_Ancho/num_labels
# Alto_prom = Suma_Alto/num_labels

# print(Ancho_prom, Alto_prom)    # Valor final

matlab.imshow(Img_color, title = 'Matriz Etiqueta RGB')

Palabras = matlab.imfill(Palabras)
kernel = np.ones((8,8),np.uint8)    # square (5,5)
Palabras = cv2.dilate(Palabras, kernel, iterations = 1)
matlab.imshow(Palabras, title = 'Palabras 1')  # Parece prometedor




## ------------------------ Deteccion de Texto ------------------------- #

dil_palabra = 0.3;  dil_parrafo = 4

Img_palabra = matlab.Deteccion_texto(Img_BW, Ancho_prom, Alto_prom, dil_palabra)
matlab.imshow(Img_palabra, title = 'Palabras detectadas')

Img_parrafo = matlab.Deteccion_texto(Img_BW, Ancho_prom, Alto_prom, dil_parrafo)
matlab.imshow(Img_parrafo, title = 'Parrafos detectados')

# # Funcion Deteccion_texto
# def Deteccion_texto(Imagen_BW, Ancho_prom, Alto_prom, proporcion):

proporcion = dil_parrafo

Imagen_BW = matlab.imfill(Img_BW)

Ancho_kernel = int( Ancho_prom*proporcion )
Alto_kernel = int( Alto_prom*proporcion )

kernel = np.ones((Ancho_kernel,Alto_kernel),np.uint8)    # square (5,5)
Deteccion = cv2.dilate(Imagen_BW, kernel, iterations = 1)
Deteccion = matlab.imfill(Deteccion)
# El tercer argumento es opcional y por defecto es 1

Output = cv2.connectedComponentsWithStats(Deteccion, 8, cv2.CV_32S)

# Resultados
num_labels =    Output[0]   # Cantidad de elementos
labels =        Output[1]   # Matriz con etiquetas

# Coloreamos los elementos
Img_color = matlab.label2rgb(labels,num_labels)

# ## ----------------------------- Rotacion ------------------------------ #
# Ang_rotacion = -Angulo
# Rotada = rotate_image(Img_BW, Ang_rotacion)
# imshow(Rotada, title = 'Imagen Rotada')

## ------------------------ Aproximar contorno ------------------------- #
Img_gris2 = cv2.cvtColor(Img_color,cv2.COLOR_BGR2GRAY)

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

        # # Se obtienen las esquinas del poligono aproximante
        # per = cv2.arcLength(cnt,True)
        # Puntos = cv2.approxPolyDP(cnt,0.01*per,True,)
        # cv2.polylines(Img_color, [Puntos], True, (255,0,0), 15 )

        # Envolvente convexa
        Hull = cv2.convexHull(cnt)
        Hull = np.resize(Hull,( int( np.size(Hull)/2 ), 2))   # Formateo
        cv2.drawContours(Img_color,[Hull],0,(255,0,255),5 )
        for points in Hull:
            cv2.circle(Img_color,(points[0],points[1]),5,(255,0,0),5)

        # Bounding box
        Bbox = cv2.boundingRect(cnt)
        cv2.rectangle(Img_color, (Bbox[0], Bbox[1]), (Bbox[0]+Bbox[2],
        Bbox[1]+Bbox[3]), (0,255,0), 5)

        # Minimo Rectangulo contenedor de la figura
        Rect = cv2.minAreaRect(cnt)
        points = cv2.boxPoints(Rect)
        Box = np.int32(points)
        # Dibujar Rectangulo de minima area que abarca al texto
        cv2.drawContours(Img_color,[Box],0,(0,0,255),5 )

        for points in Hull:
            cv2.circle(Img_color,(points[0],points[1]),5,(255,0,0),10)

# Imprimir resultados para la caja, funca relativamente bien, faltan ajustes
print(Box, Box.shape)
print(Bbox)
# Hull = np.resize(Hull,(np.int( np.size(Hull)/2 ), 2)) 
print(Hull, Hull.shape )
matlab.imshow(Img_color, title = 'Imagen Marcada')

## --------------------------------------------------------------------- #


