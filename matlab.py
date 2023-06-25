# Librerias por defecto
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Gracias a Murtaza's Workshop - Robotics and AI, sin su ayuda este
# archivo quiza no existiria

## --------------------------- Funcion imshow -------------------------- #
def imshow(Img,title = None):
    plt.figure()
    if len(Img.shape) == 3:
        plt.imshow(Img)
    else:
        plt.imshow(Img, cmap='gray')

    plt.title(title)
    plt.xticks([]), plt.yticks([])
    plt.show()
    # plt.show() espera a que la imagen se cierre para proseguir

## --------------------------------------------------------------------- #



## ----------------------- Funcion imclearborder ----------------------- #
# Funcion imclearborder (equivalente a imclearborder de MATLAB)
def imclearborder(Img_BW):
    # Es importante insertar una imagen blanco y negro, pero voy a
    # tratar de adaptarlo asi no me complico la vida
    Blanco = np.max(Img_BW)

    Img_clear = Img_BW.copy()       # Copio la imagen
    Alto,Ancho = Img_BW.shape[:2]      # Obtengo las dimensiones

    # El tamaño debe ser dos pixeles mayor al de la imagen original
    mask = np.zeros((Alto+2, Ancho+2), np.uint8)

    # Recorro los extremos de la imagen, aplicando floodfill (llenado
    # por difusion) siempre que encuentro un punto blanco

    # IMPORTANTE: El formato del punto semilla en floodfill (tercer 
    # argumento) va en el formato (x,y) y no en (fila,columna)
    for x in range(Ancho - 1):
        # Extremo superior
        if Img_clear[0,x] == Blanco:
            # Llena por desde el punto (0,x)
            cv2.floodFill(Img_clear, mask, (x,0), 0)

        # Extremo inferior
        if Img_clear[Alto-1,x] == Blanco:
            # Llena desde el punto (Alto,x)
            cv2.floodFill(Img_clear, mask, (0,Alto-1), 0)

    for y in range(Alto - 1):
        # Extremo izquierdo
        if Img_clear[y,0] == Blanco:
            # Llena desde el punto (y,0)
            cv2.floodFill(Img_clear, mask, (0,y), 0)

        # Extremo derecho
        if Img_clear[y,Ancho-1] == Blanco:
            # Llena desde el punto (y,Ancho)
            cv2.floodFill(Img_clear, mask, (Ancho-1,y), 0)

    return Img_clear

## --------------------------------------------------------------------- #



## --------------------------- Funcion imfill -------------------------- #
# Funcion equivalente a imfill de MATLAB
def imfill(Img):
# Gracias a Satya Mallick por el script
# https://learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/

    # Copia la imagen
    Img_floodfill = Img.copy()

    # Mascara usada para el llenado por difusion
    # El tamaño debe ser dos pixeles mayor al de la imagen original
    Ancho, Alto = Img.shape[:2]
    mask = np.zeros((Ancho+2, Alto+2), np.uint8)

    # Llena por difusion (floodfill) desde el punto (0, 0)
    cv2.floodFill(Img_floodfill, mask, (0,0), 255)

    # Imagen llenada invertida
    Img_floodfill_inv = cv2.bitwise_not(Img_floodfill)

    # La imagen final es la union entre la imagen original y la
    # imagen llenada invertida
    Salida = Img | Img_floodfill_inv
    return Salida

## --------------------------------------------------------------------- #



## ------------------------- Funcion label2rgb ------------------------- #
# Funcion equivalente a label2rgb de MATLAB
def label2rgb(labels,N_label,color_fondo = (0,0,0),colormap = 2):
    # Mascara logica con los pixeles correspondientes al fondo
    Fondo = labels == 0

    # Convierte la matriz etiqueta a RGB
    labels = np.uint8( (255*labels)/N_label )
    Img_color = cv2.applyColorMap(labels, colormap)

    # Usa la mascara Fondo para cambiar el color de fondo
    Img_color[Fondo] = color_fondo

    return Img_color

## --------------------------------------------------------------------- #



## ----------------------- Funcion Deteccion_ang ----------------------- #
# Funcion Deteccion_ang
def Deteccion_ang(Imagen_BW):
    Contours, Jerarquia = cv2.findContours(Imagen_BW,
                        cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    # Con esta funcion se obtienen los contornos a partir de los
    # bordes obtenidos por Canny
    Suma_ang = 0;       cont = 0

    for cnt in Contours:
        Elipse = cv2.fitEllipse(cnt)
        print(Elipse,'\n')
        # # center, axis_length and orientation of ellipse
        (Centro,Ejes,Orientacion) = Elipse
        Suma_ang = Orientacion + Suma_ang   # Sumatoria
        cont += 1                           # Contador

    return Suma_ang/cont

## --------------------------------------------------------------------- #



## ---------------------- Funcion Deteccion_texto ---------------------- #
# Funcion Deteccion_texto
def Deteccion_texto(Imagen_BW, Ancho_prom, Alto_prom, proporcion):
    Imagen_BW = imfill(Imagen_BW)
    
    Ancho_kernel = int( Ancho_prom*proporcion )
    Alto_kernel = int( Alto_prom*proporcion )

    kernel = np.ones((Ancho_kernel,Alto_kernel),np.uint8)    # square (5,5)
    Deteccion = cv2.dilate(Imagen_BW, kernel, iterations = 1)
    Deteccion = imfill(Deteccion)
    # El tercer argumento es opcional y por defecto es 1

    Angulo = Deteccion_ang(Deteccion)
    print("Angulo:", Angulo)

    Output = cv2.connectedComponentsWithStats(Deteccion, 8, cv2.CV_32S)

    # Resultados
    num_labels =    Output[0]   # Cantidad de elementos
    labels =        Output[1]   # Matriz con etiquetas

    # Coloreamos los elementos
    Img_color = label2rgb(labels,num_labels)
    return Img_color

## --------------------------------------------------------------------- #