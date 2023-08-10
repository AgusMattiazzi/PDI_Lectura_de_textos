# Librerias por defecto
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Gracias a Murtaza's Workshop - Robotics and AI, sin su ayuda este
# archivo quiza no existiria

## ---------------------------------------- Funcion imshow -------------------------------------- #
# Equivalente a la función imshow de matlab
# Parametros de entrada:
#   Img: Imagen a mostrar
#   Title: Título a mostrar
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

## ---------------------------------------------------------------------------------------------- #



## ------------------------------------ Funcion imclearborder ----------------------------------- #
# Funcion imclearborder: (equivalente a imclearborder de MATLAB), elimina los bordes blancos en
# una imagen binaria.
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

## ---------------------------------------------------------------------------------------------- #


## ---------------------------------------- Funcion imfill -------------------------------------- #
# Funcion equivalente a imfill de MATLAB. Llena los huecos en cualquier imagen binaria
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

## ---------------------------------------------------------------------------------------------- #



## -------------------------------------- Funcion label2rgb ------------------------------------- #
# Funcion equivalente a label2rgb de MATLAB. A partir de la matriz de etiquetas, forma una imagen
# RGB. Se necesita el número de etiquetas 

# Parámetros de entrada:
#   labels: Matriz de etiquetas correspondiente a la imagen
#   N_label: Número de etiquetas en la matriz

def label2rgb(labels,N_label):
    # Mascara logica con los pixeles correspondientes al fondo
    Fondo = labels == 0

    # Se debe convertir la matriz para poder aplicarle el mapa de color
    labels = np.uint8( (255*labels)/N_label )
    # Convierte la matriz etiqueta a RGB
    Img_color = cv2.applyColorMap(labels, 4)

    # Usa la mascara Fondo para cambiar el color de fondo
    Img_color[Fondo] = (0,0,0)

    return Img_color

## ---------------------------------------------------------------------------------------------- #


## --------------------------------- Funcion dilatacion_especial -------------------------------- #
# Efectúa una dilatación usando los valores promedio de las letras, y una constante de proporción
# que depende de si se quiere agrupar palabras, párrafos, o si se quiere obtener la región de
# interés del texto completo.
def dilatacion_especial(Imagen_BW, Ancho_prom, Alto_prom, proporcion):
    Imagen_BW = imfill(Imagen_BW)
    
    Ancho_kernel = int( Ancho_prom*proporcion )
    Alto_kernel = int( Alto_prom*proporcion )

    kernel = np.ones((Ancho_kernel,Alto_kernel),np.uint8)
    Deteccion = cv2.dilate(Imagen_BW, kernel, iterations = 1)
    Deteccion = imfill(Deteccion)
    # El tercer argumento es opcional y por defecto es 1

    return Deteccion

## ---------------------------------------------------------------------------------------------- #


## --------------------------------------- Función expand --------------------------------------- #
# Expande la imagen para que contenga al mínimo rectángulo que contiene al texto.

# Parámetros de entrada:
#   Img: La imagen en cuestión
#   Box: Cordenadas del rectángulo

def expand(Img,Box):
    Dim = list(reversed(Img.shape[:2]))

    [Left,Right,Top,Bottom] = [0,0,0,0]

    for i in range (0,3):
        if Box[i][0] < 0 and abs(Box[i][0]) > Left:
            Left = abs(Box[i][0])
        if Box[i][0] > Dim[0] and abs(Box[i][0]) > Right:
            Right = abs(Box[i][0])

        if Box[i][1] < 0 and abs(Box[i][1]) > Top:
            Top = abs(Box[i][1])
        if Box[i][1] > Dim[1] and abs(Box[i][1]) > Bottom:
            Bottom = abs(Box[i][1])

    Img_out = cv2.copyMakeBorder(Img, Top, Bottom, Left, Right, cv2.BORDER_CONSTANT, value = 0)
    return Img_out

## ---------------------------------------------------------------------------------------------- #


## --------------------------------- Aproximar región de interes -------------------------------- #
# Aproxima el contorno del texto y devuelve los 4 puntos de un polígono irregular que lo contiene.
# Esto es generado por chatgpt y me dió un buen par de ideas, pero aún no está probado.

# Parámetros de entrada:
#   Img_path: Path de la imagen. Se abre la imagen dentro de la función
# Parámetros de salida:
#   approx_polygons: Coordenadas del polígono aproximado que contiene al texto

def approx_contour(Img_path) -> list:
    image = cv2.imread(Img_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, 25,255)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    approx_polygons = []
    for contour in contours:
        epsilon = 0.04 * cv2.arcLength(contour, True)
        # El epsilon debe ajustarse en función de la precisión requerida
        polygon = cv2.approxPolyDP(contour, epsilon, True)
        if len(polygon) == 4:  # Filtra los polígonos con 4 puntos
            approx_polygons.append(polygon) # Agregar elementos
    cv2.drawContours(image, approx_polygons, -1, (0, 255, 0), 2) 
    # Green color and line thickness 2
    return approx_polygons

## ---------------------------------------------------------------------------------------------- #

## ----------------------------- Dimensiones promedio de caracteres ----------------------------- #
# Para efectuar la dilatación especial, primero deben obtenerse las dimensiones promedio de los
# caracteres de la imagen. Para ello se usa ésta función, tomando como parámetro de entrada las
# propiedades obtenidas a partir del algoritmo de componentes conectadas.

# Parámetros de entrada:
#   Conn_components: Resultado obtenido de la función cv2.connectedComponentsWithStats aplicado
#   a la imagen deseada

def dim_promedio(Conn_components) -> tuple:
    num_labels =    Conn_components[0]   # Cantidad de elementos
    stats =         Conn_components[2]   # Matriz de stats

    Suma_Alto = 0;    Suma_Ancho = 0    # Inicializo contadores

    # Agregamos Bounding Box
    for i in range(1,num_labels):
        Bbox = stats[i,]
        # Bbox[2] : Ancho
        # Bbox[3] : Alto

        Suma_Ancho = Suma_Ancho + Bbox[2]
        Suma_Alto = Suma_Alto + Bbox[3]

    # TODO puede estar mal, considerando siempre una etiqueta de más, la del fondo
    # El return es el ancho y alto promedio de letras.
    return Suma_Ancho/num_labels, Suma_Alto/num_labels

## ---------------------------------------------------------------------------------------------- #

## ----------------------------------- Componentes conectadas ----------------------------------- #
def resultados(Conn_components):

    num_labels =    Conn_components[0]   # Cantidad de elementos
    labels =        Conn_components[1]   # Matriz con etiquetas
    stats =         Conn_components[2]   # Matriz de stats

    Img_salida = label2rgb(labels,num_labels)

    # Agregamos Bounding Box
    for i in range(1,num_labels):
        Bbox = stats[i,]
        # Bbox[0] : Coordenada x punto superior izquierdo
        # Bbox[1] : Coordenada y punto superior izquierdo
        # Bbox[2] : Ancho
        # Bbox[3] : Alto

        cv2.rectangle(Img_salida, (Bbox[0], Bbox[1]), (Bbox[0]+Bbox[2],
            Bbox[1]+Bbox[3]), (255,255,255), 2)

    cant_elementos = (num_labels - 1) # En num_labels se cuenta el fondo también
    return Img_salida, cant_elementos

## ---------------------------------------------------------------------------------------------- #


## -------------------------------------- Analisis de texto ------------------------------------- #
def resultados_finales():
    str = 'Este es el texto que debe ser leido de manera automática. \n'
    str = str + 'El algoritmo implementado debe ser capaz de detectar todas las palabras. '
    str = str + 'Para evitar que este problema sea muy complicado '
    str = str + 'todos los caracteres estan en mayusculas y los acentos fueron omitidos. \n'
    str = str + 'Recuerde que también debe detectar los parrafos presentes en esta imagen. '
    str = str + 'Resalte mediante un recuadro de color los parrafos presentes en el texto. Como'
    str = str + ' objetivo opcional se propone detectar las oraciones presentes en el texto. \n'
    str = str + 'Para ello deberá tambien detectar los puntos que separan cada oracion. '
    str = str + 'En caso de lograrlo subraye cada oracion de texto analizado. \n'
    
    char_total = len(str)
    espacios = str.count(' ')
    puntos = str.count('.')
    parrafos = str.count('\n')
    caracteres = char_total - espacios - puntos - parrafos
    palabras = espacios # Modifique un poco el texto para tener un espacio por palabra
    print("Caracteres =", caracteres)
    print("Palabras =", palabras)
    print("Parrados =", parrafos)

## ---------------------------------------------------------------------------------------------- #