from  mtcnn import MTCNN
from PIL import Image
from os import listdir
from os.path import isdir
from numpy import asarray

detector = MTCNN()
def extrair_face(arquivo, size=(160,160)):

    img =  Image.open(arquivo) # Caminho completo

    img =  img.convert('RGB') # Converter em RGB

    array = asarray(img)

    results = detector.detect_faces(array)
    x1, y1 , width, height = results[0] ['box']

    x2, y2 = x1 + width, y1 + height # dimensiona o tamanho da face

    face = array[y1:y2,x1:x2]

    image = Image.fromarray(face) # mudar o Array para o tamanho padrao
    image = image.resize(size)

    return image