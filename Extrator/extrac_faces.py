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


def load_fotos(directory_src, directory_target):

    print(directory_src)
    print(directory_target)


def load_dir(directoy_src, directory_target):
    for subdir in listdir(directoy_src):

        path = directoy_src + subdir + "\\"

        path_tg = directory_target + subdir + "\\"

        if not isdir(path):
            continue

        load_fotos(path, path_tg)

if __name__ == '__main__':
    load_dir("C:\\programas - TI\\DATA_SETS\\fotos-Tonye Cintia",
             "C:\\programas - TI\\DATA_SETS\\Faces_family")
