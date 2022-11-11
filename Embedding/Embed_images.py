from  mtcnn import MTCNN
from PIL import Image
from os import listdir
from os.path import isdir
from numpy import asarray, expand_dims
import numpy as np

def load_face(filename): #, required_size =(160,160)):
    #carregando arquivo
    image = Image.open(filename)

    #converter par rgb

    image = image.convert("RGB")

    return asarray(image)

### CARREGANDO AS FACES DO DIRETORIO

def load_faces(directory_src):

    faces = list()
# Iterando arquivos
    for filename in listdir(directory_src):
        path = directory_src + filename

        faces.append(load_face(path))

        load_face(path)