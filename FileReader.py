import xarray as xr
import numpy as np

from PIL import Image
from numpy import reshape

##########################################################################################
# Se pueden recuperar las imagenes y features guardados con estos ficheros de esta manera:
# import pandas as pd
# import numpy as np
# images = np.load('data/images.npz', 'r', True)['arr_0']
# features = pd.read_csv('data/cloud_features.csv', sep=';')
##########################################################################################

# ORDEN DE LECTURA CORRECTO
# "TSIdatasetToCarlosIIISet600_20160609"
# "TSIdatasetToCarlosIII_diasVeranoAerosol"
# "bbdd_images_2015_SEP"
# "bbdd_images_2015_OCT"
# "bbdd_images_2015_JUN"
# "bbdd_images_2015_JUL"
# "bbdd_images_2015_AGO"

# Tamanyo estandar de la foto
stdwidth = (256, 256)

# Ficheros en el orden original,
# IMPORTANTE: Coincide con los datos de nubes del csv.
files = ["TSIdatasetToCarlosIIISet600_20160609",
         "TSIdatasetToCarlosIII_diasVeranoAerosol",
         "bbdd_images_2015_SEP",
         "bbdd_images_2015_OCT",
         "bbdd_images_2015_JUN",
         "bbdd_images_2015_JUL",
         "bbdd_images_2015_AGO"]


# Lee un fichero con el formato 1, para los ficheros SIN 3 camaras.
# Forma original= (i, 256, 256, 3).
# Devuelve un nparray de shape (i, 256, 256, 3) donde i es el numero de imagen
# Forma de salida= Forma original.
def read_format1(fn):
    DS = xr.open_dataset("data/%s.nc" % fn)
    data = DS.imgsProy.values
    return data

# Lee un fichero con el formato 1, para los ficheros SIN 3 camaras.
# Forma original= (i, 3 , 500, 500, 3).
# Devuelve un nparray de shape (i * 3, 256, 256, 3) donde i es el numero de imagenes originales
# Forma de salida= (i*3, 256, 256, 3).
def read_format2(fn, resize):
    DS = xr.open_dataset("data/%s.nc" % fn)
    data = DS.imagesProjected.values
    i, cam, row, col, ch = data.shape

    # Serializar las camaras en una dimension unica
    images = reshape(data, (i * cam, row, col, ch))

    newHeight, newWidth = resize

    # Cambiar el tamanyo de cada imagen
    images_resized = np.zeros([0, newHeight, newWidth, 3], dtype=np.uint8)
    for image in range(images.shape[0]):
        temp = Image.fromarray(images[image])
        temp = temp.resize((newHeight, newWidth))
        temp = np.array(temp)
        images_resized = np.append(images_resized, np.expand_dims(temp, axis=0), axis=0)

    return images_resized


# Wrapper para los dos formatos diferentes,
# Identificar (rudimentariamente) que tipo de fichero es y extrae sus datos
# Devuelve siempre un nparray de forma (i, 256, 256, 3)
def read(fn, resize):
    print(fn)
    if "bbdd" in fn:
        data = read_format2(fn, resize)
    else:
        data = read_format1(fn)
    return data


h, w = stdwidth
ims = np.zeros([0, h, w, 3], dtype=np.uint8)

# Recuperar todos los datos
for file in files:
    dat = read(file, stdwidth)
    ims = np.append(ims, dat, axis=0)

# Guardar en un fichero comprimido en una carpeta 'data'
np.savez_compressed('data/images.npz', ims)
