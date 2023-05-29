# AAPV_Semantic_Segmentation
Proyecto final para la asignatura "Arquitectura de Altas Prestaciones para Visión" del Máster de Ciencia de Datos e Ingeniería de Computadores de la Universidad de Granada.

# Entrenamiento
En este proyecto se ha usado PyTorch y la librería de Python llamada "Segmentation Models PyTorch', junto a esta librería se usa el paquete 'albumentations' para realizar una "aumentación" (augmentation) del dataset 'CamVid'.

Los notebooks de entrenamiento (Training_X_Model.ipynb) se basan en uno de los ejemplos del repositorio de 'Segmentation Models PyTorch'

# Scripts de inferencia
Los archivos que empiezan por 'script_' son archivos creados, sobretodo, para entender cómo funciona la inferencia con las dos librerías anteriormente mencionadas (Segmentation Models PyTorch y Albumentations) y para adaptar el código con la intención de no usar el paquete Albumentations, paquete que ha sido imposible instalarlo en la Nvidia Jetson Nano.

# Notebooks de inferencia
Los notebooks que empiezan por 'Inference_' es la inferencia de los modelos ya preparada para ser ejecutada en la Jetson Nano.

# Dockerfile
El archivo Dockerfile sirve para crear una imagen pensada para la Jetson Nano que contiene PyTorch, los drivers de Nvidia (CUDA) para ejecutar en la GPU, la librería "Segmentation Models PyTorch" y Matplotlib, para poder visualizar los resultados de la inferencia en la propia interfaz de Jupyter.

# Datasets utilizados
Dentro del directorio "data" nos encontramos con todo lo necesario para entrenar e inferir, en el subdirectorio 'CamVid' se encuentran las imágenes del dataset 'CamVid' para entrenar los distintos modelos.

En el directorio 'camera_lidar_semantic' tenemos algunos ejemplos de imágenes para inferir que contienen las máscaras para la segmentación, de manera que podemos medir el IoU (Intersection over function). Estas imágenes se han obtenido del dataset A2D2, creado por Audi para ayudar a corporaciones y universidades a desarrollar y probar modelos para conducción autónoma.

También nos encontramos con dos videos de prueba para poder probar la red y ver su rendimiento en tiempo real.

# Créditos

## Segmentation models PyTorch
```
@misc{Iakubovskii:2019,
  Author = {Pavel Iakubovskii},
  Title = {Segmentation Models Pytorch},
  Year = {2019},
  Publisher = {GitHub},
  Journal = {GitHub repository},
  Howpublished = {\url{https://github.com/qubvel/segmentation_models.pytorch}}
}
```

## Albumentations
```
@Article{info11020125,
    AUTHOR = {Buslaev, Alexander and Iglovikov, Vladimir I. and Khvedchenya, Eugene and Parinov, Alex and Druzhinin, Mikhail and Kalinin, Alexandr A.},
    TITLE = {Albumentations: Fast and Flexible Image Augmentations},
    JOURNAL = {Information},
    VOLUME = {11},
    YEAR = {2020},
    NUMBER = {2},
    ARTICLE-NUMBER = {125},
    URL = {https://www.mdpi.com/2078-2489/11/2/125},
    ISSN = {2078-2489},
    DOI = {10.3390/info11020125}
}
```

## A2D2: Audi Autonomous Driving Dataset
```
@article{geyer2020a2d2,
    title = {{A2D2: Audi Autonomous Driving Dataset}},
    author = {Jakob Geyer and Yohannes Kassahun and Mentar Mahmudi and Xavier Ricou and Rupesh Durgesh and Andrew S. Chung and Lorenz Hauswald and Viet Hoang Pham and Maximilian M{\"u}hlegg and Sebastian Dorn and Tiffany Fernandez and Martin J{\"a}nicke and Sudesh Mirashi and Chiragkumar Savani and Martin Sturm and Oleksandr Vorobiov and Martin Oelker and Sebastian Garreis and Peter Schuberth},
    year = {2020},
    eprint = {2004.06320},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV},
    url = {https://www.a2d2.audi}
}
```

## Motion-based Segmentation and Recognition Dataset


(1)
Segmentation and Recognition Using Structure from Motion Point Clouds, ECCV 2008 (pdf)
Brostow, Shotton, Fauqueur, Cipolla (bibtex)


(2)
Semantic Object Classes in Video: A High-Definition Ground Truth Database (pdf)
Pattern Recognition Letters (to appear)
Brostow, Fauqueur, Cipolla (bibtex)

## Video.mp4: 4K Video of Highway traffic!
https://www.youtube.com/watch?v=KBsqQez-O4w

## Video_2.mp4
<blockquote class="twitter-tweet"><p lang="es" dir="ltr">Persecución policial en Osasco (Sao Paulo🇧🇷). Tanto el fugitivo como el policía podrían competir en Moto GP. <a href="https://t.co/BzL0bTBdoT">pic.twitter.com/BzL0bTBdoT</a></p>&mdash; Niporwifi © (@niporwifi) <a href="https://twitter.com/niporwifi/status/1658455223239680001?ref_src=twsrc%5Etfw">May 16, 2023</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>