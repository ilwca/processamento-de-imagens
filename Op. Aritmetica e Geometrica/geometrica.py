#1 operação geométrica: rotação, translação, espelhamento ou reflexão

import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


if os.environ.get("DISPLAY", "") == "" and os.environ.get("WAYLAND_DISPLAY", "") == "":
    import matplotlib
    matplotlib.use("Agg")

def rotacao(img1):
    matriz = np.array(img1)

    altura, largura = matriz.shape

    resultado = np.zeros((largura, altura), dtype=np.uint8)

    for i in range(altura):
        for j in range(largura):
            resultado[j, altura - 1 - i] = matriz[i, j]

    return resultado


img1 = Image.open("Op. Aritmetica e Geometrica/img/desenho1.jpg").convert("L")

matriz = np.array(img1)

rot = rotacao(img1)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(img1, cmap="gray")
plt.title("Imagem original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(rot, cmap="gray")
plt.title("Rotação")
plt.axis("off")


plt.tight_layout()
plt.savefig("Op. Aritmetica e Geometrica/output/geometrica.png", dpi=300)

if plt.get_backend().lower() != "agg":
    plt.show()