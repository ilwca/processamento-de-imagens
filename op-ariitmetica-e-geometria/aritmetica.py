# 2 operações aritméticas: adição, subtração, divisão, multiplicação.

import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


if os.environ.get("DISPLAY", "") == "" and os.environ.get("WAYLAND_DISPLAY", "") == "":
    import matplotlib
    matplotlib.use("Agg")


def soma(img1, img2):
    matriz1 = np.array(img1, dtype=np.int16)
    matriz2 = np.array(img2, dtype=np.int16)

    resultado = matriz1 + matriz2
    resultado = np.clip(resultado, 0, 255).astype(np.uint8)

    return resultado

def subtracao(img1, img2):
    matriz1 = np.array(img1, dtype=np.int16)
    matriz2 = np.array(img2, dtype=np.int16)

    resultado = matriz1 - matriz2
    resultado = np.clip(resultado, 0, 255).astype(np.uint8)

    return resultado

img1 = Image.open("Op. Aritmetica e Geometrica/img/flor01.jpg").convert("L")
img2 = Image.open("Op. Aritmetica e Geometrica/img/flor02.jpg").convert("L")

matriz = np.array(img1)
matriz = np.array(img2)

soma = soma(img1,img2)
subtracao = subtracao(img1,img2)

plt.figure(figsize=(10, 10))

plt.subplot(2, 2, 1)
plt.imshow(img1, cmap="gray")
plt.title("Imagem 1")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(img2, cmap="gray")
plt.title("Imagem 2")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(soma, cmap="gray")
plt.title("Resultado soma: img1 + img2")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(subtracao, cmap="gray")
plt.title("Resultado subtração: img1 - img2")
plt.axis("off")


plt.tight_layout()
plt.savefig("Op. Aritmetica e Geometrica/output/aritmetica.png", dpi=300)

if plt.get_backend().lower() != "agg":
    plt.show()