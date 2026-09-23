import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


if os.environ.get("DISPLAY", "") == "" and os.environ.get("WAYLAND_DISPLAY", "") == "":
    import matplotlib
    matplotlib.use("Agg")


NIVEIS = 256


def montar_histograma(imagem):
    histograma = np.zeros(NIVEIS, dtype=int)

    for linha in imagem:
        for pixel in linha:
            histograma[pixel] += 1

    return histograma


def calcular_soma_normalizada(histograma, total_pixels):
    histograma_normalizado = histograma / total_pixels

    soma_normalizada = np.zeros(NIVEIS, dtype=float)
    acumulado = 0.0

    for k in range(NIVEIS):
        acumulado += histograma_normalizado[k]
        soma_normalizada[k] = acumulado

    return soma_normalizada


def criar_lookup_table(soma_normalizada):
    lut = np.round(soma_normalizada * (NIVEIS - 1)).astype(np.uint8)

    return lut


def aplicar_lookup_table(imagem, lut):
    altura, largura = imagem.shape
    imagem_saida = np.zeros_like(imagem)

    for i in range(altura):
        for j in range(largura):
            imagem_saida[i, j] = lut[imagem[i, j]]

    return imagem_saida


def equalizar_histograma(imagem):
    img = np.array(imagem)
    total_pixels = img.shape[0] * img.shape[1]

    histograma = montar_histograma(img)
    soma_normalizada = calcular_soma_normalizada(histograma, total_pixels)
    lut = criar_lookup_table(soma_normalizada)
    imagem_equalizada = aplicar_lookup_table(img, lut)

    return Image.fromarray(imagem_equalizada), histograma


imagem = Image.open("equalizacao-histograma/img/resized-pout.jpg").convert("L")

imagem_equalizada, histograma_original = equalizar_histograma(imagem)

histograma_equalizado = montar_histograma(np.array(imagem_equalizada))

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(imagem, cmap="gray", vmin=0, vmax=255)
plt.title("Imagem original")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.bar(range(NIVEIS), histograma_original, color="gray", width=1)
plt.title("Histograma original")
plt.xlabel("Nível de cinza")
plt.ylabel("Frequência")

plt.subplot(2, 2, 3)
plt.imshow(imagem_equalizada, cmap="gray", vmin=0, vmax=255)
plt.title("Imagem equalizada")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.bar(range(NIVEIS), histograma_equalizado, color="gray", width=1)
plt.title("Histograma equalizado")
plt.xlabel("Nível de cinza")
plt.ylabel("Frequência")

plt.tight_layout()
plt.savefig(
    "equalizacao-histograma/output/comparacao.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Imagem equalizada salva em: equalizacao-histograma/output/comparacao.png")
