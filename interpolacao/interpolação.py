import os
from PIL import Image
import numpy as np

#Interpolação por Vizinho Mais Próximo#

if os.environ.get("DISPLAY", "") == "" and os.environ.get("WAYLAND_DISPLAY", "") == "":
    import matplotlib
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

def vizinho_mais_proximo(matriz, nova_altura, nova_largura):

    altura, largura = matriz.shape

    nova_matriz = np.zeros(
        (nova_altura, nova_largura),
        dtype=np.uint8
    )

    escala_y = altura / nova_altura
    escala_x = largura / nova_largura

    for y in range(nova_altura):
        for x in range(nova_largura):

            origem_y = int(y * escala_y)
            origem_x = int(x * escala_x)

            origem_y = min(origem_y, altura - 1)
            origem_x = min(origem_x, largura - 1)

            nova_matriz[y, x] = matriz[origem_y, origem_x]

    return nova_matriz


imagem = Image.open("./img/pikachu-peb.jpg").convert("L")
matriz = np.array(imagem)
#print("Imagem original:", matriz.shape)

# REDUÇÃO
imagem_128 = vizinho_mais_proximo(matriz,128,128)
imagem_64 = vizinho_mais_proximo(matriz,64,64)
imagem_32 = vizinho_mais_proximo(matriz,32,32)


# AMPLIAÇÃO
ampliada_32 = vizinho_mais_proximo(imagem_32,128,128)
ampliada_64 = vizinho_mais_proximo(imagem_64,256,256)
ampliada_128 = vizinho_mais_proximo(imagem_128,512,512)

plt.figure(figsize=(15, 12))

plt.subplot(3, 3, 1)
plt.imshow(matriz, cmap="gray")
plt.title("Original: 637 × 481")
plt.axis("off")

plt.subplot(3, 3, 2)
plt.imshow(imagem_128, cmap="gray")
plt.title("Redução: 128 × 128")
plt.axis("off")

plt.subplot(3, 3, 3)
plt.imshow(imagem_64, cmap="gray")
plt.title("Redução: 64 × 64")
plt.axis("off")

plt.subplot(3, 3, 4)
plt.imshow(imagem_32, cmap="gray")
plt.title("Redução: 32 × 32")
plt.axis("off")

plt.subplot(3, 3, 5)
plt.imshow(ampliada_32, cmap="gray")
plt.title("Ampliação: 32 → 128")
plt.axis("off")

plt.subplot(3, 3, 6)
plt.imshow(ampliada_64, cmap="gray")
plt.title("Ampliação: 64 → 256")
plt.axis("off")

plt.subplot(3, 3, 7)
plt.imshow(ampliada_128, cmap="gray")
plt.title("Ampliação: 128 → 512")
plt.axis("off")


plt.tight_layout()
plt.savefig("output/vizinho_mais_proximo.png", dpi=300)

if plt.get_backend().lower() != "agg":
    plt.show()

"""# Interpolação bilinear"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def interpolacao_bilinear(matriz, nova_altura, nova_largura):

    altura, largura = matriz.shape

    nova_matriz = np.zeros(
        (nova_altura, nova_largura),
        dtype=np.uint8
    )

    escala_y = altura / nova_altura
    escala_x = largura / nova_largura

    for y in range(nova_altura):
        for x in range(nova_largura):

            # Coordenada correspondente na imagem original
            origem_y = y * escala_y
            origem_x = x * escala_x

            # Coordenadas dos 4 vizinhos
            y1 = int(origem_y)
            x1 = int(origem_x)

            y2 = min(y1 + 1, altura - 1)
            x2 = min(x1 + 1, largura - 1)

            # Distâncias até os vizinhos
            dy = origem_y - y1
            dx = origem_x - x1

            Q11 = matriz[y1, x1]
            Q21 = matriz[y1, x2]
            Q12 = matriz[y2, x1]
            Q22 = matriz[y2, x2]

            valor = (
                Q11 * (1 - dx) * (1 - dy) +
                Q21 * dx * (1 - dy) +
                Q12 * (1 - dx) * dy +
                Q22 * dx * dy
            )

            nova_matriz[y, x] = int(valor)

    return nova_matriz

imagem_128 = interpolacao_bilinear(matriz,128,128)
imagem_64 = interpolacao_bilinear(matriz,64,64)
imagem_32 = interpolacao_bilinear(matriz,32,32)

ampliada_32 = interpolacao_bilinear(imagem_32,128,128)
ampliada_64 = interpolacao_bilinear(imagem_64,256,256)
ampliada_128 = interpolacao_bilinear(imagem_128,512,512)

plt.figure(figsize=(15, 12))

plt.subplot(3, 3, 1)
plt.imshow(matriz, cmap="gray")
plt.title("Original: 637 × 481")
plt.axis("off")

plt.subplot(3, 3, 2)
plt.imshow(imagem_128, cmap="gray")
plt.title("Bilinear: 128 × 128")
plt.axis("off")

plt.subplot(3, 3, 3)
plt.imshow(imagem_64, cmap="gray")
plt.title("Bilinear: 64 × 64")
plt.axis("off")

plt.subplot(3, 3, 4)
plt.imshow(imagem_32, cmap="gray")
plt.title("Bilinear: 32 × 32")
plt.axis("off")

plt.subplot(3, 3, 5)
plt.imshow(ampliada_32, cmap="gray")
plt.title("Bilinear: 32 → 128")
plt.axis("off")

plt.subplot(3, 3, 6)
plt.imshow(ampliada_64, cmap="gray")
plt.title("Bilinear: 64 → 256")
plt.axis("off")

plt.subplot(3, 3, 7)
plt.imshow(ampliada_128, cmap="gray")
plt.title("Bilinear: 128 → 512")
plt.axis("off")


plt.tight_layout()
plt.savefig("output/interpolacao_bilinear.png", dpi=300)

if plt.get_backend().lower() != "agg":
    plt.show()