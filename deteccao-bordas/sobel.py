import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


if os.environ.get("DISPLAY", "") == "" and os.environ.get("WAYLAND_DISPLAY", "") == "":
    import matplotlib
    matplotlib.use("Agg")


MASCARA_GX = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1],
])

MASCARA_GY = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1],
])


def limitar(valor, minimo, maximo):
    # Mantém o valor dentro do intervalo [minimo, maximo]
    return max(minimo, min(valor, maximo))


def expandir_borda(imagem, raio):
    # Cria uma moldura de "raio" pixels ao redor da imagem,
    # repetindo o pixel de borda mais próximo
    altura, largura = imagem.shape
    nova_altura = altura + 2 * raio
    nova_largura = largura + 2 * raio

    imagem_expandida = np.zeros((nova_altura, nova_largura), dtype=np.float64)

    for i in range(nova_altura):
        for j in range(nova_largura):
            # Posição correspondente na imagem original
            linha_original = i - raio
            coluna_original = j - raio

            # Se a posição cair fora da imagem, usa a borda mais próxima
            linha_original = limitar(linha_original, 0, altura - 1)
            coluna_original = limitar(coluna_original, 0, largura - 1)

            imagem_expandida[i, j] = imagem[linha_original, coluna_original]

    return imagem_expandida


def convolucionar(imagem, mascara):
    altura, largura = imagem.shape
    tamanho = mascara.shape[0]
    raio = tamanho // 2

    img_expandida = expandir_borda(imagem, raio)
    img_filtrada = np.zeros((altura, largura), dtype=np.float64)

    for i in range(altura):
        for j in range(largura):
            vizinhanca = img_expandida[i:i + tamanho, j:j + tamanho]
            img_filtrada[i, j] = np.sum(vizinhanca * mascara)

    return img_filtrada


def escalar_para_exibicao(imagem_filtrada):
    # Reescala os valores (que podem ser negativos) para o intervalo [0, 255]
    minimo = imagem_filtrada.min()
    maximo = imagem_filtrada.max()

    resultado = (imagem_filtrada - minimo) / (maximo - minimo) * 255

    return resultado.astype(np.uint8)


imagem = Image.open("deteccao-bordas/img/lena.jpg").convert("L")
img = np.array(imagem, dtype=np.float64)

gx = convolucionar(img, MASCARA_GX)
gy = convolucionar(img, MASCARA_GY)
gradiente = np.sqrt(gx ** 2 + gy ** 2)

plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
plt.imshow(imagem, cmap="gray", vmin=0, vmax=255)
plt.title("Imagem original")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(escalar_para_exibicao(gx), cmap="gray", vmin=0, vmax=255)
plt.title("Gradiente Gx (Sobel horizontal)")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(escalar_para_exibicao(gy), cmap="gray", vmin=0, vmax=255)
plt.title("Gradiente Gy (Sobel vertical)")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(escalar_para_exibicao(gradiente), cmap="gray", vmin=0, vmax=255)
plt.title("Magnitude do gradiente")
plt.axis("off")

plt.tight_layout()
plt.savefig(
    "deteccao-bordas/output/sobel.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Imagens filtradas salvas em: deteccao-bordas/output/sobel.png")
