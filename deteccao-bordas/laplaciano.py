import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


if os.environ.get("DISPLAY", "") == "" and os.environ.get("WAYLAND_DISPLAY", "") == "":
    import matplotlib
    matplotlib.use("Agg")


MASCARAS = {
    "M1 (4-viz., centro -4)": np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0],
    ]),
    "M2 (8-viz., centro -8)": np.array([
        [1, 1, 1],
        [1, -8, 1],
        [1, 1, 1],
    ]),
    "M3 (4-viz., centro 4)": np.array([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0],
    ]),
    "M4 (8-viz., centro 8)": np.array([
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1],
    ]),
}


def expandir_borda(imagem, raio):
    altura, largura = imagem.shape
    imagem_expandida = np.zeros((altura + 2 * raio, largura + 2 * raio), dtype=np.float64)
    imagem_expandida[raio:raio + altura, raio:raio + largura] = imagem

    # Replicação dos pixels de borda
    for i in range(raio):
        imagem_expandida[i, raio:raio + largura] = imagem[0, :]
        imagem_expandida[raio + altura + i, raio:raio + largura] = imagem[-1, :]

    for j in range(raio):
        imagem_expandida[:, j] = imagem_expandida[:, raio]
        imagem_expandida[:, raio + largura + j] = imagem_expandida[:, raio + largura - 1]

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

resultados = {
    nome: escalar_para_exibicao(convolucionar(img, mascara))
    for nome, mascara in MASCARAS.items()
}

plt.figure(figsize=(14, 8))

plt.subplot(2, 3, 1)
plt.imshow(imagem, cmap="gray", vmin=0, vmax=255)
plt.title("Imagem original")
plt.axis("off")

for indice, (nome, resultado) in enumerate(resultados.items(), start=2):
    plt.subplot(2, 3, indice)
    plt.imshow(resultado, cmap="gray", vmin=0, vmax=255)
    plt.title(nome)
    plt.axis("off")

plt.tight_layout()
plt.savefig(
    "deteccao-bordas/output/laplaciano.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Imagens filtradas salvas em: deteccao-bordas/output/laplaciano.png")
