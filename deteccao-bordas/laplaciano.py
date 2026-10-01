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
