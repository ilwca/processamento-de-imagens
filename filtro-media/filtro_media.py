import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


if os.environ.get("DISPLAY", "") == "" and os.environ.get("WAYLAND_DISPLAY", "") == "":
    import matplotlib
    matplotlib.use("Agg")


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


def filtro_media(imagem, tamanho_kernel):
    raio = tamanho_kernel // 2
    img = np.array(imagem, dtype=np.float64)
    altura, largura = img.shape

    img_expandida = expandir_borda(img, raio)
    img_filtrada = np.zeros_like(img)

    for i in range(altura):
        for j in range(largura):
            vizinhanca = img_expandida[i:i + tamanho_kernel, j:j + tamanho_kernel]
            img_filtrada[i, j] = vizinhanca.mean()

    return np.clip(img_filtrada, 0, 255).astype(np.uint8)


imagem = Image.open("filtro-media/img/lena.jpg").convert("L")

tamanhos = [3, 5, 9, 15, 35]
resultados = {tamanho: filtro_media(imagem, tamanho) for tamanho in tamanhos}

plt.figure(figsize=(14, 8))

plt.subplot(2, 3, 1)
plt.imshow(imagem, cmap="gray", vmin=0, vmax=255)
plt.title("Imagem original")
plt.axis("off")

for indice, tamanho in enumerate(tamanhos, start=2):
    plt.subplot(2, 3, indice)
    plt.imshow(resultados[tamanho], cmap="gray", vmin=0, vmax=255)
    plt.title(f"Filtro da média {tamanho}x{tamanho}")
    plt.axis("off")

plt.tight_layout()
plt.savefig(
    "filtro-media/output/comparacao.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Imagens filtradas salvas em: filtro-media/output/comparacao.png")
