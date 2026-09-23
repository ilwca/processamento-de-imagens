import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


if os.environ.get("DISPLAY", "") == "" and os.environ.get("WAYLAND_DISPLAY", "") == "":
    import matplotlib
    matplotlib.use("Agg")


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
