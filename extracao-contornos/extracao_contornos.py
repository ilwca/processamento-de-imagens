import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


if os.environ.get("DISPLAY", "") == "" and os.environ.get("WAYLAND_DISPLAY", "") == "":
    import matplotlib
    matplotlib.use("Agg")


# Extração de contornos (Aula 11 - Morfologia Matemática)
#
#     β(A) = A − (A ⊖ B)
#
# O contorno de A é o que "sobra" quando retiramos de A a sua erosão.
# Convenção: 1 = objeto (pixel de A), 0 = fundo.

# Elemento estruturante B: quadrado 3x3 com origem no centro
ELEMENTO_ESTRUTURANTE = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
])

# Conjunto A da Figura 9 dos slides (a)
CONJUNTO_EXEMPLO = np.array([
    [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
])


def expandir_com_fundo(imagem, raio):
    # Cria uma moldura de "raio" pixels ao redor da imagem preenchida com 0,
    # ou seja, tudo que está fora da imagem é considerado fundo
    altura, largura = imagem.shape
    imagem_expandida = np.zeros((altura + 2 * raio, largura + 2 * raio), dtype=np.uint8)
    imagem_expandida[raio:raio + altura, raio:raio + largura] = imagem
    return imagem_expandida


def erosao(imagem, elemento):
    # A ⊖ B = { x | (B)x ⊆ A }
    # Um pixel x permanece na erosão somente se o elemento estruturante,
    # centralizado em x, couber totalmente dentro do objeto
    altura, largura = imagem.shape
    tamanho = elemento.shape[0]
    raio = tamanho // 2

    img_expandida = expandir_com_fundo(imagem, raio)
    img_erodida = np.zeros((altura, largura), dtype=np.uint8)

    for i in range(altura):
        for j in range(largura):
            vizinhanca = img_expandida[i:i + tamanho, j:j + tamanho]

            # Verifica se B "cabe" em A: toda posição de B que vale 1
            # precisa encontrar um pixel de objeto (1) na vizinhança
            cabe = True
            for bi in range(tamanho):
                for bj in range(tamanho):
                    if elemento[bi, bj] == 1 and vizinhanca[bi, bj] == 0:
                        cabe = False

            if cabe:
                img_erodida[i, j] = 1

    return img_erodida


def diferenca(conjunto_a, conjunto_b):
    # A − B: pixels que pertencem a A e NÃO pertencem a B
    altura, largura = conjunto_a.shape
    resultado = np.zeros((altura, largura), dtype=np.uint8)

    for i in range(altura):
        for j in range(largura):
            if conjunto_a[i, j] == 1 and conjunto_b[i, j] == 0:
                resultado[i, j] = 1

    return resultado


def extrair_contorno(imagem, elemento):
    # β(A) = A − (A ⊖ B)
    img_erodida = erosao(imagem, elemento)
    contorno = diferenca(imagem, img_erodida)
    return img_erodida, contorno


def binarizar(imagem, limiar):
    # Pixels com intensidade >= limiar viram objeto (1), os demais fundo (0)
    return (imagem >= limiar).astype(np.uint8)


def mostrar_matriz(titulo, matriz):
    print(titulo)
    for linha in matriz:
        print("  " + " ".join("■" if valor == 1 else "·" for valor in linha))
    print()


def plotar_linha(posicao, imagens, titulos, total_linhas):
    for coluna, (img, titulo) in enumerate(zip(imagens, titulos)):
        plt.subplot(total_linhas, 3, posicao * 3 + coluna + 1)
        plt.imshow(img, cmap="gray", vmin=0, vmax=1)
        plt.title(titulo)
        plt.axis("off")


# Exemplo 1: conjunto pequeno dos slides (Figura 9)
erosao_exemplo, contorno_exemplo = extrair_contorno(CONJUNTO_EXEMPLO, ELEMENTO_ESTRUTURANTE)

mostrar_matriz("(a) Conjunto original A:", CONJUNTO_EXEMPLO)
mostrar_matriz("(c) Erosão A ⊖ B:", erosao_exemplo)
mostrar_matriz("(d) Contorno β(A) = A − (A ⊖ B):", contorno_exemplo)

# Exemplo 2: imagem real binarizada
imagem = Image.open("extracao-contornos/img/contorno.png").convert("L")
img_binaria = binarizar(np.array(imagem), limiar=128)
erosao_imagem, contorno_imagem = extrair_contorno(img_binaria, ELEMENTO_ESTRUTURANTE)

titulos = ["A (original)", "A ⊖ B (erosão)", "β(A) = A − (A ⊖ B)"]

plt.figure(figsize=(12, 8))
plotar_linha(0, [CONJUNTO_EXEMPLO, erosao_exemplo, contorno_exemplo], titulos, 2)
plotar_linha(1, [img_binaria, erosao_imagem, contorno_imagem], titulos, 2)

plt.tight_layout()
plt.savefig(
    "extracao-contornos/output/extracao_contornos.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Resultado salvo em: extracao-contornos/output/extracao_contornos.png")
