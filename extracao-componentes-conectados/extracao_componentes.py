import os
import numpy as np
import matplotlib.pyplot as plt


if os.environ.get("DISPLAY", "") == "" and os.environ.get("WAYLAND_DISPLAY", "") == "":
    import matplotlib
    matplotlib.use("Agg")


# Extração de componentes conectados (Aula 11 - Morfologia Matemática)
#
#     X_k = (X_{k-1} ⊕ B) ∩ A        k = 1, 2, 3, ...
#
# Partindo de um ponto p conhecido (X_0 = p), dilatamos X e "recortamos"
# o resultado com A. Assim X cresce, mas somente por dentro de A.
# O algoritmo para quando X_k = X_{k-1}: X_k é o componente Y que contém p.
# Convenção: 1 = objeto (pixel de A), 0 = fundo.

# Elemento estruturante B: quadrado 3x3 com origem no centro (vizinhança-8)
ELEMENTO_ESTRUTURANTE = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
])

# Conjunto A da Figura 11 dos slides (a)
CONJUNTO_EXEMPLO = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
])

# Pixel de partida p (marcado com "0" na figura): linha 6, coluna 6
PONTO_INICIAL = (6, 6)


def expandir_com_fundo(imagem, raio):
    # Cria uma moldura de "raio" pixels ao redor da imagem preenchida com 0,
    # ou seja, tudo que está fora da imagem é considerado fundo
    altura, largura = imagem.shape
    imagem_expandida = np.zeros((altura + 2 * raio, largura + 2 * raio), dtype=np.uint8)
    imagem_expandida[raio:raio + altura, raio:raio + largura] = imagem
    return imagem_expandida


def dilatacao(imagem, elemento):
    # A ⊕ B = { x | (B)x ∩ A ≠ ∅ }
    # Um pixel x entra na dilatação se o elemento estruturante,
    # centralizado em x, tocar pelo menos um pixel do objeto
    altura, largura = imagem.shape
    tamanho = elemento.shape[0]
    raio = tamanho // 2

    img_expandida = expandir_com_fundo(imagem, raio)
    img_dilatada = np.zeros((altura, largura), dtype=np.uint8)

    for i in range(altura):
        for j in range(largura):
            vizinhanca = img_expandida[i:i + tamanho, j:j + tamanho]

            # Verifica se B "toca" A: basta uma posição de B que vale 1
            # encontrar um pixel de objeto (1) na vizinhança
            toca = False
            for bi in range(tamanho):
                for bj in range(tamanho):
                    if elemento[bi, bj] == 1 and vizinhanca[bi, bj] == 1:
                        toca = True

            if toca:
                img_dilatada[i, j] = 1

    return img_dilatada


def intersecao(conjunto_a, conjunto_b):
    # A ∩ B: pixels que pertencem a A e também a B
    altura, largura = conjunto_a.shape
    resultado = np.zeros((altura, largura), dtype=np.uint8)

    for i in range(altura):
        for j in range(largura):
            if conjunto_a[i, j] == 1 and conjunto_b[i, j] == 1:
                resultado[i, j] = 1

    return resultado


def sao_iguais(conjunto_a, conjunto_b):
    # Critério de parada: X_k = X_{k-1}
    return np.array_equal(conjunto_a, conjunto_b)


def extrair_componente(imagem, ponto_inicial, elemento):
    # Aplica X_k = (X_{k-1} ⊕ B) ∩ A até que X pare de crescer.
    # Retorna o componente e a lista com todos os X_k (para visualização).
    x_anterior = np.zeros(imagem.shape, dtype=np.uint8)
    x_anterior[ponto_inicial] = 1  # X_0 = p

    iteracoes = [x_anterior]

    while True:
        x_atual = intersecao(dilatacao(x_anterior, elemento), imagem)

        if sao_iguais(x_atual, x_anterior):
            break

        iteracoes.append(x_atual)
        x_anterior = x_atual

    return x_atual, iteracoes


def encontrar_pixel_objeto(imagem):
    # Procura (linha a linha) o primeiro pixel de objeto ainda não rotulado
    altura, largura = imagem.shape
    for i in range(altura):
        for j in range(largura):
            if imagem[i, j] == 1:
                return (i, j)
    return None


def rotular_componentes(imagem, elemento):
    # Extrai TODOS os componentes da imagem:
    #   1. escolhe um pixel de objeto qualquer como ponto inicial p;
    #   2. extrai o componente Y que contém p;
    #   3. dá um rótulo (1, 2, 3, ...) para Y e o remove da imagem;
    #   4. repete enquanto ainda houver pixels de objeto.
    restante = imagem.copy()
    rotulos = np.zeros(imagem.shape, dtype=int)
    rotulo_atual = 0

    ponto = encontrar_pixel_objeto(restante)
    while ponto is not None:
        componente, _ = extrair_componente(restante, ponto, elemento)

        rotulo_atual += 1
        rotulos[componente == 1] = rotulo_atual
        restante[componente == 1] = 0

        ponto = encontrar_pixel_objeto(restante)

    return rotulos, rotulo_atual


def criar_imagem_exemplo():
    # Imagem binária simples com algumas formas separadas por fundo
    imagem = np.zeros((80, 100), dtype=np.uint8)

    imagem[10:30, 10:35] = 1        # retângulo
    imagem[45:70, 15:25] = 1        # barra vertical
    imagem[60:70, 25:45] = 1        # base do "L"
    imagem[12:22, 60:90] = 1        # retângulo achatado

    # Círculo
    for i in range(80):
        for j in range(100):
            if (i - 52) ** 2 + (j - 72) ** 2 <= 14 ** 2:
                imagem[i, j] = 1

    # Anel (o buraco no meio não faz parte do objeto)
    for i in range(80):
        for j in range(100):
            distancia = (i - 40) ** 2 + (j - 48) ** 2
            if 6 ** 2 <= distancia <= 9 ** 2:
                imagem[i, j] = 1

    return imagem


def mostrar_iteracoes(imagem, iteracoes):
    # Imprime em cada pixel o número da iteração em que ele entrou em X,
    # igual à Figura 11 (e) dos slides
    print("Ordem de entrada dos pixels no componente (k):")
    altura, largura = imagem.shape
    for i in range(altura):
        linha = ""
        for j in range(largura):
            k_entrada = None
            for k, x_k in enumerate(iteracoes):
                if x_k[i, j] == 1:
                    k_entrada = k
                    break

            if k_entrada is not None:
                linha += f" {k_entrada}"
            elif imagem[i, j] == 1:
                linha += " ■"
            else:
                linha += " ·"
        print(" " + linha)
    print()


# Exemplo 1: conjunto pequeno dos slides (Figura 11)
componente, iteracoes = extrair_componente(CONJUNTO_EXEMPLO, PONTO_INICIAL, ELEMENTO_ESTRUTURANTE)

mostrar_iteracoes(CONJUNTO_EXEMPLO, iteracoes)
print(f"Convergiu após {len(iteracoes) - 1} iterações (X_{len(iteracoes)} = X_{len(iteracoes) - 1})")
print()

# Exemplo 2: rotulação de todos os componentes de uma imagem binária
img_binaria = criar_imagem_exemplo()
rotulos, total = rotular_componentes(img_binaria, ELEMENTO_ESTRUTURANTE)
print(f"Componentes conectados encontrados na imagem de exemplo: {total}")

plt.figure(figsize=(14, 7))

passos = [0, 1, 2, len(iteracoes) - 1]
for posicao, k in enumerate(passos):
    plt.subplot(2, 4, posicao + 1)
    # Cinza claro = A, branco = X_k
    plt.imshow(CONJUNTO_EXEMPLO * 0.4 + iteracoes[k] * 0.6, cmap="gray", vmin=0, vmax=1)
    plt.title(f"X_{k}")
    plt.axis("off")

plt.subplot(2, 4, 5)
plt.imshow(img_binaria, cmap="gray", vmin=0, vmax=1)
plt.title("Imagem binária A")
plt.axis("off")

plt.subplot(2, 4, 6)
plt.imshow(rotulos, cmap="nipy_spectral", vmin=0, vmax=total)
plt.title(f"Componentes rotulados ({total})")
plt.axis("off")

plt.tight_layout()
plt.savefig(
    "extracao-componentes-conectados/output/extracao_componentes.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Resultado salvo em: extracao-componentes-conectados/output/extracao_componentes.png")
