import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def rotular_componentes(imagem):
    imagem = np.array(imagem, dtype=np.uint8)

    rotulos = np.zeros_like(imagem, dtype=int)
    equivalencias = {}

    proximo_label = 1

    for i in range(imagem.shape[0]):
        for j in range(imagem.shape[1]):

            p = imagem[i, j]

            #Se p = 0 então verifica o próximo pixel;
            if p == 0:
                continue

            # Vizinho da esquerda
            if j > 0:
                r = imagem[i, j - 1]
                label_r = rotulos[i, j - 1]
            else:
                r = 0
                label_r = 0

            # Vizinho acima
            if i > 0:
                t = imagem[i - 1, j]
                label_t = rotulos[i - 1, j]
            else:
                t = 0
                label_t = 0

            # Se (r = 0 e t = 0) então rotula p com novo rótulo;
            if r == 0 and t == 0:

                rotulos[i, j] = proximo_label
                equivalencias[proximo_label] = proximo_label

                proximo_label += 1

            # Se ( r = 1 e t = 0) ou (r = 0 e t = 1) rotula p com o rótulo de r ou de t;
            elif r == 1 and t == 0:
                rotulos[i, j] = label_r

            elif r == 0 and t == 1:
                rotulos[i, j] = label_t

            else:
                #Se (r = 1 e t = 1) e possuem o mesmo rótulo então rotula p com este rótulo;
                if label_r == label_t:
                    rotulos[i, j] = label_r

                # Se (r = 1 e t = 1) e possuem rótulos diferentes então rotula p com um dos rótulos e indica equivalência de rótulos;
                else:
                    menor = min(label_r, label_t)
                    maior = max(label_r, label_t)
                    rotulos[i, j] = menor

                    equivalencias[maior] = menor

    # Resolve as equivalências
    def encontrar(label):
        while equivalencias.get(label, label) != label:
            label = equivalencias[label]

        return label

    for i in range(rotulos.shape[0]):
        for j in range(rotulos.shape[1]):

            if rotulos[i, j] != 0:
                rotulos[i, j] = encontrar(rotulos[i, j])
                
    return rotulos

imagem = Image.open("Rotulação/img/casa.jfif")
imagem = imagem.convert("L")
imagem = np.array(imagem)

imagem_binaria = (imagem > 127).astype(np.uint8)

rotulos = rotular_componentes(imagem_binaria)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(imagem_binaria, cmap="gray")
plt.title("Imagem binária")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(rotulos)
plt.title("Componentes rotuladas")
plt.axis("off")

plt.show()