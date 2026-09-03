import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


if os.environ.get("DISPLAY", "") == "" and os.environ.get("WAYLAND_DISPLAY", "") == "":
    import matplotlib
    matplotlib.use("Agg")


def transformacao_potencia(imagem, gamma, c=1):
    img = np.array(imagem)

    r = img / 255.0

    s = c * (r ** gamma)

    s = np.clip(s * 255, 0, 255)

    return Image.fromarray(s.astype(np.uint8))


imagem = Image.open('Funcao_Intensidade/img/cidade.jpg').convert("L")

gamma = 1.5

imagem_saida = transformacao_potencia(imagem, gamma)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(imagem, cmap="gray")
plt.title("Imagem original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(imagem_saida, cmap="gray")
plt.title(f"Transformação de potência - γ = {gamma}")
plt.axis("off")

plt.savefig(
    "Funcao_Intensidade/output/comparacao.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Imagem transformada: Funcao_Intensidade/output/trans_pot.png")