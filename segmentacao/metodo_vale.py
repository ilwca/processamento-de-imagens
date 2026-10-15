import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

if (os.environ.get("DISPLAY", "") == "" and os.environ.get("WAYLAND_DISPLAY", "") == ""):
    import matplotlib
    matplotlib.use("Agg")

NIVEIS = 256

def montar_histograma(imagem):
    histograma = np.zeros(NIVEIS, dtype=int)
    for linha in imagem:
        for pixel in linha:
            histograma[pixel] += 1
    return histograma

def suavizar_histograma(histograma, sigma=5):
    raio = int(3 * sigma)
    x = np.arange(-raio, raio + 1)
    kernel = np.exp(-(x**2) / (2 * sigma**2))
    kernel = kernel / np.sum(kernel)
    return np.convolve(histograma, kernel, mode="same")

def encontrar_limiar_vale(histograma, sigma=10):
    h_suave = suavizar_histograma(histograma, sigma=sigma)

    pico_escuro = int(np.argmax(h_suave[:100]))
    pico_claro = int(100 + np.argmax(h_suave[100:]))

    intervalo_vale = h_suave[pico_escuro : pico_claro + 1]
    limiar = pico_escuro + int(np.argmin(intervalo_vale))

    return limiar, pico_escuro, pico_claro, h_suave

def aplicar_limiarizacao(imagem, limiar):
    img_array = np.array(imagem)
    imagem_binarizada = np.where(img_array >= limiar, 255, 0).astype(np.uint8)
    return Image.fromarray(imagem_binarizada)


caminho_entrada = "processamento-de-imagens/segmentacao/img/imagem4.jpg"
pasta_saida = "processamento-de-imagens/segmentacao/output"
os.makedirs(pasta_saida, exist_ok=True)

imagem = Image.open(caminho_entrada).convert("L")
img_array = np.array(imagem)

hist_original = montar_histograma(img_array)
limiar, p_esq, p_dir, hist_suave = encontrar_limiar_vale(hist_original, sigma=10)

imagem_segmentada = aplicar_limiarizacao(imagem, limiar)

print(f"Pico Escuro: {p_esq} | Pico Claro: {p_dir}")
print(f"Limiar de corte calculado (Vale): {limiar}")

plt.figure(figsize=(14, 8))

plt.subplot(2, 2, 1)
plt.imshow(imagem, cmap="gray", vmin=0, vmax=255)
plt.title("Imagem Original")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.bar(
    range(NIVEIS),
    hist_original,
    color="lightgray",
    width=1,
    label="Histograma bruto",
)
plt.plot(
    range(NIVEIS), hist_suave, color="blue", linewidth=1.5, label="Suavizado"
)
plt.scatter(
    [p_esq, p_dir],
    [hist_suave[p_esq], hist_suave[p_dir]],
    color="green",
    s=45,
    zorder=5,
    label="Picos detectados",
)
plt.axvline(
    limiar,
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Limiar Vale (T={limiar})",
)
plt.title("Detecção do Vale no Histograma")
plt.xlabel("Nível de cinza")
plt.ylabel("Frequência")
plt.legend()

plt.subplot(2, 2, 3)
plt.imshow(imagem_segmentada, cmap="gray", vmin=0, vmax=255)
plt.title(f"Imagem Segmentada (T = {limiar})")
plt.axis("off")


plt.tight_layout()
caminho_saida = os.path.join(pasta_saida, "segmentacao_vale.png")
plt.savefig(caminho_saida, dpi=300, bbox_inches="tight")
plt.close()

print(f"Resultado salvo em: {caminho_saida}")