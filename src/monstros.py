"""Gera um pequeno conjunto de monstrinhos em pixel art.

Este arquivo cria os *dados* usados pelo VAE do projeto.  Não há aprendizado de
máquina nesta etapa: cada imagem é desenhada por regras simples e por escolhas
aleatórias.  Fixar ``semente`` torna essas escolhas reproduzíveis, algo útil ao
comparar experimentos em sala de aula.

Exemplo pela linha de comando::

    python monstros.py --quantidade 1200 --saida dados --semente 42
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

try:
    from PIL import Image, ImageDraw
except ImportError as erro:  # pragma: no cover - depende do ambiente do aluno
    raise ImportError(
        "A biblioteca Pillow é necessária. Instale-a com "
        "'python -m pip install Pillow'."
    ) from erro


TAMANHO = 16
"""Largura e altura, em pixels, de todas as imagens geradas."""

Cor = tuple[int, int, int]
Pixel = tuple[int, int]

COR_CONTORNO: Cor = (35, 30, 46)
COR_OLHO: Cor = (250, 247, 225)
COR_DENTE: Cor = (255, 244, 214)

FUNDOS: tuple[Cor, ...] = (
    (239, 238, 219),
    (220, 235, 239),
    (237, 224, 241),
    (244, 229, 211),
    (222, 236, 218),
    (231, 227, 245),
)

NOME_MANIFESTO = "manifesto.json"
VERSAO_DATASET = 1


@dataclass(frozen=True)
class Paleta:
    """Cores relacionadas que dão volume sem deixar de parecer pixel art."""

    corpo: Cor
    clara: Cor
    sombra: Cor
    detalhe: Cor


PALETAS: tuple[Paleta, ...] = (
    Paleta((76, 184, 91), (126, 222, 119), (42, 130, 72), (245, 203, 74)),
    Paleta((139, 91, 191), (184, 132, 226), (94, 57, 143), (240, 138, 192)),
    Paleta((231, 126, 61), (250, 169, 92), (178, 79, 49), (255, 220, 92)),
    Paleta((62, 153, 211), (102, 197, 235), (38, 99, 166), (128, 232, 190)),
    Paleta((219, 91, 139), (244, 139, 176), (164, 54, 109), (255, 209, 94)),
    Paleta((224, 190, 61), (247, 222, 109), (169, 132, 40), (238, 103, 68)),
    Paleta((50, 181, 166), (94, 220, 197), (28, 126, 124), (245, 145, 75)),
)


def _pixels_do_corpo(
    formato: str, topo: int, altura: int, meia_largura: int
) -> set[Pixel]:
    """Monta uma silhueta simétrica, uma linha horizontal por vez."""

    pixels: set[Pixel] = set()
    ultima_linha = altura - 1

    for deslocamento_y in range(altura):
        if formato == "redondo":
            distancia_da_borda = min(deslocamento_y, ultima_linha - deslocamento_y)
            reducao = 2 if distancia_da_borda == 0 else 1 if distancia_da_borda == 1 else 0
        elif formato == "quadrado":
            reducao = 1 if deslocamento_y in (0, ultima_linha) else 0
        elif formato == "pera":
            if deslocamento_y < 2:
                reducao = 2 - deslocamento_y
            elif deslocamento_y == ultima_linha:
                reducao = 1
            else:
                reducao = 0
        else:  # gota: cabeça estreita e base arredondada
            if deslocamento_y == 0:
                reducao = 2
            elif deslocamento_y == 1:
                reducao = 1
            elif deslocamento_y == ultima_linha:
                reducao = 2
            else:
                reducao = 0

        metade_da_linha = max(2, meia_largura - reducao)
        # O eixo fica entre as colunas 7 e 8; por isso a largura é sempre par.
        esquerda = 8 - metade_da_linha
        direita = 7 + metade_da_linha
        y = topo + deslocamento_y
        pixels.update((x, y) for x in range(esquerda, direita + 1))

    return pixels


def _linha_contornada(
    desenho: ImageDraw.ImageDraw, pontos: Sequence[Pixel], cor: Cor
) -> None:
    """Desenha um apêndice de um pixel cercado pelo contorno escuro."""

    desenho.line(pontos, fill=COR_CONTORNO, width=3)
    desenho.line(pontos, fill=cor, width=1)


def _desenhar_pernas(
    desenho: ImageDraw.ImageDraw, base: int, paleta: Paleta, rng: random.Random
) -> None:
    """Desenha pernas curtas atrás do corpo."""

    # Uma pequena assimetria nos pés aumenta a diversidade do conjunto.
    deslocamento = rng.choice((-1, 0, 0, 1))
    for x, direcao in ((5, -1), (10, 1)):
        desenho.rectangle((x - 1, base - 1, x + 1, 14), fill=COR_CONTORNO)
        desenho.line((x, base - 1, x, 13), fill=paleta.sombra, width=1)
        ponta = max(1, min(14, x + direcao + deslocamento))
        desenho.line((x, 13, ponta, 13), fill=paleta.corpo, width=1)


def _desenhar_bracos(
    desenho: ImageDraw.ImageDraw,
    esquerda: int,
    direita: int,
    topo: int,
    paleta: Paleta,
    rng: random.Random,
) -> None:
    """Sorteia braços para baixo, para os lados ou comemorando."""

    pose = rng.choice(("baixo", "baixo", "lado", "alto"))
    y_raiz = topo + 4
    if pose == "alto":
        esquerdo = ((esquerda + 1, y_raiz), (max(1, esquerda - 1), topo + 1))
        direito = ((direita - 1, y_raiz), (min(14, direita + 1), topo + 1))
    elif pose == "lado":
        esquerdo = ((esquerda + 1, y_raiz), (max(1, esquerda - 2), y_raiz))
        direito = ((direita - 1, y_raiz), (min(14, direita + 2), y_raiz))
    else:
        esquerdo = ((esquerda + 1, y_raiz), (max(1, esquerda - 1), y_raiz + 3))
        direito = ((direita - 1, y_raiz), (min(14, direita + 1), y_raiz + 3))

    _linha_contornada(desenho, esquerdo, paleta.corpo)
    _linha_contornada(desenho, direito, paleta.corpo)


def _desenhar_cauda(
    desenho: ImageDraw.ImageDraw,
    esquerda: int,
    direita: int,
    topo: int,
    paleta: Paleta,
    rng: random.Random,
) -> None:
    """Às vezes adiciona uma cauda lateral com uma ponta colorida."""

    if rng.random() >= 0.38:
        return

    lado = rng.choice((-1, 1))
    x_raiz = esquerda if lado < 0 else direita
    x_meio = max(1, min(14, x_raiz + 2 * lado))
    x_ponta = max(1, min(14, x_raiz + 4 * lado))
    pontos = ((x_raiz, topo + 5), (x_meio, topo + 4), (x_ponta, topo + 2))
    _linha_contornada(desenho, pontos, paleta.sombra)
    desenho.point((x_ponta, topo + 1), fill=paleta.detalhe)


def _desenhar_enfeite_da_cabeca(
    desenho: ImageDraw.ImageDraw,
    esquerda: int,
    direita: int,
    topo: int,
    paleta: Paleta,
    rng: random.Random,
) -> None:
    """Adiciona chifres, orelhas, antenas ou um pequeno topete."""

    enfeite = rng.choice(("chifres", "chifres", "orelhas", "antenas", "tufo", "nenhum"))

    if enfeite == "chifres":
        for centro in (esquerda + 2, direita - 2):
            desenho.polygon(
                ((centro - 2, topo), (centro, max(0, topo - 4)), (centro + 2, topo)),
                fill=COR_CONTORNO,
            )
            desenho.line(
                (centro, max(1, topo - 3), centro, topo - 1), fill=COR_DENTE, width=1
            )
    elif enfeite == "orelhas":
        desenho.polygon(
            ((esquerda + 1, topo + 1), (max(0, esquerda - 2), topo - 1), (esquerda, topo + 3)),
            fill=COR_CONTORNO,
        )
        desenho.polygon(
            ((direita - 1, topo + 1), (min(15, direita + 2), topo - 1), (direita, topo + 3)),
            fill=COR_CONTORNO,
        )
        desenho.point((max(0, esquerda - 1), topo), fill=paleta.clara)
        desenho.point((min(15, direita + 1), topo), fill=paleta.clara)
    elif enfeite == "antenas":
        _linha_contornada(
            desenho, ((esquerda + 3, topo + 1), (esquerda + 2, max(1, topo - 2))), paleta.corpo
        )
        _linha_contornada(
            desenho, ((direita - 3, topo + 1), (direita - 2, max(1, topo - 2))), paleta.corpo
        )
        desenho.point((esquerda + 2, max(0, topo - 3)), fill=paleta.detalhe)
        desenho.point((direita - 2, max(0, topo - 3)), fill=paleta.detalhe)
    elif enfeite == "tufo":
        desenho.polygon(
            ((6, topo + 1), (6, topo - 2), (8, topo), (9, topo - 3), (10, topo + 1)),
            fill=COR_CONTORNO,
        )
        desenho.line((7, topo - 1, 8, topo), fill=paleta.clara, width=1)


def _pintar_corpo(
    imagem: Image.Image, pixels: set[Pixel], paleta: Paleta, rng: random.Random
) -> None:
    """Pinta preenchimento, contorno e uma textura sorteada."""

    acesso = imagem.load()
    interiores: list[Pixel] = []

    for x, y in pixels:
        vizinhos = ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1))
        if any(vizinho not in pixels for vizinho in vizinhos):
            acesso[x, y] = COR_CONTORNO
        else:
            acesso[x, y] = paleta.corpo
            interiores.append((x, y))

    # Luz à esquerda e sombra à direita ajudam a leitura da silhueta minúscula.
    linhas = sorted({y for _, y in interiores})
    for y in linhas[::2]:
        xs = sorted(x for x, py in interiores if py == y)
        if len(xs) >= 4:
            acesso[xs[0], y] = paleta.clara
            acesso[xs[-1], y] = paleta.sombra

    textura = rng.choice(("liso", "liso", "pintas", "barriga", "faixa"))
    if textura == "pintas" and interiores:
        quantidade = min(rng.randint(3, 6), len(interiores))
        for x, y in rng.sample(interiores, quantidade):
            acesso[x, y] = rng.choice((paleta.clara, paleta.sombra, paleta.detalhe))
    elif textura == "barriga":
        centro_y = max(y for _, y in pixels) - 2
        for x, y in interiores:
            if 6 <= x <= 9 and centro_y - 1 <= y <= centro_y + 1:
                acesso[x, y] = paleta.clara
    elif textura == "faixa":
        faixa_y = min(y for _, y in pixels) + 4
        for x, y in interiores:
            if y == faixa_y:
                acesso[x, y] = paleta.sombra


def _desenhar_rosto(
    desenho: ImageDraw.ImageDraw, topo: int, base: int, paleta: Paleta, rng: random.Random
) -> None:
    """Desenha de um a três olhos e uma boca legível em poucos pixels."""

    y_olho = topo + 2
    direcao_do_olhar = rng.choice((-1, 0, 0, 0, 1))
    quantidade_de_olhos = rng.choice((1, 2, 2, 2, 2, 3))

    if quantidade_de_olhos == 1:
        desenho.rectangle((5, y_olho - 1, 10, y_olho + 2), fill=COR_CONTORNO)
        desenho.rectangle((6, y_olho, 9, y_olho + 1), fill=COR_OLHO)
        desenho.point((max(6, min(9, 8 + direcao_do_olhar)), y_olho + 1), fill=COR_CONTORNO)
    elif quantidade_de_olhos == 2:
        for inicio_x in (5, 9):
            desenho.rectangle((inicio_x, y_olho, inicio_x + 1, y_olho + 1), fill=COR_OLHO)
            pupila_x = inicio_x if direcao_do_olhar < 0 else inicio_x + 1
            desenho.point((pupila_x, y_olho + 1), fill=COR_CONTORNO)
    else:
        # Em 16×16, olhos extras funcionam melhor como pequenos pontos luminosos.
        for x in (5, 7, 10):
            desenho.line((x, y_olho, x, y_olho + 1), fill=COR_OLHO, width=1)
            desenho.point((x, y_olho + 1), fill=COR_CONTORNO)

    y_boca = min(topo + 6, base - 2)
    boca = rng.choice(("sorriso", "sorriso", "reta", "aberta", "surpresa", "presas"))

    if boca == "sorriso":
        desenho.point(((6, y_boca), (7, y_boca + 1), (8, y_boca + 1), (9, y_boca)), fill=COR_CONTORNO)
    elif boca == "reta":
        desenho.line((6, y_boca, 9, y_boca), fill=COR_CONTORNO, width=1)
    elif boca == "aberta":
        desenho.rectangle((6, y_boca, 9, y_boca + 1), fill=COR_CONTORNO)
        desenho.point((7, y_boca + 1), fill=paleta.detalhe)
        desenho.point((8, y_boca + 1), fill=paleta.detalhe)
    elif boca == "surpresa":
        desenho.rectangle((7, y_boca, 8, y_boca + 1), fill=COR_CONTORNO)
    else:
        desenho.line((6, y_boca, 9, y_boca), fill=COR_CONTORNO, width=1)
        desenho.point((6, y_boca + 1), fill=COR_DENTE)
        desenho.point((9, y_boca + 1), fill=COR_DENTE)


def criar_monstro(rng: random.Random | None = None) -> Image.Image:
    """Cria e devolve um monstrinho RGB de 16×16 pixels.

    Args:
        rng: Gerador aleatório a usar. Passe ``random.Random(semente)`` para
            reproduzir exatamente uma sequência de imagens. Se omitido, um
            gerador novo e não determinístico é criado.
    """

    rng = rng if rng is not None else random.Random()
    fundo = rng.choice(FUNDOS)
    paleta = rng.choice(PALETAS)
    imagem = Image.new("RGB", (TAMANHO, TAMANHO), fundo)
    desenho = ImageDraw.Draw(imagem)

    altura = rng.choice((8, 8, 9))
    topo = 4 if altura == 9 else rng.choice((4, 5))
    base = topo + altura - 1
    meia_largura = rng.choice((4, 5, 5))
    formato = rng.choice(("redondo", "redondo", "quadrado", "pera", "gota"))
    pixels = _pixels_do_corpo(formato, topo, altura, meia_largura)
    esquerda = min(x for x, _ in pixels)
    direita = max(x for x, _ in pixels)

    # Apêndices vêm primeiro, para suas raízes ficarem escondidas pelo corpo.
    _desenhar_pernas(desenho, base, paleta, rng)
    _desenhar_cauda(desenho, esquerda, direita, topo, paleta, rng)
    _desenhar_bracos(desenho, esquerda, direita, topo, paleta, rng)
    _desenhar_enfeite_da_cabeca(desenho, esquerda, direita, topo, paleta, rng)
    _pintar_corpo(imagem, pixels, paleta, rng)
    _desenhar_rosto(desenho, topo, base, paleta, rng)

    return imagem


def gerar_dataset(
    quantidade: int = 1200,
    saida: str | Path = "dados",
    semente: int = 42,
) -> list[Path]:
    """Gera ``quantidade`` imagens e retorna os caminhos criados.

    Arquivos de mesmo nome são sobrescritos. Outros arquivos que já estejam na
    pasta não são apagados, para evitar perda acidental de material do aluno.

    Raises:
        TypeError: se quantidade ou semente não forem números inteiros.
        ValueError: se quantidade não for positiva ou a semente for negativa.
        NotADirectoryError: se ``saida`` apontar para um arquivo existente.
    """

    if isinstance(quantidade, bool) or not isinstance(quantidade, int):
        raise TypeError("quantidade deve ser um número inteiro")
    if quantidade <= 0:
        raise ValueError("quantidade deve ser maior que zero")
    if isinstance(semente, bool) or not isinstance(semente, int):
        raise TypeError("semente deve ser um número inteiro")
    if semente < 0:
        raise ValueError("semente deve ser maior ou igual a zero")

    pasta = Path(saida)
    if pasta.exists() and not pasta.is_dir():
        raise NotADirectoryError(f"o caminho de saída existe e não é uma pasta: {pasta}")
    pasta.mkdir(parents=True, exist_ok=True)

    rng = random.Random(semente)
    caminhos: list[Path] = []
    for indice in range(quantidade):
        caminho = pasta / f"monstro_{indice:04d}.png"
        criar_monstro(rng).save(caminho, format="PNG")
        caminhos.append(caminho)

    # O manifesto delimita esta execucao sem precisar apagar imagens antigas.
    # Isso evita que um experimento menor reutilize PNGs residuais por engano.
    manifesto = {
        "versao": VERSAO_DATASET,
        "gerador": "procedural-monstrinhos-v1",
        "quantidade": quantidade,
        "semente": semente,
        "largura": TAMANHO,
        "altura": TAMANHO,
        "modo": "RGB",
        "arquivos": [caminho.name for caminho in caminhos],
    }
    destino_manifesto = pasta / NOME_MANIFESTO
    temporario = pasta / f".{NOME_MANIFESTO}.tmp"
    try:
        temporario.write_text(
            json.dumps(manifesto, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        temporario.replace(destino_manifesto)
    finally:
        if temporario.exists():
            temporario.unlink()

    return caminhos


def _inteiro_positivo(texto: str) -> int:
    """Conversor do argparse com uma mensagem de erro amigável."""

    try:
        valor = int(texto)
    except ValueError as erro:
        raise argparse.ArgumentTypeError("use um número inteiro") from erro
    if valor <= 0:
        raise argparse.ArgumentTypeError("deve ser maior que zero")
    return valor


def _inteiro_nao_negativo(texto: str) -> int:
    """Conversor do argparse para sementes inteiras não negativas."""

    try:
        valor = int(texto)
    except ValueError as erro:
        raise argparse.ArgumentTypeError("use um número inteiro") from erro
    if valor < 0:
        raise argparse.ArgumentTypeError("deve ser maior ou igual a zero")
    return valor


def criar_parser() -> argparse.ArgumentParser:
    """Cria o parser separadamente para facilitar testes e reutilização."""

    parser = argparse.ArgumentParser(
        description="Gera imagens RGB 16×16 de monstrinhos para treinar o VAE."
    )
    parser.add_argument(
        "--quantidade",
        type=_inteiro_positivo,
        default=1200,
        help="número de imagens a gerar (padrão: 1200)",
    )
    parser.add_argument(
        "--saida",
        type=Path,
        default=Path("dados"),
        help="pasta que receberá os PNGs (padrão: dados)",
    )
    parser.add_argument(
        "--semente",
        type=_inteiro_nao_negativo,
        default=42,
        help="semente para reprodução do dataset (padrão: 42)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Executa a interface de linha de comando."""

    parser = criar_parser()
    argumentos = parser.parse_args(argv)
    arquivos_anteriores = (
        set(argumentos.saida.glob("monstro_*.png"))
        if argumentos.saida.is_dir()
        else set()
    )
    try:
        caminhos = gerar_dataset(
            quantidade=argumentos.quantidade,
            saida=argumentos.saida,
            semente=argumentos.semente,
        )
    except (OSError, TypeError, ValueError) as erro:
        parser.error(str(erro))

    residuais = sorted(arquivos_anteriores.difference(caminhos))
    if residuais:
        print(
            f"Aviso: {len(residuais)} PNG(s) antigo(s) permanecem na pasta, "
            f"mas nao fazem parte de {NOME_MANIFESTO} e serao ignorados pelo treino."
        )

    print(
        f"Geradas {len(caminhos)} imagens RGB {TAMANHO}×{TAMANHO} "
        f"em '{argumentos.saida}' (semente {argumentos.semente})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
