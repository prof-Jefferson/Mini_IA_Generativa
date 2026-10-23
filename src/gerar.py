"""Gera novos monstrinhos a partir de um checkpoint treinado."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import torch

from treinar import escolher_dispositivo, salvar_grade
from vae import carregar_checkpoint


def criar_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sorteia vetores latentes e gera monstrinhos em pixel art."
    )
    parser.add_argument(
        "--modelo",
        type=Path,
        default=Path("resultados/modelo_vae.pt"),
        help="checkpoint treinado (padrao: resultados/modelo_vae.pt)",
    )
    parser.add_argument(
        "--saida",
        type=Path,
        default=Path("resultados/monstrinhos_novos.png"),
        help="PNG de destino (padrao: resultados/monstrinhos_novos.png)",
    )
    parser.add_argument("--quantidade", type=int, default=32, help="numero de imagens (padrao: 32)")
    parser.add_argument(
        "--temperatura", type=float, default=1.0, help="escala da amostragem latente (padrao: 1.0)"
    )
    parser.add_argument("--semente", type=int, default=123, help="semente da amostragem (padrao: 123)")
    parser.add_argument("--escala", type=int, default=8, help="ampliacao dos pixels (padrao: 8)")
    parser.add_argument(
        "--colunas", type=int, default=None, help="colunas da grade (padrao: automatico)"
    )
    parser.add_argument(
        "--dispositivo",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="hardware usado na geracao (padrao: auto)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = criar_parser().parse_args(argv)
    try:
        if args.quantidade <= 0:
            raise ValueError("--quantidade deve ser positiva.")
        if not math.isfinite(args.temperatura) or args.temperatura < 0:
            raise ValueError("--temperatura deve ser finita e nao negativa.")
        if args.escala <= 0:
            raise ValueError("--escala deve ser positiva.")
        if args.colunas is not None and args.colunas <= 0:
            raise ValueError("--colunas deve ser positiva.")

        dispositivo = escolher_dispositivo(args.dispositivo)
        modelo, _checkpoint = carregar_checkpoint(
            args.modelo, dispositivo=dispositivo, modo_avaliacao=True
        )
        dim_latente = modelo.dim_latente

        # Gerar em CPU torna a semente consistente mesmo quando a inferencia usa GPU.
        gerador = torch.Generator(device="cpu").manual_seed(args.semente)
        latentes = torch.randn(
            args.quantidade, dim_latente, generator=gerador
        ).mul_(args.temperatura).to(dispositivo)
        with torch.no_grad():
            imagens = modelo.decode(latentes)
        salvar_grade(
            imagens,
            args.saida,
            escala=args.escala,
            colunas=args.colunas,
        )
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError, KeyError) as erro:
        print(f"Erro: {erro}", file=sys.stderr)
        return 2

    print(
        f"{args.quantidade} monstrinhos gerados com temperatura "
        f"{args.temperatura:g}: {args.saida.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
