"""Treina o VAE de monstrinhos e registra evidencias do aprendizado.

O script evita depender de ``torchvision`` de proposito: converter uma imagem
Pillow em tensor aqui deixa explicita a passagem de pixels 0..255 para 0..1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split

from vae import VAE, perda_vae, salvar_checkpoint as salvar_checkpoint_vae


LARGURA = 16
ALTURA = 16
CANAIS = 3


@dataclass(frozen=True)
class ConfiguracaoTreino:
    """Parametros relevantes para reproduzir um experimento."""

    epocas: int
    tamanho_lote: int
    taxa_aprendizado: float
    beta: float
    dim_latente: int
    proporcao_validacao: float
    semente: int
    dispositivo: str
    quantidade_imagens: int
    caminho_dados: str
    identificador_dataset: str
    semente_dataset: int | None
    quantidade_amostras: int
    intervalo_amostras: int


class DatasetMonstrinhos(Dataset[Tensor]):
    """Carrega os PNGs produzidos por ``monstros.py``."""

    def __init__(self, diretorio: Path) -> None:
        self.diretorio = diretorio
        self.metadados: dict = {}
        caminho_manifesto = diretorio / "manifesto.json"
        if caminho_manifesto.is_file():
            try:
                manifesto = json.loads(caminho_manifesto.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as erro:
                raise ValueError(f"Manifesto de dataset invalido: {erro}") from erro
            if not isinstance(manifesto, dict) or manifesto.get("versao") != 1:
                raise ValueError("Manifesto de dataset ausente ou com versao incompativel.")
            nomes = manifesto.get("arquivos")
            if not isinstance(nomes, list) or not nomes:
                raise ValueError("O manifesto nao contem uma lista de arquivos valida.")
            if any(
                not isinstance(nome, str)
                or Path(nome).name != nome
                or Path(nome).suffix.lower() != ".png"
                for nome in nomes
            ):
                raise ValueError("O manifesto contem um nome de arquivo inseguro ou invalido.")
            if len(nomes) != len(set(nomes)):
                raise ValueError("O manifesto contem nomes de arquivos duplicados.")
            self.arquivos = [diretorio / nome for nome in nomes]
            ausentes = [arquivo.name for arquivo in self.arquivos if not arquivo.is_file()]
            if ausentes:
                exibidos = ", ".join(ausentes[:3])
                raise FileNotFoundError(
                    f"O manifesto referencia PNGs ausentes ({exibidos}). "
                    "Gere o dataset novamente."
                )
            if manifesto.get("quantidade") != len(self.arquivos):
                raise ValueError("A quantidade declarada no manifesto nao confere.")
            self.metadados = manifesto
        else:
            self.arquivos = sorted(diretorio.glob("*.png"))
        if not self.arquivos:
            raise FileNotFoundError(
                f"Nenhum PNG encontrado em '{diretorio}'. "
                "Execute primeiro: python monstros.py --quantidade 1200"
            )

        resumo = hashlib.sha256()
        for arquivo in self.arquivos:
            resumo.update(arquivo.name.encode("utf-8"))
            resumo.update(b"\0")
            resumo.update(arquivo.read_bytes())
        self.identificador = resumo.hexdigest()
        semente = self.metadados.get("semente")
        self.semente = semente if isinstance(semente, int) else None

    def __len__(self) -> int:
        return len(self.arquivos)

    def __getitem__(self, indice: int) -> Tensor:
        caminho = self.arquivos[indice]
        with Image.open(caminho) as imagem:
            imagem_rgb = imagem.convert("RGB")
            if imagem_rgb.size != (LARGURA, ALTURA):
                raise ValueError(
                    f"'{caminho}' tem tamanho {imagem_rgb.size}; "
                    f"esperado: {(LARGURA, ALTURA)}."
                )
            pixels = np.asarray(imagem_rgb, dtype=np.float32).copy() / 255.0
        return torch.from_numpy(pixels).permute(2, 0, 1)


def configurar_semente(semente: int) -> None:
    """Configura as fontes de aleatoriedade usadas neste programa."""

    random.seed(semente)
    np.random.seed(semente)
    torch.manual_seed(semente)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(semente)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def escolher_dispositivo(nome: str) -> torch.device:
    """Resolve ``auto``, ``cpu`` ou ``cuda`` com mensagem de erro clara."""

    if nome == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if nome == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA foi solicitada, mas nao esta disponivel neste PyTorch.")
    return torch.device(nome)


def _tensor_para_imagem(amostra: Tensor) -> Image.Image:
    pixels = (
        amostra.detach()
        .cpu()
        .clamp(0.0, 1.0)
        .permute(1, 2, 0)
        .numpy()
    )
    return Image.fromarray(np.rint(pixels * 255).astype(np.uint8), mode="RGB")


def salvar_grade(
    amostras: Tensor,
    caminho: Path,
    *,
    escala: int = 8,
    colunas: int | None = None,
) -> None:
    """Salva um lote NCHW como grade, ampliado sem borrar os pixels."""

    if amostras.ndim == 2 and amostras.shape[1] == CANAIS * ALTURA * LARGURA:
        amostras = amostras.reshape(-1, CANAIS, ALTURA, LARGURA)
    if amostras.ndim != 4 or amostras.shape[1:] != (CANAIS, ALTURA, LARGURA):
        raise ValueError(
            "As amostras devem ter formato (N, 3, 16, 16); "
            f"recebido: {tuple(amostras.shape)}."
        )
    quantidade = int(amostras.shape[0])
    if quantidade < 1:
        raise ValueError("A grade precisa conter ao menos uma imagem.")
    colunas = colunas or math.ceil(math.sqrt(quantidade))
    colunas = max(1, min(colunas, quantidade))
    linhas = math.ceil(quantidade / colunas)
    grade = Image.new("RGB", (colunas * LARGURA, linhas * ALTURA), (245, 245, 245))
    for indice, amostra in enumerate(amostras):
        grade.paste(
            _tensor_para_imagem(amostra),
            ((indice % colunas) * LARGURA, (indice // colunas) * ALTURA),
        )
    if escala > 1:
        grade = grade.resize(
            (grade.width * escala, grade.height * escala),
            resample=Image.Resampling.NEAREST,
        )
    caminho.parent.mkdir(parents=True, exist_ok=True)
    grade.save(caminho)


def dividir_dataset(
    dataset: DatasetMonstrinhos,
    proporcao_validacao: float,
    semente: int,
) -> tuple[Dataset, Dataset | None]:
    """Cria particoes deterministicas, preservando ao menos um item de treino."""

    if not 0.0 <= proporcao_validacao < 1.0:
        raise ValueError("--validacao deve estar no intervalo [0, 1).")
    if len(dataset) < 2 or proporcao_validacao == 0:
        return dataset, None
    quantidade_validacao = max(1, round(len(dataset) * proporcao_validacao))
    quantidade_validacao = min(quantidade_validacao, len(dataset) - 1)
    quantidade_treino = len(dataset) - quantidade_validacao
    gerador = torch.Generator().manual_seed(semente)
    treino, validacao = random_split(
        dataset, [quantidade_treino, quantidade_validacao], generator=gerador
    )
    return treino, validacao


def executar_epoca(
    modelo: VAE,
    carregador: DataLoader,
    dispositivo: torch.device,
    beta: float,
    otimizador: torch.optim.Optimizer | None,
) -> dict[str, float]:
    """Executa uma epoca de treino ou avaliacao e calcula medias por imagem."""

    em_treino = otimizador is not None
    modelo.train(em_treino)
    totais = {"total": 0.0, "reconstrucao": 0.0, "kl": 0.0}
    quantidade = 0

    contexto = torch.enable_grad() if em_treino else torch.no_grad()
    with contexto:
        for imagens in carregador:
            imagens = imagens.to(dispositivo)
            if em_treino:
                otimizador.zero_grad(set_to_none=True)
            reconstruidas, mu, logvar = modelo(imagens)
            perda, reconstrucao, kl = perda_vae(
                reconstruidas, imagens, mu, logvar, beta=beta
            )
            if em_treino:
                perda.backward()
                otimizador.step()

            tamanho_lote = imagens.shape[0]
            totais["total"] += perda.detach().item() * tamanho_lote
            totais["reconstrucao"] += reconstrucao.detach().item() * tamanho_lote
            totais["kl"] += kl.detach().item() * tamanho_lote
            quantidade += tamanho_lote

    return {nome: valor / quantidade for nome, valor in totais.items()}


def salvar_curvas(historico: list[dict[str, float]], caminho: Path) -> None:
    """Gera um grafico legivel mesmo quando nao ha conjunto de validacao."""

    epocas = [int(item["epoca"]) for item in historico]
    figura, eixo = plt.subplots(figsize=(8, 5))
    eixo.plot(epocas, [item["treino_total"] for item in historico], label="Treino")
    if "validacao_total" in historico[0]:
        eixo.plot(
            epocas,
            [item["validacao_total"] for item in historico],
            label="Validacao",
        )
    eixo.plot(
        epocas,
        [item["treino_reconstrucao"] for item in historico],
        linestyle="--",
        alpha=0.75,
        label="Reconstrucao (treino)",
    )
    eixo.plot(
        epocas,
        [item["treino_kl"] for item in historico],
        linestyle=":",
        alpha=0.85,
        label="KL sem beta (treino)",
    )
    eixo.set(title="Perda por epoca", xlabel="Epoca", ylabel="Media por imagem")
    eixo.grid(alpha=0.25)
    eixo.legend()
    figura.tight_layout()
    figura.savefig(caminho, dpi=140)
    plt.close(figura)


def criar_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Treina um autoencoder variacional com os monstrinhos 16x16."
    )
    parser.add_argument(
        "--dados", type=Path, default=Path("dados"), help="pasta dos PNGs (padrao: dados)"
    )
    parser.add_argument(
        "--resultados",
        type=Path,
        default=Path("resultados"),
        help="pasta dos artefatos (padrao: resultados)",
    )
    parser.add_argument("--epocas", type=int, default=35, help="numero de epocas (padrao: 35)")
    parser.add_argument(
        "--tamanho-lote", type=int, default=64, help="imagens por lote (padrao: 64)"
    )
    parser.add_argument(
        "--taxa-aprendizado", type=float, default=1e-3, help="taxa do Adam (padrao: 0.001)"
    )
    parser.add_argument("--beta", type=float, default=1.0, help="peso do termo KL (padrao: 1.0)")
    parser.add_argument(
        "--latente", type=int, default=12, help="dimensao latente (padrao: 12)"
    )
    parser.add_argument(
        "--validacao", type=float, default=0.1, help="fracao para validacao (padrao: 0.1)"
    )
    parser.add_argument("--semente", type=int, default=42, help="semente do treino (padrao: 42)")
    parser.add_argument(
        "--amostras", type=int, default=16, help="imagens por grade de epoca (padrao: 16)"
    )
    parser.add_argument(
        "--salvar-cada", type=int, default=5, help="intervalo das grades (padrao: 5)"
    )
    parser.add_argument(
        "--dispositivo",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="hardware usado no treino (padrao: auto)",
    )
    return parser


def validar_argumentos(args: argparse.Namespace) -> None:
    positivos = {
        "--epocas": args.epocas,
        "--tamanho-lote": args.tamanho_lote,
        "--taxa-aprendizado": args.taxa_aprendizado,
        "--latente": args.latente,
        "--amostras": args.amostras,
        "--salvar-cada": args.salvar_cada,
    }
    invalidos = [nome for nome, valor in positivos.items() if valor <= 0]
    if invalidos:
        raise ValueError(f"Os parametros devem ser positivos: {', '.join(invalidos)}.")
    if not math.isfinite(args.taxa_aprendizado):
        raise ValueError("--taxa-aprendizado deve ser finita.")
    if not math.isfinite(args.beta) or args.beta < 0:
        raise ValueError("--beta deve ser finito e nao negativo.")
    if not math.isfinite(args.validacao) or not 0 <= args.validacao < 1:
        raise ValueError("--validacao deve estar no intervalo [0, 1).")


def executar_treino(args: argparse.Namespace) -> int:
    """Executa um experimento ja validado pela interface de linha de comando."""

    validar_argumentos(args)
    configurar_semente(args.semente)
    dispositivo = escolher_dispositivo(args.dispositivo)
    dataset = DatasetMonstrinhos(args.dados)
    treino, validacao = dividir_dataset(dataset, args.validacao, args.semente)

    args.resultados.mkdir(parents=True, exist_ok=True)
    resultados_anteriores = list(args.resultados.glob("epoca_*.png"))
    if resultados_anteriores:
        print(
            "Aviso: a pasta de resultados ja contem grades de outra execucao; "
            "consulte manifesto_resultados.json para identificar as grades atuais."
        )
    gerador_lotes = torch.Generator().manual_seed(args.semente)
    carregador_treino = DataLoader(
        treino,
        batch_size=args.tamanho_lote,
        shuffle=True,
        num_workers=0,
        generator=gerador_lotes,
    )
    carregador_validacao = (
        DataLoader(validacao, batch_size=args.tamanho_lote, shuffle=False, num_workers=0)
        if validacao is not None
        else None
    )

    modelo = VAE(dim_latente=args.latente).to(dispositivo)
    otimizador = torch.optim.Adam(modelo.parameters(), lr=args.taxa_aprendizado)
    configuracao = ConfiguracaoTreino(
        epocas=args.epocas,
        tamanho_lote=args.tamanho_lote,
        taxa_aprendizado=args.taxa_aprendizado,
        beta=args.beta,
        dim_latente=args.latente,
        proporcao_validacao=args.validacao,
        semente=args.semente,
        dispositivo=str(dispositivo),
        quantidade_imagens=len(dataset),
        caminho_dados=str(args.dados.resolve()),
        identificador_dataset=dataset.identificador,
        semente_dataset=dataset.semente,
        quantidade_amostras=args.amostras,
        intervalo_amostras=args.salvar_cada,
    )
    (args.resultados / "configuracao.json").write_text(
        json.dumps(asdict(configuracao), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # A mesma amostra latente e reutilizada: assim as imagens entre epocas sao comparaveis.
    gerador_latente = torch.Generator(device="cpu").manual_seed(args.semente + 1)
    latentes_fixos = torch.randn(
        args.amostras, args.latente, generator=gerador_latente
    ).to(dispositivo)
    historico: list[dict[str, float]] = []
    grades_criadas: list[str] = []

    print(
        f"Treinando com {len(treino)} imagens"
        + (f" e validando com {len(validacao)}" if validacao is not None else "")
        + f" em {dispositivo}."
    )
    for epoca in range(1, args.epocas + 1):
        metricas_treino = executar_epoca(
            modelo, carregador_treino, dispositivo, args.beta, otimizador
        )
        registro: dict[str, float] = {
            "epoca": float(epoca),
            **{f"treino_{chave}": valor for chave, valor in metricas_treino.items()},
        }
        if carregador_validacao is not None:
            metricas_validacao = executar_epoca(
                modelo, carregador_validacao, dispositivo, args.beta, None
            )
            registro.update(
                {f"validacao_{chave}": valor for chave, valor in metricas_validacao.items()}
            )
        historico.append(registro)

        resumo = (
            f"Epoca {epoca:03d}/{args.epocas:03d} | "
            f"perda={metricas_treino['total']:.4f} "
            f"recon={metricas_treino['reconstrucao']:.4f} "
            f"kl={metricas_treino['kl']:.4f}"
        )
        if carregador_validacao is not None:
            resumo += f" | val={metricas_validacao['total']:.4f}"
        print(resumo)

        deve_salvar = epoca == 1 or epoca % args.salvar_cada == 0 or epoca == args.epocas
        if deve_salvar:
            modelo.eval()
            with torch.no_grad():
                geradas = modelo.decode(latentes_fixos)
            nome_grade = f"epoca_{epoca:03d}.png"
            salvar_grade(geradas, args.resultados / nome_grade)
            grades_criadas.append(nome_grade)

    salvar_curvas(historico, args.resultados / "perda.png")
    salvar_checkpoint_vae(
        modelo,
        args.resultados / "modelo_vae.pt",
        epoca=args.epocas,
        metadados={
            "configuracao_treino": asdict(configuracao),
            "historico": historico,
        },
    )
    (args.resultados / "historico.json").write_text(
        json.dumps(historico, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    manifesto_resultados = {
        "versao": 1,
        "identificador_dataset": dataset.identificador,
        "epocas": args.epocas,
        "arquivos": [
            *grades_criadas,
            "perda.png",
            "modelo_vae.pt",
            "configuracao.json",
            "historico.json",
            "manifesto_resultados.json",
        ],
    }
    (args.resultados / "manifesto_resultados.json").write_text(
        json.dumps(manifesto_resultados, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Treinamento concluido. Resultados em: {args.resultados.resolve()}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = criar_parser().parse_args(argv)
    try:
        return executar_treino(args)
    except (OSError, RuntimeError, TypeError, ValueError) as erro:
        print(f"Erro: {erro}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
