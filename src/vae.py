"""Autoencoder variacional usado no projeto dos monstrinhos.

O modelo recebe imagens RGB de 16 x 16 pixels (ou os mesmos pixels ja
achatados) e aprende uma representacao latente configuravel.  Este modulo
concentra tambem a funcao de perda e pequenos auxiliares de checkpoint para
que os scripts de treino e geracao compartilhem o mesmo formato de arquivo.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F


ALTURA_IMAGEM = 16
LARGURA_IMAGEM = 16
CANAIS_IMAGEM = 3
DIMENSAO_ENTRADA = CANAIS_IMAGEM * ALTURA_IMAGEM * LARGURA_IMAGEM
VERSAO_CHECKPOINT = 1


class VAE(nn.Module):
    """VAE denso para imagens RGB de 16 x 16 pixels.

    Args:
        dim_latente: Numero de variaveis da representacao latente. O valor
            padrao (12) segue o experimento proposto no guia do projeto.

    O metodo :meth:`forward` aceita lotes com formato ``[N, 3, 16, 16]`` ou
    ``[N, 768]`` e devolve ``(reconstruida, mu, logvar)``. A reconstrucao e
    sempre achatada, com formato ``[N, 768]``.
    """

    def __init__(self, dim_latente: int = 12) -> None:
        super().__init__()
        if isinstance(dim_latente, bool) or not isinstance(dim_latente, int):
            raise TypeError("dim_latente deve ser um numero inteiro.")
        if dim_latente <= 0:
            raise ValueError("dim_latente deve ser maior que zero.")

        self.dim_latente = dim_latente

        self.encoder = nn.Sequential(
            nn.Linear(DIMENSAO_ENTRADA, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(128, dim_latente)
        self.fc_logvar = nn.Linear(128, dim_latente)

        self.decoder = nn.Sequential(
            nn.Linear(dim_latente, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, DIMENSAO_ENTRADA),
            nn.Sigmoid(),
        )

    @staticmethod
    def _achatar_entrada(x: Tensor) -> Tensor:
        """Valida e achata um lote de imagens, preservando o batch."""

        if not isinstance(x, Tensor):
            raise TypeError("A entrada do VAE deve ser um torch.Tensor.")

        if x.ndim == 4:
            formato_imagem = (CANAIS_IMAGEM, ALTURA_IMAGEM, LARGURA_IMAGEM)
            if tuple(x.shape[1:]) != formato_imagem:
                raise ValueError(
                    "Esperava imagens no formato [N, 3, 16, 16], "
                    f"mas recebi {list(x.shape)}."
                )
            return x.flatten(start_dim=1)

        if x.ndim == 2 and x.shape[1] == DIMENSAO_ENTRADA:
            return x

        raise ValueError(
            "A entrada deve ter formato [N, 3, 16, 16] ou [N, 768]; "
            f"recebi {list(x.shape)}."
        )

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Codifica imagens nos parametros ``mu`` e ``logvar`` da posterior."""

        h = self.encoder(self._achatar_entrada(x))
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparametrizar(mu: Tensor, logvar: Tensor) -> Tensor:
        """Amostra ``z`` com o truque de reparametrizacao.

        A forma ``z = mu + epsilon * sigma`` permite que gradientes atravessem
        a operacao estocastica durante o treinamento.
        """

        if not isinstance(mu, Tensor) or not isinstance(logvar, Tensor):
            raise TypeError("mu e logvar devem ser tensores.")
        if mu.shape != logvar.shape:
            raise ValueError(
                "mu e logvar devem ter o mesmo formato; "
                f"recebi {list(mu.shape)} e {list(logvar.shape)}."
            )
        desvio_padrao = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(desvio_padrao)
        return mu + epsilon * desvio_padrao

    @staticmethod
    def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
        """Alias em ingles para interoperar com exemplos usuais de VAE."""

        return VAE.reparametrizar(mu, logvar)

    def decode(self, z: Tensor) -> Tensor:
        """Decodifica vetores latentes em pixels no intervalo ``[0, 1]``."""

        if not isinstance(z, Tensor):
            raise TypeError("z deve ser um torch.Tensor.")
        if z.ndim == 1:
            z = z.unsqueeze(0)
        if z.ndim != 2 or z.shape[1] != self.dim_latente:
            raise ValueError(
                f"z deve ter formato [N, {self.dim_latente}]; "
                f"recebi {list(z.shape)}."
            )
        return self.decoder(z)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Reconstrói um lote e retorna ``(reconstruida, mu, logvar)``."""

        mu, logvar = self.encode(x)
        z = self.reparametrizar(mu, logvar)
        return self.decode(z), mu, logvar

    def obter_configuracao(self) -> dict[str, int]:
        """Retorna a configuracao minima necessaria para recriar o modelo."""

        return {
            "dim_latente": self.dim_latente,
            "canais": CANAIS_IMAGEM,
            "altura": ALTURA_IMAGEM,
            "largura": LARGURA_IMAGEM,
        }


def perda_vae(
    reconstruida: Tensor,
    original: Tensor,
    mu: Tensor,
    logvar: Tensor,
    beta: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Calcula a perda do VAE e separa seus componentes didaticos.

    O erro quadratico e somado nos 768 pixels de cada imagem e, em seguida,
    calculado como media do lote. O KL tambem e somado nas dimensoes latentes
    por imagem e calculado como media do lote. Assim, a escala nao muda apenas
    porque o tamanho do batch mudou.

    Args:
        reconstruida: Pixels produzidos pelo decoder, normalmente ``[N, 768]``.
        original: Imagens alvo em ``[N, 3, 16, 16]`` ou ``[N, 768]``.
        mu: Medias da distribuicao latente, com formato ``[N, L]``.
        logvar: Log-variancias latentes, com o mesmo formato de ``mu``.
        beta: Peso nao negativo aplicado ao termo de divergencia KL.

    Returns:
        Tupla ``(total, reconstrucao, kl)`` de tensores escalares, em que
        ``total = reconstrucao + beta * kl``.
    """

    if not isinstance(reconstruida, Tensor):
        raise TypeError("reconstruida deve ser um torch.Tensor.")
    if not isinstance(original, Tensor):
        raise TypeError("original deve ser um torch.Tensor.")
    if not isinstance(mu, Tensor) or not isinstance(logvar, Tensor):
        raise TypeError("mu e logvar devem ser tensores.")
    if isinstance(beta, bool) or not isinstance(beta, (int, float)):
        raise TypeError("beta deve ser um numero real.")
    if not math.isfinite(float(beta)) or beta < 0:
        raise ValueError("beta deve ser finito e maior ou igual a zero.")
    if reconstruida.ndim < 2:
        raise ValueError("reconstruida deve incluir as dimensoes batch e pixels.")
    if original.ndim < 2:
        raise ValueError("original deve incluir as dimensoes batch e pixels.")
    if reconstruida.shape[0] == 0:
        raise ValueError("Nao e possivel calcular a perda de um batch vazio.")
    if reconstruida.shape[0] != original.shape[0]:
        raise ValueError("reconstruida e original devem ter o mesmo batch.")
    if mu.shape != logvar.shape or mu.ndim != 2:
        raise ValueError("mu e logvar devem ter o mesmo formato bidimensional.")
    if mu.shape[0] != reconstruida.shape[0]:
        raise ValueError("mu, logvar e imagens devem ter o mesmo batch.")

    reconstruida_flat = reconstruida.flatten(start_dim=1)
    original_flat = original.flatten(start_dim=1)
    if reconstruida_flat.shape != original_flat.shape:
        raise ValueError(
            "reconstruida e original devem conter a mesma quantidade de pixels "
            f"por imagem; recebi {list(reconstruida_flat.shape)} e "
            f"{list(original_flat.shape)}."
        )

    # Soma os pixels de cada imagem e tira a media entre imagens do lote.
    reconstrucao = F.mse_loss(
        reconstruida_flat, original_flat, reduction="none"
    ).sum(dim=1).mean()
    kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)).mean()
    total = reconstrucao + float(beta) * kl
    return total, reconstrucao, kl


def criar_checkpoint(
    modelo: VAE,
    *,
    otimizador: torch.optim.Optimizer | None = None,
    epoca: int | None = None,
    metadados: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Monta um checkpoint com pesos e configuracao do VAE.

    Guardar ``dim_latente`` junto aos pesos evita recriar acidentalmente um
    modelo incompatível durante a geracao.
    """

    if not isinstance(modelo, VAE):
        raise TypeError("modelo deve ser uma instancia de VAE.")
    if epoca is not None and (isinstance(epoca, bool) or not isinstance(epoca, int)):
        raise TypeError("epoca deve ser um inteiro ou None.")
    if epoca is not None and epoca < 0:
        raise ValueError("epoca nao pode ser negativa.")

    checkpoint: dict[str, Any] = {
        "versao": VERSAO_CHECKPOINT,
        "configuracao_modelo": modelo.obter_configuracao(),
        "estado_modelo": modelo.state_dict(),
    }
    if otimizador is not None:
        checkpoint["estado_otimizador"] = otimizador.state_dict()
    if epoca is not None:
        checkpoint["epoca"] = epoca
    if metadados is not None:
        checkpoint["metadados"] = dict(metadados)
    return checkpoint


def salvar_checkpoint(
    modelo: VAE,
    caminho: str | os.PathLike[str],
    *,
    otimizador: torch.optim.Optimizer | None = None,
    epoca: int | None = None,
    metadados: Mapping[str, Any] | None = None,
) -> None:
    """Salva atomicamente um checkpoint produzido por :func:`criar_checkpoint`."""

    destino = Path(caminho)
    destino.parent.mkdir(parents=True, exist_ok=True)
    temporario = destino.with_name(destino.name + ".tmp")
    checkpoint = criar_checkpoint(
        modelo,
        otimizador=otimizador,
        epoca=epoca,
        metadados=metadados,
    )
    try:
        torch.save(checkpoint, temporario)
        temporario.replace(destino)
    finally:
        if temporario.exists():
            temporario.unlink()


def carregar_checkpoint(
    caminho: str | os.PathLike[str],
    *,
    dispositivo: str | torch.device = "cpu",
    modo_avaliacao: bool = True,
) -> tuple[VAE, dict[str, Any]]:
    """Carrega com seguranca um VAE e devolve o checkpoint completo.

    ``weights_only=True`` restringe os tipos desserializados pelo PyTorch. A
    configuracao gravada e validada antes de os pesos serem aplicados.
    """

    origem = Path(caminho)
    if not origem.is_file():
        raise FileNotFoundError(f"Checkpoint nao encontrado: {origem}")

    try:
        checkpoint = torch.load(
            origem, map_location=dispositivo, weights_only=True
        )
    except TypeError:
        # Compatibilidade com versoes antigas do PyTorch, anteriores ao
        # parametro weights_only. Use apenas checkpoints criados localmente.
        checkpoint = torch.load(origem, map_location=dispositivo)

    if not isinstance(checkpoint, dict):
        raise ValueError("O arquivo nao contem um checkpoint valido.")
    if checkpoint.get("versao") != VERSAO_CHECKPOINT:
        raise ValueError(
            "Versao de checkpoint ausente ou incompativel: "
            f"{checkpoint.get('versao')!r}."
        )

    configuracao = checkpoint.get("configuracao_modelo")
    estado_modelo = checkpoint.get("estado_modelo")
    if not isinstance(configuracao, dict) or not isinstance(estado_modelo, dict):
        raise ValueError("Checkpoint sem configuracao ou pesos do modelo.")

    formato_esperado = {
        "canais": CANAIS_IMAGEM,
        "altura": ALTURA_IMAGEM,
        "largura": LARGURA_IMAGEM,
    }
    for nome, esperado in formato_esperado.items():
        if configuracao.get(nome) != esperado:
            raise ValueError(
                f"Configuracao incompativel para {nome}: "
                f"esperava {esperado}, recebi {configuracao.get(nome)!r}."
            )

    dim_latente = configuracao.get("dim_latente")
    modelo = VAE(dim_latente=dim_latente)
    modelo.load_state_dict(estado_modelo, strict=True)
    modelo.to(dispositivo)
    if modo_avaliacao:
        modelo.eval()
    return modelo, checkpoint


__all__ = [
    "ALTURA_IMAGEM",
    "CANAIS_IMAGEM",
    "DIMENSAO_ENTRADA",
    "LARGURA_IMAGEM",
    "VAE",
    "carregar_checkpoint",
    "criar_checkpoint",
    "perda_vae",
    "salvar_checkpoint",
]
