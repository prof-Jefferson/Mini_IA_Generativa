"""Testes pequenos do fluxo educacional.

Execute na raiz do projeto com::

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import contextlib
import io
import random
import tempfile
import unittest
from pathlib import Path

from PIL import Image
import torch

import gerar
from monstros import criar_monstro, gerar_dataset
import treinar
from vae import VAE, carregar_checkpoint, perda_vae, salvar_checkpoint


class TesteDatasetProcedural(unittest.TestCase):
    def test_monstro_tem_formato_esperado(self) -> None:
        imagem = criar_monstro(random.Random(7))
        self.assertEqual(imagem.mode, "RGB")
        self.assertEqual(imagem.size, (16, 16))

    def test_dataset_e_reprodutivel(self) -> None:
        with tempfile.TemporaryDirectory() as temporario:
            raiz = Path(temporario)
            primeira = gerar_dataset(4, raiz / "a", semente=123)
            segunda = gerar_dataset(4, raiz / "b", semente=123)
            self.assertEqual(
                [arquivo.read_bytes() for arquivo in primeira],
                [arquivo.read_bytes() for arquivo in segunda],
            )
            self.assertTrue((raiz / "a" / "manifesto.json").is_file())

    def test_manifesto_ignora_imagens_residuais(self) -> None:
        with tempfile.TemporaryDirectory() as temporario:
            pasta = Path(temporario) / "dados"
            gerar_dataset(5, pasta, semente=1)
            gerar_dataset(2, pasta, semente=2)
            self.assertEqual(len(list(pasta.glob("monstro_*.png"))), 5)
            self.assertEqual(len(treinar.DatasetMonstrinhos(pasta)), 2)


class TesteVAE(unittest.TestCase):
    def test_formas_perda_e_gradientes(self) -> None:
        modelo = VAE(dim_latente=5)
        imagens = torch.rand(3, 3, 16, 16)
        reconstruidas, mu, logvar = modelo(imagens)

        self.assertEqual(tuple(reconstruidas.shape), (3, 768))
        self.assertEqual(tuple(mu.shape), (3, 5))
        self.assertEqual(tuple(logvar.shape), (3, 5))

        total, reconstrucao, kl = perda_vae(
            reconstruidas, imagens, mu, logvar, beta=0.5
        )
        self.assertAlmostEqual(
            total.item(), reconstrucao.item() + 0.5 * kl.item(), places=4
        )
        total.backward()
        self.assertTrue(any(parametro.grad is not None for parametro in modelo.parameters()))

    def test_checkpoint_preserva_configuracao(self) -> None:
        with tempfile.TemporaryDirectory() as temporario:
            caminho = Path(temporario) / "modelo.pt"
            salvar_checkpoint(VAE(dim_latente=2), caminho, epoca=1)
            restaurado, checkpoint = carregar_checkpoint(caminho)
            self.assertEqual(restaurado.dim_latente, 2)
            self.assertEqual(checkpoint["epoca"], 1)


class TesteFluxoCompleto(unittest.TestCase):
    def test_rejeita_parametros_nao_finitos(self) -> None:
        argumentos = treinar.criar_parser().parse_args(["--beta", "nan"])
        with self.assertRaisesRegex(ValueError, "finito"):
            treinar.validar_argumentos(argumentos)
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(gerar.main(["--temperatura", "nan"]), 2)

    def test_dataset_treino_e_geracao(self) -> None:
        with tempfile.TemporaryDirectory() as temporario:
            raiz = Path(temporario)
            dados = raiz / "dados"
            resultados = raiz / "resultados"
            gerar_dataset(20, dados, semente=10)

            codigo_treino = treinar.main(
                [
                    "--dados",
                    str(dados),
                    "--resultados",
                    str(resultados),
                    "--epocas",
                    "1",
                    "--tamanho-lote",
                    "8",
                    "--latente",
                    "3",
                    "--amostras",
                    "4",
                    "--semente",
                    "10",
                    "--dispositivo",
                    "cpu",
                ]
            )
            self.assertEqual(codigo_treino, 0)
            for nome in (
                "epoca_001.png",
                "perda.png",
                "modelo_vae.pt",
                "configuracao.json",
                "historico.json",
                "manifesto_resultados.json",
            ):
                self.assertTrue((resultados / nome).is_file(), nome)

            saida = resultados / "novos.png"
            codigo_geracao = gerar.main(
                [
                    "--modelo",
                    str(resultados / "modelo_vae.pt"),
                    "--saida",
                    str(saida),
                    "--quantidade",
                    "5",
                    "--colunas",
                    "3",
                    "--escala",
                    "2",
                    "--semente",
                    "10",
                    "--dispositivo",
                    "cpu",
                ]
            )
            self.assertEqual(codigo_geracao, 0)
            with Image.open(saida) as imagem:
                self.assertEqual(imagem.mode, "RGB")
                self.assertEqual(imagem.size, (96, 64))


if __name__ == "__main__":
    unittest.main()
