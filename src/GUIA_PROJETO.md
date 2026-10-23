# Mini IA generativa de monstrinhos — laboratório local em Python

**Disciplina:** Inteligência Artificial · **Curso:** Sistemas de Informação  
**Objetivo:** construir e compreender uma rede neural que aprende características de pequenos monstros em pixel art e gera novas imagens imperfeitas.

## 1. O que a IA faz?

Trabalharemos com imagens RGB de **16 × 16 pixels**. Primeiro, um gerador **procedural** produz exemplos com olhos, boca, corpos e chifres variados. Essa etapa **não é IA**: ela serve para criar o conjunto de treinamento sem depender de arquivos externos. Depois, um **autoencoder variacional (VAE)** aprende padrões estatísticos presentes nesses exemplos. Ao fim do treinamento, sorteamos vetores latentes e o decodificador produz novas imagens. A saída pode ficar borrada, estranha ou parecida com os exemplos: isso faz parte do experimento.

### Arquitetura resumida

`imagem 16×16×3 → encoder (768→256→128) → μ, log(σ²) (12 dimensões) → amostragem z → decoder (12→128→256→768) → nova imagem`

A função objetivo combina erro de reconstrução e regularização do espaço latente:

\[ L = \mathrm{MSE}_{\mathrm{soma\ por\ imagem}} + \beta\,D_{KL}(q(z|x)\Vert p(z)) \]

Usamos **Adam**, um otimizador baseado em gradientes, para ajustar os pesos por retropropagação.

## 2. Preparação no Windows (PowerShell)

No diretório do projeto, crie um ambiente isolado:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Se a política do PowerShell impedir a ativação, use diretamente `\.venv\Scripts\python.exe` no lugar de `python` nos comandos seguintes. O primeiro download do PyTorch pode ser relativamente grande. Funciona com **CPU**, sem exigir GPU NVIDIA.

## 3. Executar — três comandos

### Gerar os exemplos de treinamento

```powershell
python monstros.py --quantidade 1200
```

Cria `dados/monstro_0000.png`, etc. Abra algumas imagens e note que são minúsculas: 16×16 pixels é uma escolha deliberada.

### Treinar a rede

```powershell
python treinar.py --epocas 35
```

A pasta `resultados/` conterá `epoca_001.png`, `epoca_005.png`, ... com imagens **geradas** pela rede durante o aprendizado, `perda.png` com a curva de treinamento e `modelo_vae.pt` com os pesos aprendidos. Os PNGs das épocas permitem observar o desenvolvimento do modelo.

### Gerar novos monstrinhos

```powershell
python gerar.py --quantidade 32
```

Abra `resultados/monstrinhos_novos.png`. As figuras são ampliadas por *nearest neighbor* para manter pixels quadrados.

## 4. O que cada arquivo ensina?

| Arquivo | Responsabilidade | Conceito |
|---|---|---|
| `monstros.py` | Desenha imagens de exemplo | Dataset, variação, distribuição dos dados |
| `vae.py` | Define encoder, amostragem e decoder | Redes neurais e representação latente |
| `treinar.py` | Calcula perda, gradientes e atualiza pesos | Retropropagação, Adam, épocas, avaliação visual |
| `gerar.py` | Sorteia vetores latentes e produz PNGs | Amostragem e geração |

## 5. Experimentos orientados

1. **Quantidade de exemplos:** compare 300 e 1200 imagens. O que muda nos resultados?
2. **Tempo de treinamento:** compare imagens da época 1, 5, 15 e 35. Há progresso? Sempre?
3. **Espaço latente:** treine com `--latente 2` e depois com `--latente 12`. O que a dimensionalidade influencia?
4. **Regularização:** compare `--beta 0.1` com `--beta 1.0`. Como muda o equilíbrio entre reconstrução e geração?
5. **Diversidade:** execute `python gerar.py --temperatura 0.5` e `--temperatura 1.5`. Qual parece produzir amostras mais variadas? Quais se parecem menos com os dados?
6. **Limitações:** explique por que um VAE com perda de pixels costuma gerar imagens suaves ou borradas.

**Importante:** cada execução de `treinar.py` sobrescreve o modelo e arquivos de épocas correspondentes. Copie `resultados/` antes de comparar experimentos diferentes.

## 6. Desafio para a turma

> Proponha uma modificação que torne os monstrinhos mais expressivos (um novo traço visual, uma regra do dataset, mais resolução ou uma variante da arquitetura). Justifique a hipótese, treine, apresente imagens antes/depois e analise os resultados. Não basta mostrar uma imagem bonita: a equipe deve explicar a relação entre dados, arquitetura e aprendizado.

### Entregáveis sugeridos

- 1 imagem com 16 ou 32 amostras geradas;
- 1 gráfico da perda de treinamento;
- 1 tabela resumindo pelo menos dois experimentos;
- comentário crítico sobre limitações, inclusive o fato de que os dados de treino são gerados artificialmente.

## 7. Problemas frequentes

- **`ModuleNotFoundError: torch`**: confirme que instalou as dependências no mesmo interpretador Python que executa o projeto.
- **Nenhuma imagem encontrada**: execute primeiro `python monstros.py`.
- **Modelos muito borrados**: é esperado em um VAE pequeno treinado com perda de reconstrução por pixel. Teste mais épocas ou diferentes `beta`, sem garantir resultados melhores.
- **Treino demorado**: comece com `python monstros.py --quantidade 300` e `python treinar.py --epocas 8` para validar tudo, depois rode o experimento completo.
- **Não há necessidade de internet durante a geração e o treino** após instalar os pacotes.

## 8. Fundamentação e referências

- KINGMA, D. P.; WELLING, M. *Auto-Encoding Variational Bayes*. ICLR, 2014. https://arxiv.org/abs/1312.6114
- GOODFELLOW, I.; BENGIO, Y.; COURVILLE, A. *Deep Learning*. MIT Press, 2016. https://www.deeplearningbook.org/
- KINGMA, D. P.; BA, J. *Adam: A Method for Stochastic Optimization*. ICLR, 2015. https://arxiv.org/abs/1412.6980

*Uso didático: imagens são sintéticas e não representam a geração de alta resolução obtida por modelos generativos modernos.*
