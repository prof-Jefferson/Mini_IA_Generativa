# Mini IA generativa de monstrinhos em pixel art

Projeto educacional em Python que treina um **autoencoder variacional (VAE)** para criar pequenos monstrinhos RGB de **16 × 16 pixels**. O objetivo é tornar visível o fluxo completo de um experimento generativo: criar dados, treinar uma rede e amostrar novas imagens.

O projeto funciona localmente e não baixa imagens prontas. `monstros.py` cria um conjunto sintético por regras de desenho; só depois o VAE aprende padrões desse conjunto. Portanto, o gerador procedural **não é a IA**: ele é a fonte dos dados de treinamento.

## Requisitos

- Windows 10 ou 11;
- Python 3.10 ou mais recente;
- espaço em disco para o PyTorch e para os artefatos do experimento;
- GPU não é obrigatória.

No PowerShell, dentro da pasta do projeto:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Se o PowerShell bloquear a ativação do ambiente, use o interpretador diretamente. Por exemplo:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Depois substitua `python` por `.\.venv\Scripts\python.exe` nos demais comandos.

## Os três comandos básicos

Execute-os nesta ordem:

```powershell
python monstros.py --quantidade 1200
python treinar.py --epocas 35
python gerar.py --quantidade 32
```

O primeiro comando cria o dataset, o segundo treina o VAE e salva seu checkpoint, e o terceiro usa o modelo treinado para produzir uma grade de monstrinhos inéditos.

## Execução rápida

Antes de iniciar um experimento longo, valide todo o fluxo com poucos dados e poucas épocas:

```powershell
python monstros.py --quantidade 300 --semente 42
python treinar.py --epocas 8 --semente 42
python gerar.py --quantidade 16 --semente 42
```

Esse teste confirma a instalação e a comunicação entre os scripts. A qualidade tende a ser inferior à de um treinamento mais longo.

## Testes automatizados

Com o ambiente virtual ativo e as dependências instaladas, execute:

```powershell
python -m unittest discover -s tests -v
```

A suíte verifica componentes do dataset e do VAE, além de percorrer uma integração curta de uma época. Os testes usam diretórios temporários e não substituem os resultados dos seus experimentos.

## Arquivos gerados

| Caminho | Conteúdo |
|---|---|
| `dados/monstro_0000.png`, ... | Imagens RGB de treinamento, cada uma com 16 × 16 pixels |
| `dados/manifesto.json` | Lista exata dos PNGs da última geração, sua semente e dimensões |
| `resultados/epoca_001.png`, ... | Amostras produzidas durante o treinamento |
| `resultados/perda.png` | Curva das perdas de treino e validação |
| `resultados/modelo_vae.pt` | Checkpoint: pesos e configuração necessários para gerar novas imagens |
| `resultados/configuracao.json` | Parâmetros usados e identidade SHA-256 do dataset |
| `resultados/historico.json` | Valores das perdas em cada época, úteis para análise ou novos gráficos |
| `resultados/manifesto_resultados.json` | Lista os artefatos pertencentes à execução atual |
| `resultados/monstrinhos_novos.png` | Grade final ampliada com *nearest neighbor*, preservando os pixels quadrados |

As pastas `dados/`, `resultados/` e `pesos/`, assim como checkpoints, são ignoradas pelo Git porque podem ser recriadas localmente.

## Parâmetros principais

Consulte todas as opções disponíveis com `python nome_do_script.py --help`.

### Dataset — `monstros.py`

- `--quantidade`: número de imagens criadas; padrão `1200`;
- `--saida`: pasta que recebe os PNGs; padrão `dados`;
- `--semente`: controla a geração procedural; padrão `42`.

O script sobrescreve arquivos de mesmo nome, mas não apaga PNGs antigos que excedam a nova quantidade. Em vez disso, atualiza `manifesto.json`; `treinar.py` usa somente os PNGs enumerados nele e avisa sobre resíduos. Assim, gerar 300 exemplos depois de 1200 não reutiliza silenciosamente os 900 antigos. Ainda é recomendável usar uma pasta de saída diferente para preservar cada dataset.

### Treinamento — `treinar.py`

- `--dados`: pasta de entrada; padrão `dados`;
- `--resultados`: pasta de saída; padrão `resultados`;
- `--epocas`: passagens pelo conjunto de treino; padrão `35`;
- `--tamanho-lote`: imagens processadas por atualização; padrão `64`;
- `--taxa-aprendizado`: tamanho dos ajustes feitos pelo Adam; padrão `0.001`;
- `--latente`: dimensões do espaço latente; padrão `12`;
- `--beta`: peso da regularização KL; padrão `1.0`;
- `--validacao`: fração reservada para validação; padrão `0.1`;
- `--semente`: controla divisão dos dados, inicialização e lotes; padrão `42`;
- `--amostras`: imagens mostradas nas grades de acompanhamento; padrão `16`;
- `--salvar-cada`: intervalo entre grades de acompanhamento; padrão `5`;
- `--dispositivo`: `auto`, `cpu` ou `cuda`; padrão `auto`.

O encoder transforma `16 × 16 × 3 = 768` valores em dois vetores — média (`mu`) e log-variância (`logvar`) — que parametrizam uma distribuição gaussiana diagonal no espaço latente. Uma amostra `z` passa pelo decoder, que reconstrói os 768 valores da imagem. A perda combina reconstrução por pixel e divergência KL:

```text
perda total = perda de reconstrução + beta × perda KL
```

O gráfico identifica a KL bruta, antes da multiplicação por `beta`. Caso a pasta de resultados já contenha outro experimento, o treino emite um aviso; use `manifesto_resultados.json` para saber quais grades pertencem à execução atual ou, preferencialmente, escolha outra pasta com `--resultados`.

### Geração — `gerar.py`

- `--modelo`: checkpoint de entrada; padrão `resultados/modelo_vae.pt`;
- `--saida`: arquivo PNG de saída; padrão `resultados/monstrinhos_novos.png`;
- `--quantidade`: monstrinhos na grade; padrão `32`;
- `--temperatura`: escala dos vetores latentes; padrão `1.0`. Valores baixos tendem a gerar amostras mais conservadoras; valores altos aumentam a variação e também a chance de artefatos;
- `--semente`: permite repetir a mesma amostragem; padrão `123`;
- `--escala`: fator de ampliação de cada pixel; padrão `8`;
- `--colunas`: número de colunas da grade; por padrão, é calculado automaticamente;
- `--dispositivo`: `auto`, `cpu` ou `cuda`; padrão `auto`.

A dimensão latente é recuperada automaticamente do checkpoint, impedindo que a geração dependa de repetir manualmente o valor de `--latente`.

## CPU e GPU

O projeto funciona normalmente usando somente a CPU da máquina, sem exigir uma GPU. Nesse caso, o treinamento pode levar alguns minutos, dependendo do processador, do número de imagens e das épocas.

Quando o PyTorch detecta uma GPU compatível com CUDA, `--dispositivo auto` pode usá-la automaticamente. A disponibilidade pode ser conferida no PowerShell:

```powershell
python -c "import torch; print('CUDA disponível:', torch.cuda.is_available())"
```

Ter uma placa NVIDIA não basta por si só: a instalação do PyTorch também precisa oferecer suporte ao ambiente CUDA. Consulte as instruções oficiais do PyTorch se quiser uma instalação específica para GPU. Use `--dispositivo cpu` para forçar CPU ou `--dispositivo cuda` para exigir CUDA; a segunda opção mostra um erro claro quando CUDA não está disponível.

## Reprodutibilidade

Use a mesma semente em todas as etapas para repetir, tanto quanto possível, o dataset, a inicialização e a amostragem:

```powershell
python monstros.py --quantidade 1200 --semente 42
python treinar.py --epocas 35 --semente 42
python gerar.py --quantidade 32 --semente 42
```

Uma semente igual melhora a comparação, mas não garante resultados numericamente idênticos entre versões do PyTorch, sistemas operacionais ou tipos diferentes de hardware. CPU e GPU podem executar algumas operações de modos distintos.

## Como comparar experimentos com segurança

O treinamento pode sobrescrever o checkpoint, a curva de perda e imagens de épocas anteriores. A forma mais segura de comparar configurações é dar uma pasta própria a cada execução:

```powershell
python treinar.py --beta 0.1 --resultados resultados_beta_01
python treinar.py --beta 1.0 --resultados resultados_beta_10
python gerar.py --modelo resultados_beta_01\modelo_vae.pt --saida resultados_beta_01\novos.png --semente 123
python gerar.py --modelo resultados_beta_10\modelo_vae.pt --saida resultados_beta_10\novos.png --semente 123
```

Se um experimento já tiver sido feito na pasta padrão, copie-a antes do próximo treinamento:

```powershell
Copy-Item -Recurse resultados resultados_experimento_01
```

Se o destino já existir, escolha outro nome. Cada pasta já inclui `configuracao.json` e `historico.json`; preserve-os junto das imagens e do checkpoint. Em uma comparação controlada:

1. mantenha dataset, semente, épocas e tamanho do lote iguais;
2. altere somente `--beta`;
3. salve cada pasta de resultados com um nome diferente;
4. compare a curva de perda e amostras obtidas com as mesmas sementes;
5. repita o experimento antes de concluir que uma configuração é melhor.

Não misture o checkpoint de um experimento com gráficos ou configurações de outro. `gerar.py` reconstrói a arquitetura a partir dos metadados armazenados no próprio checkpoint.

## Estrutura do projeto

```text
.
├── monstros.py          # cria o dataset sintético
├── vae.py               # define a rede, a perda e o formato do checkpoint
├── treinar.py           # treina, valida e registra resultados
├── gerar.py             # amostra o espaço latente e monta a grade final
├── tests/test_mvp.py    # testes unitários e integração curta
├── requirements.txt     # dependências Python
├── GUIA_PROJETO.md      # roteiro pedagógico e propostas de experimento
├── dados/               # criado localmente; imagens de treino
└── resultados/          # criado localmente; gráficos, grades e checkpoint
```

## Limitações importantes

- O dataset é **sintético e programado por regras**. O modelo aprende seus padrões e seus vieses; ele não aprendeu o conceito geral de “monstro” no mundo real.
- As imagens têm somente 16 × 16 pixels. Isso acelera o laboratório, mas limita detalhes, poses e expressões.
- Um VAE pequeno treinado com perda por pixel costuma produzir contornos suaves ou borrados. Isso é uma característica conhecida do método, não necessariamente um erro de execução.
- Mais épocas não garantem imagens melhores. O equilíbrio entre reconstrução e regularização depende de `beta`, dos dados e da capacidade da rede.
- A “criatividade” observada é amostragem estatística do espaço aprendido. Os resultados devem ser analisados em relação aos exemplos de treinamento.
- Este MVP é didático; ele não busca a qualidade de modelos generativos modernos nem é uma ferramenta de produção.

## Ideias de investigação

- comparar datasets com 300 e 1200 imagens;
- observar as grades das épocas 1, 5, 15 e 35;
- comparar espaços latentes de 2 e 12 dimensões;
- testar `beta` igual a `0.1` e `1.0`;
- gerar com temperaturas `0.5`, `1.0` e `1.5`;
- acrescentar um traço visual ao gerador procedural e analisar como o VAE reage.

Uma boa análise não escolhe apenas a imagem mais bonita: ela relaciona os dados, a configuração do modelo, a curva de perda e as amostras geradas.
