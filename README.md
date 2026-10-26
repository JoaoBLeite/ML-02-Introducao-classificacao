# ML-02-Introducao-classificacao

Uma **introdução prática e didática à classificação em Machine Learning**, usando Python e o dataset **Adult Income**.

Este é o segundo repositório de uma série de estudos. O primeiro, [ML-01-Introducao-regressao](https://github.com/JoaoBLeite/ML-01-Introducao-regressao), ensina **regressão** (prever um número). Aqui o objetivo é a **classificação**: prever uma **categoria**.

## 🎯 Objetivo

Este repositório é um caderno de estudos para quem está **começando em Machine Learning e em Python**. Ao longo de 7 notebooks, construímos passo a passo um modelo que prevê se uma pessoa **ganha mais de US$ 50 mil por ano**. Cada etapa é explicada do zero.

O foco não é só chegar a um bom modelo. É **entender o porquê** de cada decisão:

- 🎓 **Cada conceito novo** de Machine Learning é apresentado e contextualizado antes de ser usado.
- 💬 **Cada célula de código** tem comentários explicando tanto a ideia de ML quanto a sintaxe Python.
- 📌 **Cada análise termina com uma conclusão**: o que os números e gráficos estão dizendo.
- 🔁 **Cada notebook continua o anterior**: os problemas encontrados em um motivam as técnicas do próximo.
- 🧪 **As técnicas que não funcionam também aparecem**: testar e descobrir que algo não ajuda também é uma lição.

Os notebooks pressupõem os conceitos do ML-01 (treino e teste, `Pipeline`, validação cruzada, árvores e random forest), mas relembram cada um quando ele aparece.

### O dataset

O [Adult Income](https://www.openml.org/d/1590) (também chamado de *Census Income*) traz dados do **censo americano de 1994**. Cada uma das 48.842 linhas é **uma pessoa**, descrita por 14 colunas: idade, tipo de empregador, escolaridade, estado civil, profissão, papel na família, raça, sexo, ganho e perda de capital, horas trabalhadas por semana, país de origem e um peso amostral.

O alvo (`class`) diz se a renda anual da pessoa é `<=50K` ou `>50K`. Só **23,9%** das pessoas estão na segunda classe: as classes são **desbalanceadas**.

Em relação ao California Housing do ML-01, este dataset traz desafios novos: **colunas de texto** (categóricas), **valores ausentes**, classes desbalanceadas e questões de **justiça** (*fairness*) envolvendo sexo e raça.

## 📚 Trilha de estudos

| Notebook | O que você aprende |
|---|---|
| [01 · Análise exploratória](01_analise_exploratoria.ipynb) | Atributos numéricos × categóricos, classes desbalanceadas, valores ausentes informativos, taxa da classe por categoria, colunas redundantes e um primeiro olhar sobre fairness |
| [02 · Primeiro classificador](02_primeiro_classificador.ipynb) | Classificação × regressão, divisão estratificada, `StratifiedKFold`, acurácia, baseline, regressão logística, sigmoide, matriz de confusão e probabilidades |
| [03 · Pré-processamento de categóricas](03_preprocessamento_categoricas.ipynb) | Codificação ordinal × one-hot, imputação, categorias raras e desconhecidas, `ColumnTransformer` e quando o logaritmo **não** ajuda |
| [04 · Métricas e limiar de decisão](04_metricas_e_limiar.ipynb) | Precisão, recall, F1, limiar de decisão e custo dos erros, curvas ROC e precisão-recall, AUC e `TunedThresholdClassifierCV` |
| [05 · Modelos de árvore](05_modelos_de_arvore.ipynb) | Árvore de decisão e impureza de Gini, random forest e `min_samples_leaf`, `HistGradientBoostingClassifier` com categorias nativas, ajuste de hiperparâmetros e dependência parcial |
| [06 · Probabilidades e desbalanceamento](06_probabilidades_e_desbalanceamento.ipynb) | Calibração, curva de calibração, log loss e Brier score, `class_weight="balanced"` (e por que ele não ajudou) e `CalibratedClassifierCV` |
| [07 · Avaliação final e justiça](07_avaliacao_final_e_justica.ipynb) | Avaliação no teste, permutation importance, métricas de fairness por sexo e raça, variáveis substitutas (*proxies*), salvar o modelo, limitações e uso responsável |

### Evolução dos resultados (no conjunto de teste)

| Notebook | Modelo | AUC | F1 | Recall | Acurácia |
|---|---|---|---|---|---|
| 02 | Baseline (sempre `<=50K`) | 0,500 | 0,000 | 0% | 76,1% |
| 02 | Regressão logística só com as numéricas | 0,832 | 0,506 | 39% | 81,9% |
| 03 | Regressão logística + categóricas (one-hot) | 0,905 | 0,665 | 60% | 85,5% |
| 04 | Regressão logística + limiar ajustado | 0,905 | 0,691 | 76% | 83,8% |
| 05 | Gradient boosting | 0,930 | 0,716 | 65% | 87,6% |
| 07 | **Gradient boosting + limiar ajustado (modelo final)** | **0,930** | **0,731** | **73%** | **87,0%** |

Repare que a maior **acurácia** não é a do modelo final: o limiar foi escolhido para o **F1**, que valoriza encontrar as pessoas de renda alta. O notebook 04 explica por quê.

## 🛠️ Configurando o ambiente

### Pré-requisitos

- **Python 3.10 ou mais recente.** Para conferir sua versão:
  ```bash
  python3 --version
  ```
- **Git**, para baixar o repositório.

### 1. Baixe o repositório

```bash
git clone https://github.com/JoaoBLeite/ML-02-Introducao-classificacao.git
cd ML-02-Introducao-classificacao
```

### 2. Crie um ambiente virtual

Um **ambiente virtual** é uma pasta isolada com o Python e as bibliotecas **só deste projeto**. Assim, as versões instaladas aqui não interferem em outros projetos do seu computador, e vice-versa.

```bash
python3 -m venv .venv
```

Isso cria a pasta `.venv/` dentro do projeto. Ela não vai para o Git, porque já está no `.gitignore`.

> 💡 **No Ubuntu/Debian**, se aparecer um erro dizendo que o `venv` não está disponível, instale-o com:
> `sudo apt install python3-venv`

### 3. Ative o ambiente virtual

O ambiente precisa ser **ativado em cada novo terminal** que você abrir.

| Sistema | Comando |
|---|---|
| Linux / macOS | `source .venv/bin/activate` |
| Windows (PowerShell) | `.venv\Scripts\Activate.ps1` |
| Windows (Prompt de Comando) | `.venv\Scripts\activate.bat` |

Quando o ambiente está ativo, o nome `(.venv)` aparece no começo da linha do terminal. Para desativar, use `deactivate`.

### 4. Instale as bibliotecas

Com o ambiente **ativado**:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

O arquivo `requirements.txt` lista as versões exatas usadas no projeto. As principais são:

| Biblioteca | Para quê |
|---|---|
| `pandas` | Manipular tabelas de dados |
| `numpy` | Cálculos numéricos |
| `matplotlib` | Gráficos |
| `scikit-learn` | Modelos de Machine Learning, métricas e o download do dataset |
| `scipy` | Distribuições usadas na busca aleatória de hiperparâmetros |
| `joblib` | Salvar e carregar o modelo final |
| `jupyter` | Rodar os notebooks |

Para conferir se deu tudo certo:

```bash
python -c "import pandas, sklearn, matplotlib; print('Tudo instalado!')"
```

## ▶️ Como executar

### 1. Abra o Jupyter

Com o ambiente virtual **ativado**, na pasta do projeto:

```bash
jupyter notebook
```

O navegador vai abrir com a lista de arquivos do projeto. Se preferir uma interface mais completa, use `jupyter lab`.

> 💡 **Usa o VS Code?** Abra a pasta do projeto, abra um notebook e, no canto superior direito, escolha o *kernel* do Python que está em `.venv`.

### 2. Siga os notebooks em ordem

Comece pelo **01** e siga até o **07**. Cada notebook pressupõe os conceitos dos anteriores.

Dentro de um notebook:
- **`Shift + Enter`** executa a célula atual e passa para a próxima.
- Execute as células **de cima para baixo**: uma célula pode depender de variáveis criadas nas anteriores.
- Para rodar tudo do zero: menu **Kernel → Restart Kernel and Run All Cells**.
- Enquanto uma célula está rodando, aparece `[*]` à esquerda dela.

### Observações

- **Na primeira execução**, o dataset é baixado do [OpenML](https://www.openml.org) e fica salvo em `~/scikit_learn_data`. Nas próximas vezes, não é preciso conexão.
- **O notebook 05 demora cerca de 2 a 3 minutos**, porque treina centenas de árvores com validação cruzada. Os outros rodam em menos de 1 minuto.
- **O notebook 07 salva o modelo final** na pasta `modelos/` (cerca de 400 KB). Essa pasta não vai para o Git: ela é recriada sempre que o notebook 07 é executado.
- Os notebooks já estão salvos **com os resultados**, então dá para lê-los aqui no GitHub mesmo sem executá-los.

## 📁 Estrutura do projeto

```
ML-02-Introducao-classificacao/
├── 01_analise_exploratoria.ipynb
├── 02_primeiro_classificador.ipynb
├── 03_preprocessamento_categoricas.ipynb
├── 04_metricas_e_limiar.ipynb
├── 05_modelos_de_arvore.ipynb
├── 06_probabilidades_e_desbalanceamento.ipynb
├── 07_avaliacao_final_e_justica.ipynb
├── requirements.txt      # bibliotecas e versões do projeto
└── modelos/              # criada pelo notebook 07 (não versionada)
```

## ⚠️ Limitações

Este projeto é para **aprender**, não para tomar decisões sobre pessoas reais. Os dados são de 1994 e só dos Estados Unidos, e o modelo **reproduz as desigualdades** de renda entre homens e mulheres presentes nesses dados. O notebook 07 mede essas diferenças, mostra por que remover a coluna `sex` não resolve o problema e discute os cuidados com o uso responsável de modelos como este.
