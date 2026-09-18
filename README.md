# Active Ants

Aplicação de dados de ponta a ponta sobre transporte cooperativo em colônias de formigas: da simulação ao dashboard, passando por ETL, banco em nuvem e modelos de machine learning.

O trabalho é inspirado no estudo de Gelblum et al. sobre como formigas coordenam o carregamento de uma carga grande demais para qualquer indivíduo. A pergunta central é quando o grupo passa de movimento desorganizado para transporte coletivo eficiente.

**Demo ao vivo: [activeants.streamlit.app](https://activeants.streamlit.app)**

---

## Arquitetura

```
simulação  →  ETL  →  AWS RDS (PostgreSQL)  →  Streamlit
                ↓
            AWS S3 (modelos treinados e arquivos)
```

O dashboard não lê arquivos locais: cada interação dispara uma query no banco em nuvem, e os modelos treinados são carregados do S3. Credenciais ficam fora do repositório, em `secrets.toml` (veja `secrets.toml.example`).

---

## Pipeline de dados

**Extração** As simulações geram séries temporais com a posição e o estado de cada formiga a cada passo, além das grandezas coletivas do sistema.

**Transformação** Tratamento de tipos, cálculo de variáveis derivadas e validação de consistência antes da carga.

**Carga** Criação do schema e inserção em lote no PostgreSQL da AWS RDS, com mais de 17.500 registros validados por query.

---

## Modelos

| Modelo | Tarefa | Algoritmo |
|---|---|---|
| Classificador de regime | Identificar em qual regime o sistema se encontra a partir das variáveis observadas | Random Forest |
| Detector de ponto crítico | Localizar a transição entre regimes | Regressão |
| Clusterização de trajetórias | Agrupar padrões de movimento coletivo | KMeans |
| Rede neural | Aprender a dinâmica de alinhamento diretamente dos dados, sem partir das equações do modelo | PyTorch, treinada sobre 668 mil frames |

Cada modelo foi validado contra a predição teórica do sistema, e as métricas estão documentadas no próprio dashboard.

---

## O dashboard

Quatro abas com sliders de parâmetros. Alterar um parâmetro dispara nova consulta ao banco e os gráficos Plotly recalculam em tempo real. Dá para percorrer o espaço de parâmetros e ver onde o comportamento coletivo muda.

---

## Rodando localmente

```bash
git clone https://github.com/leo-lopes/Ants.git
cd Ants
pip install -r requirements.txt

# copie o exemplo e preencha com suas credenciais
cp secrets.toml.example .streamlit/secrets.toml

streamlit run app.py
```

Para testar a conexão com a AWS antes de subir o app:

```bash
python test_aws_connection.py
```

---

## Stack

`Python` · `PostgreSQL` · `AWS RDS` · `AWS S3` · `Scikit-learn` · `PyTorch` · `Pandas` · `NumPy` · `Streamlit` · `Plotly`

---

## Contexto

Projeto derivado da minha pesquisa de doutorado em física computacional na UFMG, sobre modelos de matéria ativa e sistemas complexos fora do equilíbrio.

**Leonardo Lopes** · [GitHub](https://github.com/leo-lopes) · [LinkedIn](https://linkedin.com/in/leo-slopes)
