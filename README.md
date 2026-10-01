# Transaction Risk Scoring — Fraud Analytics

## Detecção de Fraudes com Machine Learning e Análise de Risco Transacional

Este projeto apresenta o desenvolvimento de um **modelo de score de risco transacional**, explorando como probabilidades geradas por Machine Learning podem ser transformadas em decisões e políticas de risco.

O objetivo não é apenas classificar uma transação como fraude ou legítima, mas compreender uma questão mais importante:

> **Como transformar o score produzido por um modelo em uma política de risco capaz de equilibrar detecção de fraude, falsos positivos e impacto financeiro?**

---

## 🎯 Objetivo

Construir um pipeline simplificado de Fraud Analytics capaz de:

- criar características transacionais e comportamentais;
- treinar um modelo de classificação;
- gerar scores de risco;
- analisar a distribuição dos scores;
- avaliar Precision e Recall;
- estudar diferentes thresholds;
- medir falsos positivos e falsos negativos;
- estimar o valor financeiro de fraudes capturadas e não detectadas;
- relacionar performance estatística com impacto de negócio.

O fluxo desenvolvido foi:

```text
Dados Transacionais
        ↓
Feature Engineering
        ↓
Train / Test
        ↓
Regressão Logística
        ↓
Probabilidade de Fraude
        ↓
Score de Risco
        ↓
Distribuição do Score
        ↓
Threshold
        ↓
Precision × Recall
        ↓
Impacto Financeiro
        ↓
Política de Risco
```

---

## 🧪 Dataset

Foi utilizado um **dataset sintético com 5.000 transações**, desenvolvido especificamente para fins educacionais e experimentais.

Cada linha representa uma transação.

Entre as variáveis utilizadas estão:

- valor da transação;
- idade da conta;
- ticket médio do cliente;
- quantidade de tentativas recentes;
- utilização de dispositivo novo;
- compra realizada durante a madrugada;
- indicador de risco da loja;
- relação entre o valor atual da transação e o comportamento histórico do comprador.

A variável-alvo é:

```text
fraude

0 = transação legítima
1 = fraude
```

### Importante

A distribuição de fraude foi deliberadamente elevada para facilitar a visualização do comportamento das métricas, scores e thresholds.

Portanto, os resultados deste experimento **não representam taxas reais de fraude em ambientes de produção**.

---

## ⚙️ Feature Engineering

Uma das features comportamentais desenvolvidas foi a relação entre o valor da transação atual e o ticket médio histórico do cliente:

```python
df["razao_valor_ticket"] = (
    df["valor_transação"] /
    (df["ticket_medio_cliente"] + 1)
)
```

Essa variável permite avaliar a transação dentro do contexto comportamental do comprador.

Por exemplo:

```text
Valor da transação = R$ 900
Ticket médio       = R$ 150

Razão ≈ 6
```

Nesse cenário, a compra possui valor aproximadamente seis vezes superior ao comportamento médio daquele cliente.

Em Fraud Analytics, **contexto comportamental pode ser mais informativo do que analisar somente valores absolutos**.

---

## 🤖 Modelo

Como baseline foi utilizada uma:

**Logistic Regression**

A escolha permite construir um primeiro modelo relativamente simples e interpretável antes da comparação com algoritmos mais sofisticados.

O dataset foi dividido em:

```text
70% → treinamento
30% → teste
```

utilizando estratificação da variável-alvo para preservar aproximadamente a distribuição das classes.

---

## 📊 Score de Risco

Em vez de utilizar apenas a classificação binária:

```text
Fraude
ou
Legítima
```

o modelo utiliza a probabilidade estimada:

```python
score_risco = modelo.predict_proba(X_test)[:, 1]
```

Exemplo:

```text
Transação A → 0.08
Transação B → 0.37
Transação C → 0.72
Transação D → 0.94
```

Quanto maior o score, maior o risco estimado pelo modelo.

Isso permite transformar a saída do Machine Learning em um instrumento para definição de políticas de risco.

---

## 🔍 Matriz de Confusão

Utilizando inicialmente threshold de `0.50`, o modelo apresentou:

```text
TN = 948
FP = 85
FN = 269
TP = 198
```

Interpretação:

**True Positive (TP)**  
Fraudes corretamente identificadas.

**True Negative (TN)**  
Transações legítimas corretamente classificadas.

**False Positive (FP)**  
Transações legítimas classificadas como fraude.

**False Negative (FN)**  
Fraudes que não foram detectadas pelo modelo.

No contexto de prevenção a fraudes:

```text
FP → possível fricção sobre clientes legítimos

FN → fraude que atravessou o mecanismo de detecção
```

Esse trade-off é fundamental para definição da política de risco.

---

## 🎯 Precision × Recall

Com threshold `0.50`, foram obtidos aproximadamente:

```text
Precision = 69,96%
Recall    = 42,40%
```

### Precision

Responde:

> Das transações classificadas como fraude, quantas realmente eram fraudulentas?

### Recall

Responde:

> De todas as fraudes existentes, quantas foram detectadas pelo modelo?

A análise demonstra por que métricas isoladas não devem determinar uma política antifraude.

---

## 🎚️ Análise de Threshold

Foram avaliados diferentes pontos de corte entre `0.20` e `0.80`.

| Threshold | Precision | Recall |
|---:|---:|---:|
| 0.20 | 45,77% | 84,58% |
| 0.30 | 52,43% | 67,02% |
| 0.40 | 59,29% | 53,32% |
| 0.50 | 69,96% | 42,40% |
| 0.60 | 79,56% | 30,84% |
| 0.70 | 88,89% | 23,98% |
| 0.80 | 90,14% | 13,70% |

O comportamento observado evidencia o trade-off:

```text
Threshold menor
        ↓
Maior Recall
        ↓
Maior captura de fraude
        ↓
Mais falsos positivos
```

Enquanto:

```text
Threshold maior
        ↓
Maior Precision
        ↓
Menos falsos positivos
        ↓
Mais fraudes não detectadas
```

Portanto, **0.50 não deve ser tratado automaticamente como o threshold ideal apenas por ser o valor convencional utilizado na classificação binária**.

---

## 💰 Impacto Financeiro

Além da quantidade de fraudes detectadas, foi analisado o valor financeiro associado às transações fraudulentas.

| Threshold | Recall | Captura do valor fraudulento | Fraude capturada | Fraude não detectada |
|---:|---:|---:|---:|---:|
| 0.20 | 84,58% | 94,95% | R$ 162.828,23 | R$ 8.660,94 |
| 0.30 | 67,02% | 84,93% | R$ 145.648,76 | R$ 25.840,41 |
| 0.40 | 53,32% | 75,60% | R$ 129.648,96 | R$ 41.840,21 |
| 0.50 | 42,40% | 66,25% | R$ 113.611,44 | R$ 57.877,73 |
| 0.60 | 30,84% | 55,77% | R$ 95.641,29 | R$ 75.847,88 |
| 0.70 | 23,98% | 47,05% | R$ 80.689,83 | R$ 90.799,34 |
| 0.80 | 13,70% | 31,30% | R$ 53.683,14 | R$ 117.806,03 |

Um resultado particularmente interessante aparece no threshold `0.20`.

O modelo captura:

```text
84,58% da quantidade de fraudes

mas

94,95% do valor financeiro fraudulento
```

Isso demonstra que **Recall e captura financeira não necessariamente contam a mesma história**.

Um modelo pode não capturar todas as ocorrências e, ainda assim, capturar uma parcela significativamente maior do valor financeiro em risco.

---

## 💵 Função de Custo

Para aproximar a análise da tomada de decisão, foi criada uma função de custo simplificada.

Para fins exclusivamente experimentais, foi considerado:

```text
Custo por falso positivo = R$ 5
```

e a fraude não detectada foi incorporada ao custo pelo valor da transação.

Os resultados foram:

| Threshold | Custo total simulado |
|---:|---:|
| 0.20 | R$ 11.000,94 |
| 0.30 | R$ 27.260,41 |
| 0.40 | R$ 42.695,21 |
| 0.50 | R$ 58.302,73 |
| 0.60 | R$ 76.032,88 |
| 0.70 | R$ 90.869,34 |
| 0.80 | R$ 117.841,03 |

Sob **essas premissas específicas**, o threshold `0.20` apresentou o menor custo total simulado.

Isso não significa que `0.20` seja um threshold universalmente ideal.

Em um ambiente real, seria necessário considerar fatores como:

- custo real de chargeback;
- recuperação financeira;
- custo de revisão manual;
- abandono e experiência do cliente;
- perda de conversão;
- capacidade operacional;
- regras específicas do negócio;
- diferentes perfis e verticais de risco.

A escolha do threshold é, portanto, uma decisão que combina **Machine Learning, risco e estratégia de negócio**.

---

## 🛡️ Do Score à Política de Risco

Em produção, o score não precisa resultar simplesmente em:

```text
Fraude → bloquear
Legítima → aprovar
```

Ele pode alimentar diferentes políticas:

```text
Score baixo
      ↓
Aprovação

Score intermediário
      ↓
Monitoramento / autenticação adicional

Score elevado
      ↓
Revisão

Score crítico
      ↓
Intervenção conforme política de risco
```

Os limites e tratamentos devem ser calibrados com dados históricos, custos, capacidade operacional, experiência do cliente e apetite de risco.

---

## 🧠 Principal Aprendizado

O objetivo de um modelo antifraude não é simplesmente maximizar uma métrica estatística.

O desafio é encontrar um equilíbrio entre:

```text
Fraude capturada
        ×
Fraude não detectada
        ×
Falsos positivos
        ×
Impacto financeiro
        ×
Experiência do cliente
```

O score produzido pelo modelo é apenas uma parte do sistema.

**A decisão surge quando o score é conectado a uma política de risco.**

---

## 🔬 Próxima Evolução — Graph-Enhanced Fraud Detection

Modelos tabulares analisam características importantes de cada transação.

Porém, fraude frequentemente possui também uma dimensão relacional.

Uma transação aparentemente normal pode fazer parte de uma estrutura envolvendo:

```text
Comprador ─── Cartão
    │            │
    │            │
Dispositivo ─── IP
    │
    │
   Loja
```

Isso abre uma próxima linha experimental para o projeto:

### Graph Intelligence + Machine Learning

A proposta futura será representar entidades e relacionamentos utilizando **Neo4j** e investigar features relacionais, por exemplo:

- quantidade de compradores associados ao mesmo dispositivo;
- cartões compartilhando dispositivos;
- contas relacionadas ao mesmo IP;
- densidade de conexões suspeitas;
- comunidades;
- centralidade;
- distância para entidades previamente associadas a fraude;
- padrões relacionais difíceis de representar diretamente em estruturas tabulares.

Essas informações poderão ser transformadas em **graph features** e incorporadas ao modelo de Machine Learning.

A pergunta experimental será:

> **Features relacionais extraídas do grafo acrescentam poder preditivo ao modelo tabular de detecção de fraudes?**

A hipótese deverá ser testada comparando:

```text
Modelo A
Features tabulares
        ↓
ML

versus

Modelo B
Features tabulares
        +
Features relacionais
        ↓
Neo4j / Graph Analytics
        ↓
ML
```

A comparação poderá considerar:

- Precision;
- Recall;
- F1-score;
- ROC-AUC;
- PR-AUC;
- falsos positivos;
- falsos negativos;
- captura financeira;
- impacto sobre a política de risco.

O objetivo não será presumir que grafos melhoram o modelo, mas **medir experimentalmente se informação relacional acrescenta sinal preditivo relevante**.

---

## 🛠️ Tecnologias

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Jupyter Notebook
- VS Code

### Evolução planejada

- Neo4j
- Cypher
- Graph Data Science
- Graph Features
- Machine Learning

---

## 📌 Status

**Fase 1 — Transaction Risk Scoring:** concluída.

**Fase 2 — Graph-Enhanced Fraud Detection:** planejada como evolução experimental.

---

## Autor

**Alexandre Lotti**

Data Science | Fraud Analytics | Graph Intelligence

`<!LOTTI>`
