# Dicionário de dados

Fonte dos dados: [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud)

| Nome da coluna | Descrição             | Tipo de dado |
|----------------|-----------------------|--------------|
| `Time`         | Intervalo em segundos entre cada transação e a primeira | float     |
| `Vx`           | Resultado de PCA das variáveis originais | float   |
| `Amount`       | Valor de cada transação | float         |
| `Class`        | Classificação da transação entre legítima, 0, e fraudulenta, 1 | int        |

Na descrição da base de dados no Kaggle, são fornecidas algumas explicações acerca das variáveis:

- por questões de confidencialidade, não são disponibilizadas as identificações de boa
parte das variáveis originais;
- as representadas como V1, V2,... V28 são resultado de
uma transformação de análise de componentes principais - PCA, uma técnica utilizada para
condensar a informação contida em várias variáveis originais em um conjunto menor de
variáveis estatísticas (componentes) com uma perda mínima de informação. No trabalho
veremos algumas consequências dessa transformação em nossa análise;
