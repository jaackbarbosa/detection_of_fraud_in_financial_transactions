import pandas as pd

from sklearn.model_selection import cross_validate, GridSearchCV

from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler


RANDOM_STATE = 42


def construir_pipeline_modelo_classificacao(classificador, preprocessor=None):
    '''Contruir pipeline do modelo.

        A contrução da pipeline espera uma condição para ser gerada, se o parâmetro preprocessor receber um valor diferente de None e receber um preprocessor criado, a pipeline criará a chave nomeada, receberá as colunas transformadas e um classificador. Caso o preprocessor permaneça com valor None, a pipeline será criada e receberá apenas o algoritmo do classificador, excluindo o pré-processamento. A pipeline após o pré-processamento passa por uma subamostragen via RandomUnderSampler, para reduzir a quantidades de registros da classe majoritária até o balanceamento com a classe minoritária, ou adoção de outros parâmetros de proporção das classes.
    
    Parâmetros
    --------------------------------------------
    classificador : sklearn.pipeline.Pipeline
        Pipeline.   
    preprocessor : ColumnsTransformer().
        Transformações de cada categoria ou tipo das features independentes. 

    
    Retornos
    --------------------------------------------
    sklearn.pipeline.Pipeline
        Será criado uma pipeline com as escalas das distribuições ajustadas.
    
    '''
    if preprocessor is not None:
        pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("sampler", RandomUnderSampler(random_state=RANDOM_STATE)),
                ("clf", classificador)
            ]
        )
    else:
        pipeline = Pipeline(
            [
                ("sampler", RandomUnderSampler(random_state=RANDOM_STATE)),
                ("clf", classificador)
            ]
        )

    model = pipeline

    return model


def treinar_e_validar_modelo_classificacao(
    X,
    y,
    cv,
    classificador,
    preprocessor=None,
):
    '''Treinar e validar o modelo.

    Recebe uma pipeline com as escalas das distribuições ajustada da função construir_pipeline_modelo_classificacao() e extrai os valores de algumas métricas do modelo.
    
    Parâmetros
    --------------------------------------------
    X : pd.DataFrame
        Dataframe com as variáveis independentes.
    y : pd.DataFrame
        Dataframe com a variável dependente.
    cv : function
        Recebe uma classe de validação cruzada externa, já que diferentes métodos podem ser utilizados em modelos de classificação, como por exemplo, se haverá extratificação ou não, tudo isso para o melhor treinamento do modelo.
    classificador : sklearn.pipeline.Pipeline
        Pipeline.
    preprocessor : ColumnsTransformer(). Por padrão é None.
        Transformações de cada categoria ou tipo das features independentes. 
    
    
    Retornos
    --------------------------------------------
    dict
        Valores das métricas a partir de uma validação cruzada.
    
    '''
    model = construir_pipeline_modelo_classificacao(
        classificador,
        preprocessor,
    )

    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=[
            "accuracy",
            "balanced_accuracy",
            "f1",
            "precision",
            "recall",
            "roc_auc",
            "average_precision",
        ],
    )

    return scores


def grid_search_cv_classificador(
    classificador,
    param_grid,
    cv,
    preprocessor=None,
    return_train_score=False,
    refit_metric="roc_auc",
):
    '''Pesquisa em grade para encontrar os melhores parâmetros de criação para um modelo de classificação.
    
    Parâmetros
    ---------------------------------------------------------
    classificador : sklearn."alguma tipo de modelo"."algum modelo de classificação"
        Usa o classificador do modelo.
    param_grid : dict
        Dicionário com os hiperparâmetros e seus valores de teste.
    cv : function
        Recebe uma classe de validação cruzada externa, já que diferentes métodos podem ser utilizados em modelos de classificação, como por exemplo, se haverá extratificação ou não, tudo isso para o melhor treinamento do modelo.
    preprocessor : default: None.
        Constrói a estrutura de uma pipeline de processamento.
    return_train_score : boo. Default: False
        Retorna a pontuação e treinamento. Default: False.
    refit_metric : str
        Métrica de desempate, passado pelo usuário, para encontrar a diferença no desempenho do modelo. Default: roc_auc.

        
    Retorno
    ---------------------------------------------------------
    Encontra os melhores parâmetros, segundo os pré-processamentos, transformações e classificadores passados, para que assim crie o melhor modelo.
    
    
    '''
    model = construir_pipeline_modelo_classificacao(classificador, preprocessor)

    grid_search = GridSearchCV(
        model,
        cv=cv,
        param_grid=param_grid,
        scoring=[
            "accuracy",
            "balanced_accuracy",
            "f1",
            "precision",
            "recall",
            "roc_auc",
            "average_precision",
        ],
        refit=refit_metric,
        n_jobs=-1,
        return_train_score=return_train_score,
        verbose=1,
    )

    return grid_search


def organiza_resultados(resultados):
    '''Organiza os resultados das métricas do modelo e transforma em um dataframe.

    Alguns passos são realizados para chegar no dataframe ideal para trabalho. Primeiro foi criado uma coluna chamada time_seconds que recebe a soma de fit_time e score_time. O dataframe precisa passar por uma transposição, para inverter a posição das colunas com os índices, ter seus índices resetados, para que os índices virem uma coluna e essa coluna seja renomeada como "model", para pegar o nome do modelo que as métricas correspondem. 
    
    Será realizada uma explosão com os valores do dataframe, que atualmente teriam o nome do modelo e os valores das métricas agrupados em um np.array ocupando apenas uma célula e não cada um valor em uma linha. O método pd.explode recebe algo que se parece com uma lista para uma coluna, enquanto replica os valores do índice. O fatiamente ([1:]) será incluído para selecionar apenas os valor da segunda coluna em diante, já que o nome do modelo não consta como array e sim string, depois precisa transformar essas colunas passadas em uma lista. Como o fatiamento pegou apenas da segunda coluna em diante ([1:]), o nome da primeira coluna será replicado de acordo com os arrays das células que teve cada um de seus valores transformados em linhas. Novamente os índices foram resetados para corrigir um problema causado pela explosão, que é replicar o índice também, por exemplo, [0, 0, 0, 0, 1, 1, 1, 1] e deveria ser [0, 1, 2, 3, 4, 5, 6, 7].
    
    Um problema a ser corrigido é o tipo das features, que a partir da inclusão dos valores em um array, todas viraram do tipo 'object', já que o pandas não reconheceu tais valores como numéricos. Será usado um try (tentar/forçar) para transformar as colunas que possam ser numéricas em tipo numérica, e o except ValueError será usado para ignorar o erro, que vai acontecer em valores de texto como na coluna model, e continuar tentando transformar a próxima coluna em tipo numérico.
    
    
    Parâmetros
    --------------------------------------------
    resultados : dict
        Dicionário contendo as chaves e valores referentes às métricas do modelo ou dos modelos.
    
    Retornos
    --------------------------------------------
    pd.DataFrame
        Dataframe contendo os valores das métricas de cada modelo.
    
    '''
    for chave, valor in resultados.items():
        resultados[chave]["time_seconds"] = (
            resultados[chave]["fit_time"] + resultados[chave]["score_time"]
        )

    df_resultados = (
        pd.DataFrame(resultados).T.reset_index().rename(columns={"index": "model"})
    )

    df_resultados_expandido = df_resultados.explode(
        df_resultados.columns[1:].to_list()
    ).reset_index(drop=True)

    try:
        df_resultados_expandido = df_resultados_expandido.apply(pd.to_numeric)
    except ValueError:
        pass

    return df_resultados_expandido
