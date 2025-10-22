import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(palette="bright")

PALETTE = "coolwarm"
SCATTER_ALPHA = 0.2

def plot_comparar_metricas_modelos(df_resultados, salvar_imagem=False):
    '''Compara as métricas dos modelos de classificação.
    
    Parâmetro:
    --------------------------------------------------------
    df_resultados: pandas.core.frame.DataFrame
        Usa um DataFrame com as métricas dos modelos.
    salvar_imagem : boo. Default=False
        Salva a imagem gerada para uso externo.
    
    retorno
    --------------------------------------------------------
    Boxplots com a comparação do desempenho dos modelos em cada métrica.
    
    '''
    fig, axs = plt.subplots(4, 2, figsize=(9, 9), sharex=True)

    comparar_metricas = [
        "time_seconds",
        "test_accuracy",
        "test_balanced_accuracy",
        "test_f1",
        "test_precision",
        "test_recall",
        "test_roc_auc",
        "test_average_precision",
    ]

    nomes_metricas = [
        "Tempo (s)",
        "Acurácia",
        "Acurácia balanceada",
        "F1",
        "Precisão",
        "Recall",
        "AUROC",
        "AUPRC",
    ]

    for ax, metrica, nome in zip(axs.flatten(), comparar_metricas, nomes_metricas):
        sns.boxplot(
            x="model",
            y=metrica,
            data=df_resultados,
            ax=ax,
            showmeans=True,
        )
        ax.set_title(nome)
        ax.set_ylabel(nome)
        ax.tick_params(axis="x", rotation=90)

    plt.tight_layout()

    if salvar_imagem:
        return plt.savefig("comparar_metricas_modelos.png", bbox_inches="tight")

    plt.show()
