import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import joblib

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from scipy.stats import shapiro, ttest_ind, mannwhitneyu





# Configuração inicial da página
st.set_page_config(
    page_title="Meu Dashboard Analítico",
    page_icon="📊",
    layout="wide"
)

st.markdown(
    """
    <style>
        .stApp {
            background-color: #f0f0f0;  /* Cor de fundo cinza claro */
        }
    </style>
    """, 
    unsafe_allow_html=True
)

# Carregar os dados
@st.cache_data
def load_data():
    data = pd.read_csv('arquivos/oasis_cross-sectional.csv')
    data = data.dropna(subset=['MMSE', 'CDR'])
    data.drop('Delay', axis=1, inplace=True)
    return data

data = load_data()

@st.cache_data
def load_data2():
    data = pd.read_csv('arquivos/oasis_cross-sectional.csv')
    # Filtrar os dados onde MMSE e CDR não são nulos
    data = data.dropna(subset=['MMSE', 'CDR'])
    data.drop('Delay', axis=1, inplace=True)
    data["CDR"] = data["CDR"].astype(str)

    return data

data2 = load_data2()








# Criando as abas
title, tab1,  estudo, tab2, tab3, tab4, tab_pred, tab_pca, tab_simulacao = st.tabs(["-", "📌 Introdução ao Problema", "Estudo",  
                                                  "📊 Introdução aos Dados", "📈 Análises", "Conclusões",
                                                  "Predições", "PCA e Agrupamento", "Simulações"])

with title:
    st.markdown(
        """
        <h1 style="text-align: center; font-size: 40px; color: #0073e6;">
            Análise de Alzheimer: Diagnóstico e Tendências
        </h1>
        """,
        unsafe_allow_html=True
    )
    col1, col2, col3 = st.columns([1, 1, 1])

    with col2:
        st.image("image_title.webp", use_container_width =False,  width=400)


with tab1:
    
    col1, col2 = st.columns([1, 1])

    with col1:
        
        # st.subheader("Introdução ao Problema")
        with st.container(border=True):  
            st.write("""
                    - A doença de Alzheimer é uma doença cerebral degenerativa sem cura
                    - É caracterizada por atrofia progressiva do córtex cerebral
                    - Causa perda de memória, aumento dos déficits cognitivos e potencial perda das funções motoras
                    - Com um diagnóstico precoce, a progressão pode ser retardada e os sintomas tratados.
            """)

            # Criando três colunas: esquerda, centralizada e direita
        col_empty1, col_img, col_empty2 = st.columns([1, 2, 3])

        with col_img:
            st.image("brain_atrophy.jpg", caption="Atrofia Cerebral", width=400)
        
    with col2:  
        # Você pode adicionar imagens, gráficos ou outros elementos
        
        st.image("sinais-de-alzheimer.jpg", caption="Sinais de Alzheimer", width=500)


with estudo: 

    st.subheader("Descrição da coleta dos dados:")

    with st.container(border=True):  # Disponível no Streamlit >= 1.29.0
            st.write("""
                    - 416 pessoas participaram do estudo;
                    - Idades entre 18 e 96 anos;
                    - Para cada pessoa, são incluídas dados de ressonâncias magnéticas individuais, obtidas em sessões de varredura única;
                    - Todos destros;
                    - Inclui homens e mulheres;
                    - Um conjunto de dados de confiabilidade é incluído contendo 20 sujeitos não dementes fotografados em uma visita subsequente dentro de 90 dias de sua sessão inicial.
            """)

with tab2:

    
    # st.header("Introdução aos Dados")
    
    # Criando duas colunas (a primeira será mais larga para a imagem)
    col1, col2 = st.columns([1, 1])  # Proporção 2:3 (ajuste conforme necessário)
    
    

    with col1:
        info_variaveis_ = {
        "Variável": ["ID", "M/F", "Hand", "Age", "Educ", "SES", "eTIV", "ASF", "nWBV",  "MMSE", "CDR"],
        "Definição": [
            "Identification", "Gender", "Dominant Hand", "Age in years", 
            "Education Level", "Socioeconomic Status", 
            "Estimated Total Intracranial Volume",
            "Atlas Scaling Factor",
            "Normalize Whole Brain Volume",
            "Mini Mental State Examination",  
            "Clinical Dementia Rating"
        ],
        "Valores": [
            " ", "M = Male, F = Female", "R = Right, L = Left", " ", 
            "1 = < Ensino Médio\n2 = Ensino Médio Completo\n3 = Ensino Superior Incompleto\n4 = Ensino Superior Completo\n5 = Pós-Graduação",
            "1 = Classe Baixa\n2 = Classe Média Baixa\n3 = Classe Média\n4 = Classe Média Alta\n5 = Classe Alta",
            
            
            " ", " ", " ", " 0 - 30 ", 
            "0 = Sem Demência\n0.5 = Demência Muito Leve\n1 = Demência Leve\n2 = Demência Moderada",
        ]
        }
        info_variaveis = pd.DataFrame(info_variaveis_)
        st.subheader("Tabela de Variáveis")
        # st.dataframe(info_variaveis, use_container_width=True,hide_index=True)
        st.markdown(
            """
            <style>
            table {
                width: 80%;
                
            }
            th {
                color: #0073e6 !important; /* Força a mudança da cor */
                font-weight: bold; /* Deixa o texto em negrito */
                white-space: nowrap; /* Mantém o texto em uma linha */
                border: 1px solid black !important;
            }
            td {
                white-space: pre-wrap;
                border: 1px solid black !important;
            }
            </style>
            """,
            unsafe_allow_html=True
            )
        
        st.table(info_variaveis)  

    with col2:
        st.subheader("Exemplo de Mini Mental State Examination")

        st.image("mmse.jpg", use_container_width=False, width=800)

with tab3:
    
    # st.header("Análise de Correlação")

    subtab1, subtab2, subtab3, subtab4 = st.tabs(["Distribuição de CDR", "Análise de Correlação", "vWBV vs CDR", "MMSE vs CDR"])
    
    

    with subtab1:

        col1, col2 = st.columns([2, 1])
        
        with col1:
            cdr_table = data.groupby(['CDR']).size().reset_index(name='Count')
            
            cdr_descricao = {
                    0.0: 'Sem demência',
                    0.5: 'Demência muito leve',
                    1.0: 'Demência leve',
                    2.0: 'Demência moderada'
                }

                        # Substituir os valores da coluna CDR
            cdr_table['Interpretação'] = cdr_table['CDR'].map(cdr_descricao)

            cdr_table = cdr_table[['CDR','Interpretação','Count']]

            #st.dataframe(
            #        cdr_table.style
            #        .background_gradient(cmap='Blues', subset=['Count'])
            #        .format({'Count': '{:,}', 'CDR': '{:.1f}'}),  # Formata números com separador de milhar
            #        use_container_width=False,  # Não usar toda a largura
            #        hide_index=True
            #    )
        
            plt.figure(figsize=(4, 2))
            ax = sns.barplot(x='Count', y='Interpretação', data=cdr_table, hue='Interpretação', palette='viridis', dodge=False)

            # Adicionando os valores de contagem no final de cada barra
            for index, row in cdr_table.iterrows():
                ax.text(row['Count'] + 1, index, str(row['Count']), color='black', va='center')

            # Remover as bordas do gráfico
            for spine in ax.spines.values():
                spine.set_visible(False)

            # Adicionando títulos e rótulos
            plt.title('Distribuição de Casos por Tipo de Demência')
            plt.xlabel('Contagem')
            plt.ylabel('Tipo de Demência')

            # Exibindo o gráfico no Streamlit
            st.pyplot(plt, use_container_width=False)
        
        with col2:
            st.markdown("""
            - 100 dos sujeitos incluídos com mais de 60 anos foram clinicamente diagnosticados com doença de Alzheimer muito leve a moderada.
            - Um conjunto de dados de confiabilidade é incluído contendo 20 sujeitos não dementes.
            """)
        

    with subtab2:
    
        # Criar duas colunas (1:2 - a figura ocupará 1/3 do espaço)
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Cria a figura menor
            fig, ax = plt.subplots(figsize=(5, 3))  # Tamanho reduzido
            corr_spearman = data.corr(method='spearman', numeric_only=True)
            mask = np.triu(np.ones_like(corr_spearman, dtype=bool))
            
            sns.heatmap(
                corr_spearman,
                mask=mask,
                annot=True,
                fmt=".2f",
                cmap='coolwarm',
                center=0,
                vmin=-1,
                vmax=1,
                cbar_kws={"shrink": 0.6},
                ax=ax,
                annot_kws={"size": 6} 
            )
            
            # Ajustar fonte da colorbar
            cbar = ax.collections[0].colorbar
            cbar.ax.tick_params(labelsize=6) 

            ax.set_title("Correlação (Spearman)", fontsize=10)
            plt.xticks(rotation=45, fontsize=6)
            plt.yticks(rotation=0, fontsize=6)
            plt.tight_layout()
            
            st.pyplot(fig, use_container_width=False)
        
        with col2:
            st.markdown("""
            **Mapa de Correlação de Spearman**
            
            Este gráfico mostra as relações entre as variáveis numéricas:
            
            - **Correlação Positiva (+1)**
            - **Correlação Negativa (-1)**
            - **Sem Correlação (0)**
            """)

    with subtab3:

        st.title("Análise Estatística de nWBV entre Pacientes com e sem Demência")

        fig2 = px.scatter(
        data2,
        x='Age',
        y='nWBV',
        color='CDR',
        size='eTIV',
        hover_name='ID',
        title='Volume Cerebral Normalizado por Idade e CDR',
        labels={'Age': 'Idade', 'nWBV': 'Volume Cerebral Normalizado', 'CDR': 'CDR'},
        color_discrete_sequence=['#6baed6', '#1f78b4', '#fb9a99', '#e31a1c']  
    )
        st.plotly_chart(fig2, use_container_width=True)

        nwbv_doentes_maiores_60 = data.loc[(data['Age'] > 60) & (data['CDR'] >0), ['nWBV']].reset_index(drop=True)
        nwbv_nao_doentes_maiores_60 = data.loc[(data['Age'] > 60) & (data['CDR']  == 0), ['nWBV']]

        def cohens_d(x, y):
            nx = len(x)
            ny = len(y)
            dof = nx + ny - 2
            pooled_std = np.sqrt(((nx-1)*np.std(x, ddof=1)**2 + (ny-1)*np.std(y, ddof=1)**2) / dof)
            return (np.mean(x) - np.mean(y)) / pooled_std
    
        st.header("1. Distribuição de nWBV")
    
        col1, col2 = st.columns(2)
        
        # Gráficos individuais
        with col1:
            fig, ax = plt.subplots()
            sns.histplot(nwbv_doentes_maiores_60['nWBV'], kde=True, color='red', label='CDR > 0')
            plt.title("Distribuição para Doentes (CDR > 0)")
            plt.xlabel("nWBV")
            plt.legend()
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots()
            sns.histplot(nwbv_nao_doentes_maiores_60['nWBV'], kde=True, color='green', label='CDR = 0')
            plt.title("Distribuição para Não Doentes (CDR = 0)")
            plt.xlabel("nWBV")
            plt.legend()
            st.pyplot(fig)

        # Gráfico de comparação - Boxplot
        st.subheader("Comparação entre Grupos")
        fig, ax = plt.subplots(figsize=(10, 6))

        # Preparar os dados para comparação
        nwbv_doentes_maiores_60['Grupo'] = 'CDR > 0'
        nwbv_nao_doentes_maiores_60['Grupo'] = 'CDR = 0'
        dados_comparacao = pd.concat([nwbv_doentes_maiores_60, nwbv_nao_doentes_maiores_60])

        # Criar o boxplot
        sns.boxplot(x='Grupo', y='nWBV', data=dados_comparacao,
                    hue='Grupo', 
                    palette={'CDR > 0': 'red', 'CDR = 0': 'green'}, 
                    order=['CDR > 0', 'CDR = 0'],
                    legend=False)

        plt.title("Comparação de nWBV entre Doentes e Não Doentes (CDR > 0 vs CDR = 0)")
        plt.ylabel("Valor de nWBV")
        plt.xlabel("Grupo")
        st.pyplot(fig)



        # Seção 2: Testes de Normalidade
        st.header("Testes de Normalidade (Shapiro-Wilk)")
        with st.container():
    
            st.write("")  # Espaçamento
            alpha1 = st.slider("Nível de significância (α)", 
                            min_value=0.01, 
                            max_value=0.10, 
                            value=0.05, 
                            step=0.01,
                            help="Nível de significância para os testes estatísticos",
                            key = "alpha_nwbv")
    
        
        stat_doentes, p_doentes = shapiro(nwbv_doentes_maiores_60['nWBV'])
        stat_nao_doentes, p_nao_doentes = shapiro(nwbv_nao_doentes_maiores_60['nWBV'])
        
        norm_col1, norm_col2 = st.columns(2)
        
        with norm_col1:
            st.metric(label="Doentes (CDR > 0)", 
                    value=f"p = {p_doentes:.4f}",
                    help="H₀: Os dados são normalmente distribuídos")
            st.write("Conclusão:", "Normal" if p_doentes > alpha1 else "Não normal")
        
        with norm_col2:
            st.metric(label="Não Doentes (CDR = 0)", 
                    value=f"p = {p_nao_doentes:.4f}",
                    help="H₀: Os dados são normalmente distribuídos")
            st.write("Conclusão:", "Normal" if p_nao_doentes > alpha1 else "Não normal")
        
        # Seção 3: Teste T e Tamanho do Efeito
        st.header("3. Comparação entre Grupos")
        
        t_stat, p_valor = ttest_ind(
            nwbv_doentes_maiores_60['nWBV'],
            nwbv_nao_doentes_maiores_60['nWBV'],
            alternative='less'
        )
        
        d = cohens_d(nwbv_doentes_maiores_60['nWBV'], nwbv_nao_doentes_maiores_60['nWBV'])
        
        st.subheader("Teste T para Amostras Independentes")
        st.write(f"""
        - **Hipótese nula (H₀):** Não há diferença no nWBV entre os grupos (ou doentes têm nWBV maior/igual)
        - **Hipótese alternativa (H₁):** Doentes têm nWBV menor (teste unilateral)
        """)
        
        st.metric(label="Valor-p", 
                value=f"{p_valor:.6f}",
                delta="Significativo" if p_valor < alpha1 else "Não significativo",
                delta_color="inverse")
        
        st.write(f"**Conclusão:** {'Rejeitamos H₀' if p_valor < alpha1 else 'Não rejeitamos H₀'} a um α = {alpha1}")
        
        st.subheader("Tamanho do Efeito (Cohen's d)")
        
        effect_size = st.container()
        with effect_size:
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.metric(label="Cohen's d", value=f"{d:.2f}")
            
            with col2:
                st.write("""
                | Valor | Interpretação |
                |-------|---------------|
                | 0.2   | Pequeno       |
                | 0.5   | Médio         |
                | 0.8   | Grande        |
                """)
                st.write(f"**Interpretação:** {'Grande' if abs(d) >= 0.8 else 'Médio' if abs(d) >= 0.5 else 'Pequeno'} efeito")



    with subtab4:
        
        st.header("Análise Comparativa de MMSE entre Pacientes com e sem Demência")

        col1, col2 = st.columns([2, 1])
        
        with col1:
            custom_colors = ['#6baed6', '#1f78b4', '#fb9a99', '#e31a1c']   # Azul claro, azul escuro, vermelho claro, vermelho escuro

            # Criando o gráfico
            fig5 = px.histogram(
                data,
                x='MMSE',
                color='CDR',
                nbins=20,
                title='Distribuição do MMSE por CDR',
                labels={'MMSE': 'Pontuação MMSE', 'CDR': 'CDR'},
                color_discrete_sequence=custom_colors  # Usando a paleta de cores personalizada
            )

            st.plotly_chart(fig5, use_container_width=False)

        with col2:
            with st.container(border=True): 
                st.markdown("""
                Temos que 82% dos pacientes considerados obtiveram resultado superior a 23 no MMSE, 
                onde 70% deles apresentaram um CDR = 0 e um 27% apresentaram CDR = 0.5. 
                E 100% dos pacientes com CDR = 0 obtiveram resultado superior a 23.
                """)


        
    
        with st.container():
            st.write("")  # Espaçamento
            nivel_significancia  = st.slider("Nível de significância (α)", 
                    min_value=0.01, 
                    max_value=0.10, 
                    value=0.05, 
                    step=0.01,
                    help="Nível de significância para os testes estatísticos",
                    key = "alpha_mmse")

          
        idade_minima = 60


        #  Filtrar os dados
        mmse_doentes = data.loc[(data['Age'] > idade_minima) & (data['CDR'] > 0), ['MMSE']].reset_index(drop=True)
        mmse_nao_doentes = data.loc[(data['Age'] > idade_minima) & (data['CDR'] == 0), ['MMSE']].reset_index(drop=True)

        # Layout em colunas
        col1, col2 = st.columns(2)

        # Coluna 1: Estatísticas Descritivas
        with col1:
            # Teste de normalidade
            st.write("**Teste de Normalidade (Shapiro-Wilk):**")
            stat_doentes, p_doentes = shapiro(mmse_doentes['MMSE'])
            stat_nao_doentes, p_nao_doentes = shapiro(mmse_nao_doentes['MMSE'])
            
            st.write(f"- Pacientes com CDR > 0: p-valor = {p_doentes:.4f}")
            st.write(f"- Pacientes com CDR = 0: p-valor = {p_nao_doentes:.4f}")
            
            # Interpretação
            if p_doentes < 0.05 or p_nao_doentes < 0.05:
                st.warning("Os dados não seguem uma distribuição normal (p < 0.05). Usando teste não paramétrico.")
            else:
                st.success("Os dados seguem uma distribuição normal (p ≥ 0.05). Pode-se usar teste paramétrico.")

        # Coluna 2: Testes Estatísticos
        with col2:
                    
            
            # Teste de Mann-Whitney
            st.write("\n**Teste de Mann-Whitney U (diferença entre grupos):**")
            u_stat, p_valor = mannwhitneyu(
                mmse_doentes['MMSE'],
                mmse_nao_doentes['MMSE'],
                alternative='less'
            )
            
            st.write(f"- Estatística U = {u_stat:.2f}")
            st.write(f"- p-valor = {p_valor:.4f}")
            
            # Interpretação do resultado
            st.write("\n**Interpretação:**")
            st.write("Hipótese nula (H₀): Não há diferença no MMSE entre os grupos.")
            st.write("Hipótese alternativa (H₁): Pacientes com demência (CDR>0) têm MMSE menor.")
            
            if p_valor < nivel_significancia:
                st.error(f"Rejeitamos H₀ (p < {nivel_significancia}). Há evidências de que pacientes com demência têm MMSE significativamente menor.")
            else:
                st.success(f"Não rejeitamos H₀ (p ≥ {nivel_significancia}). Não há evidências suficientes para afirmar que pacientes com demência têm MMSE menor.")

        # Gráficos
        st.subheader("Visualização dos Dados")

        mmse_doentes['Grupo'] = 'CDR > 0'
        mmse_nao_doentes['Grupo'] = 'CDR = 0'
        dados_comparacao_mmse = pd.concat([mmse_doentes, mmse_nao_doentes])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Boxplot
        
        # Criar o boxplot
        sns.boxplot(x='Grupo', y='MMSE', data=dados_comparacao_mmse,
                    hue='Grupo', 
                    palette={'CDR > 0': 'red', 'CDR = 0': 'green'}, 
                    order=['CDR > 0', 'CDR = 0'],
                    ax=ax1,
                    legend=False)
        
        ax1.set_xticks([0, 1])  # Explicitly set ticks before labels
        ax1.set_xticklabels(['CDR > 0', 'CDR = 0'])  # Fixed order to match the boxplot
        ax1.set_title('Distribuição de MMSE entre Doentes e Não Doentes')
        ax1.set_ylabel('Pontuação MMSE')
        ax1.set_xlabel('Grupo')
        

        # Histograma
        sns.histplot(mmse_nao_doentes['MMSE'], color='skyblue', label='CDR = 0', 
                    kde=True, ax=ax2, alpha=0.5)
        sns.histplot(mmse_doentes['MMSE'], color='salmon', label='CDR > 0', 
                    kde=True, ax=ax2, alpha=0.5)
        ax2.set_xlabel('Pontuação MMSE')
        ax2.set_ylabel('Frequência')
        ax2.set_title('Distribuição de MMSE')
        ax2.legend()

        st.pyplot(fig)

        # Informações adicionais
        with st.expander("Sobre esta análise"):
            st.write("""
            **Metodologia:**
            - Comparação de pontuações MMSE entre pacientes com e sem demência (CDR > 0 vs CDR = 0)
            - Teste de Shapiro-Wilk para verificar normalidade dos dados
            - Teste U de Mann-Whitney (não paramétrico) para comparar os grupos
            
            **MMSE (Mini-Mental State Examination):**
            - Avaliação cognitiva com pontuação de 0 a 30
            - Pontuações mais baixas indicam maior comprometimento cognitivo
            
            **CDR (Clinical Dementia Rating):**
            - 0: Sem demência
            - 0.5: Demência questionável
            - 1: Demência leve
            - 2: Demência moderada
            """)



























with tab4:
      st.subheader("Conclusões")

      with st.container(border=True): 
                st.markdown("""
                - Neste estudo, é um exemplo de que baixos resultados no MMSE são um sinal de alerta para possíveis casos de demência.

                - Pode-se considerar realizar o MMSE a partir dos 60 anos.

                - Exames de imagem são recomendados para fornecer uma conclusão após os resultados do MMSE.
                """)

with tab_pred:
    # Carregar o modelo salvo
    model = joblib.load('decision_tree_model.pkl')

    # Título do app
    st.title("Classificação de Demência (CDR)")

    st.subheader("Preencha os dados do paciente:")

    # Layout em duas colunas
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Sexo (M/F)", ['M', 'F'])
        educ = st.selectbox("Nível educacional", [1, 2, 3, 4, 5])
        age = st.number_input("Idade", min_value=0, max_value=120, value=75)
        etiv = st.number_input("Volume Total Intracraniano Estimado", value=1500.0)

    with col2:
        ses = st.selectbox("Status socioeconômico", [1, 2, 3, 4, 5])
        mmse = st.number_input("Mini-Exame do Estado Mental", min_value=0, max_value=30, value=28)
        nwbv = st.number_input("Volume Normalizado de Matéria Branca", value=0.75)

    # Montar DataFrame com os dados inseridos
    input_df = pd.DataFrame({
        'M/F': [gender],
        'Educ': [educ],
        'SES': [ses],
        'Age': [age],
        'MMSE': [mmse],
        'eTIV': [etiv],
        'nWBV': [nwbv]
    })

    # Botão para acionar a previsão
    if st.button("Classificar"):
        prediction = model.predict(input_df)
        proba = model.predict_proba(input_df)
        st.success(f"Resultado previsto (CDR): {prediction[0]}")

with tab_pca:

    def preprocess_data(data):
        dados_pca = data[['M/F', 'Age', 'MMSE', 'ASF', 'nWBV', 'CDR']].copy()
        dados_pca['M/F'] = dados_pca['M/F'].map({'M': 0, 'F': 1})
        return dados_pca
    
    # Aplicar PCA
    def apply_pca(dados_pca):
        dados_normalizados = (dados_pca - dados_pca.mean()) / dados_pca.std()
        pca_3d = PCA(n_components=3)
        pca_resultado = pca_3d.fit_transform(dados_normalizados)
        return pd.DataFrame(data=pca_resultado, columns=['PC1', 'PC2', 'PC3'])

    # Aplicar KMeans
    def apply_kmeans(pca_df, n_clusters=4):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(pca_df)
        pca_df['Cluster'] = labels
        return pca_df

    # Função para criar gráficos
    def plot_3d_scatter(pca_df):
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pca_df['PC1'], pca_df['PC2'], pca_df['PC3'], s=50, alpha=0.6, c=pca_df['Cluster'])
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.set_title('PCA - Visualização 3D')
        return fig

    def plot_boxplots(dados_pca, pca_df):
        colunas = ['Age', 'MMSE', 'ASF', 'nWBV', 'CDR']
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))
        axes = axes.flatten()
        dados_pca['Cluster'] = pca_df['Cluster'].values
        for i, coluna in enumerate(colunas):
            sns.boxplot(x='Cluster', y=coluna, data=dados_pca, ax=axes[i])
            axes[i].set_title(f'Distribuição de {coluna} por Cluster')
        plt.tight_layout()
        return fig

    st.title("Dashboard PCA e KMeans")

    st.subheader("Análise de Componentes Principais (PCA) e KMeans")

    data_pca = load_data()
    dados_pca = preprocess_data(data_pca)
    pca_df = apply_pca(dados_pca)
    cluster_pca_df = apply_kmeans(pca_df)

    # Exibir visualizações
    st.pyplot(plot_3d_scatter(cluster_pca_df))
    st.pyplot(plot_boxplots(dados_pca, cluster_pca_df))


with tab_simulacao:

    # Função para carregar e preprocessar os dados de simulação
    @st.cache_data
    def load_data_simulacao():
        data = pd.read_csv('arquivos/oasis_cross-sectional.csv')
        data = data.dropna(subset=['MMSE', 'CDR']).drop('Delay', axis=1)
        return data

    # Função para preparar as tabelas de distribuição de CDR por faixa etária
    def preprocess_cdr_tables(data_simulacao):
        data_simulacao_idades_cdr = data_simulacao[['Age', 'CDR']].dropna().reset_index(drop=True)

        # Filtrar apenas idades 65+
        data_65_plus = data_simulacao_idades_cdr[data_simulacao_idades_cdr['Age'] >= 65].copy()

        # Criar faixas etárias
        bins_65_plus = [65, 70, 75, 80, 85, 90, float('inf')]
        labels_65_plus = ['65-69', '70-74', '75-79', '80-84', '85-89', '90+']
        data_65_plus['faixa_etaria'] = pd.cut(data_65_plus['Age'], bins=bins_65_plus, labels=labels_65_plus, right=False)

        # Criar tabela agrupada por faixa etária e CDR
        cdr_faixa_table = data_65_plus.groupby(['faixa_etaria', 'CDR'], observed=True).size().reset_index(name='Count')
        cdr_faixa_table = cdr_faixa_table.pivot(index='faixa_etaria', columns='CDR', values='Count').fillna(0)

        # Filtrar idades >= 60 e CDR > 0
        data_cdr_pos = data_simulacao_idades_cdr[(data_simulacao_idades_cdr['Age'] >= 60) & (data_simulacao_idades_cdr['CDR'] > 0)].copy()

        # Criar faixas etárias ajustadas
        bins_cdr_pos = [60, 70, 80, 90, float('inf')]
        labels_cdr_pos = ['60-69', '70-79', '80-89', '90+']
        data_cdr_pos['faixa_etaria'] = pd.cut(data_cdr_pos['Age'], bins=bins_cdr_pos, labels=labels_cdr_pos, right=False)

        # Criar tabela de porcentagem por faixa etária e CDR
        cdr_faixa_count = data_cdr_pos.groupby(['faixa_etaria', 'CDR'], observed=True).size().reset_index(name='Count')
        total_por_faixa = cdr_faixa_count.groupby('faixa_etaria', observed=True)['Count'].transform('sum')
        cdr_faixa_count['Percent'] = (cdr_faixa_count['Count'] / total_por_faixa) * 100
        cdr_faixa_percent_table = cdr_faixa_count.pivot(index='faixa_etaria', columns='CDR', values='Percent').fillna(0)

        return cdr_faixa_table, cdr_faixa_percent_table

    # Função para calcular a projeção de Alzheimer
    def calcular_projecao_alzheimer():
        populacao_df = pd.read_csv("arquivos/populacao_idosos_2024_2040.csv")
        alzheimer_df = pd.read_csv("arquivos/alzheimer_por_faixa_etaria.csv")

        populacao_long = populacao_df.melt(id_vars="faixa_etaria", var_name="Ano", value_name="Populacao")
        populacao_long = populacao_long.merge(alzheimer_df, on="faixa_etaria")
        populacao_long["Alzheimer_Projecao"] = populacao_long["Populacao"] * (populacao_long["Alzheimer (%)"] / 100)
        populacao_long["Ano"] = populacao_long["Ano"].astype(int)

        return populacao_long

    # Função para visualizar projeção de Alzheimer
    def plot_alzheimer_projection(populacao_long):
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=populacao_long, x="Ano", y="Alzheimer_Projecao", hue="faixa_etaria", marker="o")
        plt.title("Projeção de Pessoas com Alzheimer por Faixa Etária (2024–2040)")
        plt.xlabel("Ano")
        plt.ylabel("Número Estimado de Pessoas com Alzheimer")
        plt.legend(title="Faixa Etária", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        st.pyplot(plt)

    # Função para calcular projeção por CDR
    def calcular_projecao_cdr(populacao_long, cdr_faixa_percent_table):
        alz_idade_ano = populacao_long[['Ano', 'faixa_etaria', 'Alzheimer_Projecao']].copy()

        alz_idade_ano["faixa_etaria_ajustada"] = alz_idade_ano["faixa_etaria"].replace({
            "65-69": "60-69", "70-74": "70-79", "75-79": "70-79",
            "80-84": "80-89", "85-89": "80-89", "90+": "90+"
        })

        df_final = alz_idade_ano.merge(cdr_faixa_percent_table, left_on="faixa_etaria_ajustada", right_on="faixa_etaria", how="left")

        for cdr in [0.5, 1.0, 2.0]:
            df_final[f"CDR {cdr} Projecao"] = (df_final["Alzheimer_Projecao"] * (df_final[cdr] / 100)).round(0).astype(int)

        df_agrupado = df_final.groupby("Ano")[["CDR 0.5 Projecao", "CDR 1.0 Projecao", "CDR 2.0 Projecao"]].sum()
        
        return df_agrupado

    # Função para visualizar projeção por CDR
    def plot_cdr_projection(df_agrupado):
        plt.figure(figsize=(10, 5))
        for col in ["CDR 0.5 Projecao", "CDR 1.0 Projecao", "CDR 2.0 Projecao"]:
            plt.plot(df_agrupado.index, df_agrupado[col], label=col)

        plt.xlabel("Ano")
        plt.ylabel("Quantidade de Casos")
        plt.title("Evolução do número de casos de Alzheimer por gravidade")
        plt.legend()
        plt.grid()
        st.pyplot(plt)

    # Configuração do Streamlit
    st.title("Dashboard de Análise de Alzheimer")

    data_simulacao = load_data_simulacao()
    cdr_faixa_table, cdr_faixa_percent_table = preprocess_cdr_tables(data_simulacao)
    populacao_long = calcular_projecao_alzheimer()
    df_agrupado = calcular_projecao_cdr(populacao_long, cdr_faixa_percent_table)

    # Exibir o gráfico principal ocupando toda a largura
    st.subheader("Projeção de Pessoas com Alzheimer")
    plot_alzheimer_projection(populacao_long)

    # Criar duas colunas abaixo do gráfico principal
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Evolução do Número de Casos por Gravidade")
        plot_cdr_projection(df_agrupado)

    with col2:
        st.subheader("Tabela de Projeção por CDR")
        st.write(df_agrupado)
