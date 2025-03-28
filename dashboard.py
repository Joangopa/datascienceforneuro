import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


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
    # Filtrar os dados onde MMSE e CDR não são nulos
    data = data.dropna(subset=['MMSE', 'CDR'])
    data.drop('Delay', axis=1, inplace=True)
    return data


@st.cache_data
def load_data2():
    data = pd.read_csv('arquivos/oasis_cross-sectional.csv')
    # Filtrar os dados onde MMSE e CDR não são nulos
    data = data.dropna(subset=['MMSE', 'CDR'])
    data.drop('Delay', axis=1, inplace=True)
    data["CDR"] = data["CDR"].astype(str)

    return data

data2 = load_data2()



data = load_data()


# Criando as abas
tab1, tab2, tab3, tab4 = st.tabs(["📌 Introdução ao Problema", "📊 Introdução aos Dados", "📈 Análises", "Conclusões"])

with tab1:
    st.header("Introdução ao Problema")


    col1, col2 = st.columns([1, 1])

    with col1:

        with st.container(border=True):  # Disponível no Streamlit >= 1.29.0
            st.write("""
                    - A doença de Alzheimer é uma doença cerebral degenerativa sem cura
                    - É caracterizada por atrofia progressiva do córtex cerebral
                    - Causa perda de memória, aumento dos déficits cognitivos e potencial perda das funções motoras
                    - É o tipo mais comum de demência e a sexta principal causa de morte nos EUA
                    - O diagnóstico é um processo intenso, lento e caro que envolve exames físicos e mentais, testes laboratoriais e neurológicos, e exames de imagem
                    - Com um diagnóstico precoce, a progressão pode ser retardada e os sintomas tratados.
            """)

        st.image("brain_atrophy.jpg", caption="Atrofia Cerebral",  width=600)
        
    with col2:  
        # Você pode adicionar imagens, gráficos ou outros elementos
        
        st.image("sinais-de-alzheimer.jpg", caption="Sinais de Alzheimer", width=700)



with tab2:
    st.header("Introdução aos Dados")
    
    # Criando duas colunas (a primeira será mais larga para a imagem)
    col1, col2 = st.columns([2, 3])  # Proporção 2:3 (ajuste conforme necessário)
    
    with col2:
        #st.subheader("Descrição das Variáveis")
        #st.write("""
        #- **CDR : Clinical Dementia Rating.** 
        #- **eTIV : Estimated Total Intracranial Volume.** A variável eTIV estima o volume cerebral intracraniano.
        #- **nWBV : Normalize Whole Brain Volume.** Representa a porcentagem da cavidade intracraniana ocupada pelo cérebro.
        #- **ASF : Atlas Scaling Factor.** A variável ASF é um fator de escala de um parâmetro que permite a comparação do volume intracraniano total estimado (eTIV) com base nas diferenças na anatomia humana.
        #- **MMSE : Mini Mental State Examination**. O Mini Exame do Estado Mental (MMSE) é uma ferramenta que pode ser usada para avaliar sistematicamente e completamente o estado mental.
        #""")
        #if st.button("MMSE"):
            
        st.image("mmse.jpg", use_container_width=False, width=800)

    with col1:
        info_variaveis_ = {
        "Variable": ["ID", "M/F", "Hand", "Age", "Educ", "SES", "eTIV", "ASF", "nWBV",  "MMSE", "CDR"],
        "Definition": [
            "Identification", "Gender", "Dominant Hand", "Age in years", 
            "Education Level", "Socioeconomic Status", 
            "Estimated Total Intracranial Volume",
            "Atlas Scaling Factor",
            "Normalize Whole Brain Volume",
            "Mini Mental State Examination",  
            "Clinical Dementia Rating"
        ],
        "Domain": [
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
                white-space: nowrap;
            }
            td {
                white-space: pre-wrap;
            }
            </style>
            """,
            unsafe_allow_html=True
            )
        st.table(info_variaveis)  

with tab3:
    
    # st.header("Análise de Correlação")

    subtab1, subtab2, subtab3, subtab4 = st.tabs(["Distribuição de CDR", "📊 Análise de Correlação", "vWBV vs CDR", "MMSE vs CDR"])
    
    

    with subtab1:


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
    
        plt.figure(figsize=(5, 3))
        ax = sns.barplot(x='Count', y='Interpretação', data=cdr_table, hue='Interpretação', palette='viridis', dodge=False)

        # Adicionando os valores de contagem no final de cada barra
        for index, row in cdr_table.iterrows():
            ax.text(row['Count'] + 1, index, str(row['Count']), color='black', va='center')

        # Remover as bordas do gráfico
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Adicionando títulos e rótulos
        plt.title('Distribuição de Contagem de Casos por Tipo de Demência')
        plt.xlabel('Contagem')
        plt.ylabel('Tipo de Demência')

        # Exibindo o gráfico no Streamlit
        st.pyplot(plt, use_container_width=False)
        

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

    with subtab4:
        
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

with tab4:
      st.subheader("Conclusões")