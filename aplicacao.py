import streamlit as st
import yfinance as yf
import pandas as pd
from prophet import Prophet
from datetime import datetime, timedelta
import plotly.graph_objects as go

# Função para prever os próximos 5 dias
def fazer_previsao(ticker='BZ=F'):
    # Calculando as datas dinâmicas
    hoje = datetime.today()
    start_date = (hoje - timedelta(days=30)).strftime('%Y-%m-%d')  # Subtrai 30 dias de hoje
    end_date = hoje.strftime('%Y-%m-%d')  # Data de hoje formatada

    # Baixando os dados históricos
    df = yf.download(ticker, start=start_date, end=end_date, interval='1d')
    
    # Verifica se os dados são suficientes
    if df.empty or len(df) < 5:
        raise ValueError("Dados insuficientes para realizar a previsão.")

    # Preparando os dados para o Prophet
    df_prophet = df[['Close']].reset_index()
    df_prophet = df_prophet.rename(columns={'Date': 'ds', 'Close': 'y'})

    # Criando o modelo Prophet
    m = Prophet(daily_seasonality=False)
    m.fit(df_prophet)

    # Criando o dataframe futuro para previsão
    future = m.make_future_dataframe(periods=5, freq='B')
    forecast = m.predict(future)

    # Filtrando as previsões dos próximos 5 dias
    prev_5_dias = forecast[['ds', 'yhat']].tail()

    # Renomeando as colunas para exibição
    prev_5_dias = prev_5_dias.rename(columns={'ds': 'DATA', 'yhat': 'PREVISÃO'})

    return df, prev_5_dias, forecast

# Função para gerar o gráfico interativo
def plot_interactive(df, prev_5_dias, ticker):
    fig = go.Figure()

    # Preço de fechamento real
    fig.add_trace(go.Scatter(
        x=df.index, 
        y=df['Close'], 
        mode='lines', 
        name='Preço de Fechamento Real'
    ))

    # Previsão
    fig.add_trace(go.Scatter(
        x=prev_5_dias['DATA'], 
        y=prev_5_dias['PREVISÃO'], 
        mode='lines+markers', 
        name='Previsão (Modelo Prophet)', 
        line=dict(dash='dot', color='green')
    ))

    fig.update_layout(
        title=f"Previsão do Preço de Fechamento para os Próximos 5 Dias - {ticker}",
        yaxis_title="Preço de Fechamento (USD)",
        legend_title="Legenda",
        template="plotly_white"
    )

    return fig

# Aplicação Streamlit
st.title("Previsão de Preços de Commodities - Petróleo Brent (BZ=F)")
st.write("Esta aplicação utiliza o modelo Prophet para prever os preços futuros do Petróleo Brent com base nos dados históricos.")

# Entrada do nome do usuário
nome = st.text_input("Digite seu nome:")

# Botão para fazer a previsão
if st.button("Fazer Previsão"):
    if nome:
        st.write(f"Olá, {nome}! Aqui está a previsão para o Petróleo Brent (BZ=F):")
        try:
            df, prev_5_dias, forecast = fazer_previsao()

            # Formatando as datas apenas para exibição
            prev_5_dias_exibicao = prev_5_dias.copy()
            prev_5_dias_exibicao['DATA'] = pd.to_datetime(prev_5_dias_exibicao['DATA']).dt.strftime('%d-%m-%Y')

            # Exibindo intervalo de dados formatado
            data_inicio = df.index.min().strftime('%d-%m-%Y')
            data_fim = df.index.max().strftime('%d-%m-%Y')
            st.write(f"Dados históricos utilizados: {data_inicio} até {data_fim}")

            # Removendo o índice da tabela
            prev_5_dias_reset = prev_5_dias_exibicao.reset_index(drop=True)

            # Exibindo as previsões
            st.subheader("Previsões para os próximos 5 dias:")
            st.write(prev_5_dias_reset)

            # Gerando o gráfico interativo
            st.subheader("Gráfico Interativo:")
            fig = plot_interactive(df, prev_5_dias, ticker='BZ=F')
            st.plotly_chart(fig)
        except Exception as e:
            st.error(f"Ocorreu um erro ao processar os dados: {e}")
    else:
        st.warning("Por favor, insira seu nome para continuar.")
