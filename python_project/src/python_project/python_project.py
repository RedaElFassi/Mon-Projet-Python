from pybacktestchain.data_module import FirstTwoMoments
from pybacktestchain.broker import Backtest, StopLoss
from pybacktestchain.blockchain import load_blockchain
from datetime import datetime
import pandas as pd
from io import StringIO

# Set verbosity for logging
verbose = False  # Set to True to enable logging, or False to suppress it

backtest = Backtest(
    initial_date=datetime(2019, 1, 1),
    final_date=datetime(2020, 1, 1),
    information_class=FirstTwoMoments,
    risk_model=StopLoss,
    name_blockchain='backtest',
    verbose=verbose
)

backtest.run_backtest()

block_chain = load_blockchain('backtest')
print(str(block_chain))
# check if the blockchain is valid
print(block_chain.is_valid())




for block in block_chain.chain:
    if block.name_backtest == "Genesis Block":  # Ignorer le bloc Genesis
        continue
    
    print(f"Processing block: {block.name_backtest}")
    
    try:
        # Convertir les données en DataFrame
        df = pd.read_csv(StringIO(block.data), delim_whitespace=True)
        print("Data as DataFrame:")
        print(df.head())
    except Exception as e:
        print(f"Error converting block data to DataFrame: {e}")
    print("-" * 80)

print(df)




import plotly.express as px
from dash import Dash, dcc, html, Input, Output

import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px

# Charger les données dans un DataFrame (vous avez déjà df, donc on l'utilise directement)
# df = ... (votre dataframe ici)

import numpy as np
import pandas as pd

def compute_pnl(df):
    """
    Compute the Profit and Loss (PnL) evolution over time for each ticker and the overall portfolio.
    Args:
        df (pd.DataFrame): DataFrame containing transaction data.
    Returns:
        pd.DataFrame: DataFrame with PnL evolution over time for the overall portfolio and stocks.
    """
    # Calculate the average buy price for each Ticker
    df['Average_Buy_Price'] = df.groupby('Ticker', group_keys=False)['Price'].transform(
        lambda x: x.expanding().mean()
    )
    
    # Compute PnL for each transaction
    df['Transaction_PnL'] = np.where(
        df['Action'] == 'SELL',
        (df['Price'] - df['Average_Buy_Price']) * df['Quantity'],
        0  # PnL for BUY transactions is 0
    )
    
    # Compute cumulative PnL over time
    df['Cumulative_PnL'] = df.groupby('Date')['Transaction_PnL'].cumsum()
    
    # Aggregate overall and per-stock PnL
    overall_pnl = df.groupby('Date')['Cumulative_PnL'].sum().reset_index(name='Overall_PnL')
    stock_pnl = df.groupby(['Date', 'Ticker'])['Cumulative_PnL'].sum().reset_index()

    return overall_pnl, stock_pnl




def compute_returns(df):
    """
    Computes portfolio returns and stock returns.
    Adds `Daily_Return`, `Portfolio_Value`, and `Average_Buy_Price` columns to the DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame with transaction data.

    Returns:
        tuple: portfolio_returns and stock_returns as DataFrames.
    """
    # Calculate the average buy price per ticker
    df['Average_Buy_Price'] = df.groupby('Ticker', group_keys=False)['Price'].transform(
        lambda x: x.expanding().mean()
    )
    
    # Calculate portfolio value as cash plus holdings
    df['Stock_Value'] = df['Quantity'] * df['Price']
    df['Portfolio_Value'] = df['Cash'] + df.groupby('Date', group_keys=False)['Stock_Value'].transform('sum')

    # Calculate daily returns
    df['Daily_Return'] = df['Portfolio_Value'].pct_change().fillna(0)

    # Extract portfolio and stock returns
    portfolio_returns = df[['Date', 'Daily_Return']].drop_duplicates()
    stock_returns = df[['Date', 'Ticker', 'Daily_Return']].drop_duplicates()

    return portfolio_returns, stock_returns






# Sharpe Ratio Calculation
def sharpe_ratio(returns, risk_free_rate=0.01):
    """
    Calculate Sharpe Ratio based on returns.
    """
    excess_returns = returns - risk_free_rate / 252  # Adjust risk-free rate for daily returns
    mean_excess_return = excess_returns.mean()
    std_excess_return = excess_returns.std()
    
    if std_excess_return == 0:
        return 0  # Avoid division by zero
    return mean_excess_return / std_excess_return


# Calculer les métriques
overall_pnl, stock_pnl = compute_pnl(df)
portfolio_returns, stock_returns = compute_returns(df)
portfolio_sharpe = sharpe_ratio(portfolio_returns['Daily_Return'])

# Initialiser l'application Dash avec Bootstrap
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Disposition du tableau de bord
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            dcc.Dropdown(
                id='stock-selector',
                options=[{'label': ticker, 'value': ticker} for ticker in df['Ticker'].unique()],
                value=df['Ticker'].unique(),
                multi=True,
                placeholder="Select stocks to display"
            )
        ], width=12)
    ], className="mb-3"),

    dbc.Row([
        dbc.Col([
            dcc.Graph(id='overall-pnl-chart', config={'displayModeBar': False})
        ], width=6),
        dbc.Col([
            dcc.Graph(id='stock-pnl-chart', config={'displayModeBar': False})
        ], width=6),
    ], className="mb-4"),

    dbc.Row([
        dbc.Col([
            dcc.Graph(id='returns-chart', config={'displayModeBar': False})
        ], width=12),
    ])
], fluid=True)

# Callback pour le graphique du PnL global
@app.callback(
    Output('overall-pnl-chart', 'figure'),
    Input('stock-selector', 'value')
)
def update_pnl_chart(selected_stocks):
    # Filter data based on selected stocks
    filtered_df = df[df['Ticker'].isin(selected_stocks)]
    overall_pnl, _ = compute_pnl(filtered_df)
    
    # Create line chart for overall PnL evolution
    fig = px.line(
        overall_pnl,
        x='Date',
        y='Overall_PnL',
        title="Overall Portfolio PnL Over Time",
        labels={'Overall_PnL': 'PnL', 'Date': 'Date'}
    )
    return fig


# Callback pour le graphique du PnL par action
@app.callback(
    Output('stock-pnl-chart', 'figure'),
    Input('stock-selector', 'value')
)
def update_stock_pnl_chart(selected_stocks):
    # Filter data based on selected stocks
    filtered_df = df[df['Ticker'].isin(selected_stocks)]
    _, stock_pnl = compute_pnl(filtered_df)
    
    # Create line chart for stock-specific PnL evolution
    fig = px.line(
        stock_pnl,
        x='Date',
        y='Cumulative_PnL',
        color='Ticker',
        title="Stock PnL Over Time",
        labels={'Cumulative_PnL': 'PnL', 'Date': 'Date', 'Ticker': 'Stock'}
    )
    return fig


# Callback pour le graphique des retours
@app.callback(
    Output('returns-chart', 'figure'),
    Input('stock-selector', 'value')
)
def update_returns_chart(selected_stocks):
    filtered_df = df[df['Ticker'].isin(selected_stocks)]
    _, stock_returns = compute_returns(filtered_df)
    fig = px.line(
        stock_returns,
        x='Date',
        y='Daily_Return',
        color='Ticker',
        title="Daily Returns by Stock",
        labels={'Daily_Return': 'Daily Return', 'Date': 'Date'}
    )
    return fig

import webbrowser
from threading import Timer

# Function to open the browser
def open_browser():
    webbrowser.open_new("http://127.0.0.1:8050/")

# Run the Dash app
if __name__ == "__main__":
    Timer(1, open_browser).start()  # Open the browser after a 1-second delay
    app.run_server(debug=False, port=8050)





