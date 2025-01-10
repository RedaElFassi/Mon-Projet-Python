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

# Calculate PnL
def compute_pnl(df):
    """
    Compute the PnL for each stock and the overall portfolio.
    """
    df['PnL'] = np.where(
        df['Action'] == 'SELL',
        (df['Price'] - df['Average_Buy_Price']) * df['Quantity'],
        0
    )
    
    # Aggregate PnL for each stock
    stock_pnl = df.groupby('Ticker')['PnL'].sum().reset_index(name='Stock_PnL')
    
    # Total PnL for the portfolio
    overall_pnl = df['PnL'].sum()
    
    return overall_pnl, stock_pnl


# Compute Returns
def compute_returns(df):
    """
    Compute daily returns for the portfolio and for each stock.
    """
    # Calculate cumulative portfolio value (Cash + Value of Positions)
    df['Portfolio_Value'] = df['Cash'] + df.groupby('Date')['Price'].apply(
        lambda x: x.sum()
    )
    
    # Compute portfolio returns
    df['Daily_Return'] = df['Portfolio_Value'].pct_change().fillna(0)
    
    # Compute stock-specific returns
    stock_returns = df.groupby('Ticker').apply(
        lambda group: group['Price'].pct_change().fillna(0)
    ).reset_index(level=0, drop=True)
    
    return df['Daily_Return'], stock_returns


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
    filtered_df = df[df['Ticker'].isin(selected_stocks)]
    overall_pnl, _ = compute_pnl(filtered_df)
    fig = px.bar(
        x=['Overall PnL'],
        y=[overall_pnl],
        labels={'x': 'Metric', 'y': 'PnL'},
        title="Overall Portfolio PnL"
    )
    return fig

# Callback pour le graphique du PnL par action
@app.callback(
    Output('stock-pnl-chart', 'figure'),
    Input('stock-selector', 'value')
)
def update_stock_pnl_chart(selected_stocks):
    filtered_df = df[df['Ticker'].isin(selected_stocks)]
    _, stock_pnl = compute_pnl(filtered_df)
    fig = px.bar(
        stock_pnl,
        x='Ticker',
        y='PnL',
        title="PnL by Stock",
        labels={'PnL': 'PnL', 'Ticker': 'Stock'}
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
    app.run_server(debug=True, port=8050)




