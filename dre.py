from dash import html, dcc
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import openpyxl
from dash.exceptions import PreventUpdate
from dash_bootstrap_templates import load_figure_template

load_figure_template('lux')

# import from folders/theme changer
from app import *
# ========= Autenticação ======= #

# ========== Styles ============ #

tab_card = {'height': '100%'}

main_config = {
    "hovermode": "x unified",
    "legend": {"yanchor":"top",
                "y":0.9,
                "xanchor":"left",
                "x":0.1,
                "title": {"text": None},
                "font" :{"color":"white"},
                "bgcolor": "rgba(0,0,0,0.5)"},
    "margin": {"l":10, "r":10, "t":10, "b":10}
}

config_graph = {"displayModeBar": False, "showTips": False}

#=========== Jupyter ==============

df = pd.read_csv('datasets/dados.csv',index_col=0)

df['Data'] = pd.to_datetime(df['Data'], format='%Y-%m-%d')
df['Mes'] = df['Mes'].astype(str)

df_orig = df.copy()

meses = {'Jan': 1, 'Fev': 2, 'Mar': 3, 'Abr': 4, 'Maio': 5, 'Jun': 6, 'Jul': 7, 'Ago': 8, 'Set': 9, 'Out': 10,
         'Nov': 11, 'Dez': 12}

df['Mes'] = df['Mes'].map(meses)

# Criando opções pros filtros que virão
options_month = [{'label': 'Acumulado Ano', 'value': 0}]
for i, j in zip(df_orig['Mes'].unique(), df['Mes'].unique()):
    options_month.append({'label': i, 'value': j})
options_month = sorted(options_month, key=lambda x: x['value'])

# ========= Função dos Filtros ========= #

def month_filter(month):
    if month == 0:
        mask = df['Mes'].isin(df['Mes'].unique())
    else:
        mask = df['Mes'].isin([month])
    return mask

def formatar_porc(valor):
    return '{:.2%}'.format(valor)

def formatar_valor(valor):
    return '{:,.2f}'.format(valor)

# =========  Layout  =========== #
app.layout = dbc.Container(children=[

    # Layout

    # Linha 1
    dbc.Row([
        dbc.Col([
            dbc.Col([
                html.Img(src=r'assets/logo.jpg',className='perfil_avatar',
                    style={'background-color': 'transparent', 'border-color': 'transparent'})
                    ]),
            ],sm=2, md=2,lg=1),
        dbc.Col([
            dbc.Card([
               dbc.CardBody([
                    dbc.Col([
                        html.H5('Relatório - Demonstração de Resultados'),
                    ], style={'margin-top':'15px'})
                ])
            ], style=tab_card)
        ],sm=10, md=10,lg=11),
    ], className='g-1 my-auto', style={'margin-top': '7px'}),

    # Linha 2
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dbc.Row(
                        dbc.Col([
                            html.H6('Selecione o Mês'),
                            dbc.RadioItems(
                                id="radio-month",
                                options=options_month,
                                value=0,
                                inline=True,
                                labelCheckedClassName="text-success",
                                inputCheckedClassName="border border-success bg-success",
                            ),
                        ])
                    )
                ])
            ], style=tab_card)
        ],md=4, sm=4, lg=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col(
                            html.H6('Resultado')
                        ),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id='graph1', className='dbc', config=config_graph),
                            dcc.Graph(id='graph3', className='dbc', config=config_graph)
                        ], sm=6, md=6),
                        dbc.Col([
                            dcc.Graph(id='graph2', className='dbc', config=config_graph),
                            dcc.Graph(id='graph4', className='dbc', config=config_graph)
                        ], sm=6, md=6)
                    ])
                ])
            ], style=tab_card)
        ], sm=8, md=8, lg=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H6('Receita x Despesas')
                        ],sm=5, lg=7,md=6),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id='graph6',className='dbc',config=config_graph)
                        ],sm=12, md=6),
                        dbc.Col([
                            dcc.Graph(id='graph15',className='dbc',config=config_graph)
                        ],sm=12, md=6),
                    ])
               ])
            ], style=tab_card)
        ], sm=12, md=6, lg=4),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H6('CMV x Estoque')
                        ],sm=4,md=6,lg=4),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id='graph7',className='dbc',config=config_graph)
                        ], sm=12, md=6),
                        dbc.Col([
                            dcc.Graph(id='graph8',className='dbc',config=config_graph)
                        ], sm=12,md=6)
                    ])
               ])
            ], style=tab_card)
        ], sm=12, lg=4, md=6),
    ], className='g-1 my-auto', style={'margin-top': '7px'}),

    # Linha 3
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H6('Análise Horizontal')
                        ],sm=4,lg=3),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id='graph14',className='dbc',config=config_graph)
                        ], sm=12,md=12),
                    ]),
                ])
            ], style=tab_card)
        ], sm=12,lg=6),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H6('Análise de Faturamento por Produto')
                            ],lg=6,sm=8),
                        ]),
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id='graph13',className='dbc',config=config_graph),
                        ], sm=12,md=12),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dbc.Button('Relatório Vendas',color='primary', id='btn_xlsx', n_clicks=0,style={'margin-top': '10px'}),
                            dcc.Download(id='download-xlsx'),
                        ])
                    ]),
                ])
            ], style=tab_card)
        ], sm=12,lg=6),
    ], className='g-1 my-auto', style={'margin-top': '7px'}),

    # Linha 4
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H6('Desp. Administrativas.')
                        ],sm=6,md=6,lg=5),
                        ]),
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id='graph9',className='dbc',config=config_graph),
                        ], sm=12,md=6,lg=6),
                        dbc.Col([
                            dcc.Graph(id='graph10',className='dbc',config=config_graph),
                        ], sm=12, md=6,lg=6),
                    ])
                ])
            ], style=tab_card)
        ],sm=6),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H6('Impostos sobre Faturamento')
                        ],sm=8,md=8,lg=4),
                        ]),
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id='graph11',className='dbc',config=config_graph),
                        ], sm=12,md=5,lg=5),
                        dbc.Col([
                            dcc.Graph(id='graph12',className='dbc',config=config_graph),
                        ], sm=12, md=7,lg=7),
                    ])
                ])
            ], style=tab_card)
        ],sm=6),
    ], className='g-1 my-auto', style={'margin-top': '7px'})
], fluid=True, style={'height': '100vh'})


#========= CallBack =========

# graph 1
@app.callback(
    Output('graph1','figure'),
    Input('radio-month','value'),
)

def graph1(month):

    mask = month_filter(month)
    df1 = df.loc[mask]

    df1 = df1.loc[(df1['Tipo_Categoria'] == 'Lucro Bruto')].groupby('Tipo_Categoria')[
        'Vendas', 'Lucro_Bruto'].sum().reset_index()



    fig1 = go.Figure()
    fig1.add_trace(go.Indicator(mode='number',
                                title={
                                    "text": f"<span style='font-size:60%'>Margem Bruta</span>"},
                                value=df1['Lucro_Bruto'].iloc[0] / df1['Vendas'].iloc[0] * 100,
                                number={'suffix': "%"},
                                ))

    fig1.update_layout(margin=dict(l=0, r=0, t=20, b=20), height=90, template='lux')

    return fig1

# Graph 2
@app.callback(
        Output('graph2', 'figure'),
        Input('radio-month', 'value'),
    )

def graph2(month):

    mask = month_filter(month)
    df2 = df.loc[mask]

    df2 = df2.loc[(df2['Tipo_Categoria'] == 'Lucro Operacional')].groupby('Tipo_Categoria')[
        'Vendas', 'Lucro_Operacional'].sum().reset_index()

    fig2 = go.Figure()
    fig2.add_trace(go.Indicator(mode='number',
                                title={
                                    "text": f"<span style='font-size:60%'>Margem EBTIDA</span>"},
                                value=df2['Lucro_Operacional'].iloc[0] / df2['Vendas'].iloc[0] * 100,
                                number={'suffix': "%"},
                                ))

    fig2.update_layout(margin=dict(l=0, r=0, t=20, b=20), height=90, template='lux')

    return fig2

# Graph 3
@app.callback(
        Output('graph3', 'figure'),
        Input('radio-month', 'value'),
    )
def graph3(month):

    mask = month_filter(month)
    df3 = df.loc[mask]

    df3 = df3.loc[(df3['Tipo_Categoria'] == 'Lucro Líquido')].groupby('Tipo_Categoria')[
        'Vendas', 'Lucro_Líquido'].sum().reset_index()

    fig3 = go.Figure()
    fig3.add_trace(go.Indicator(mode='number',
                                title={
                                    "text": f"<span style='font-size:60%'>Margem Líquida</span>"},
                                value=df3['Lucro_Líquido'].iloc[0] / df3['Vendas'].iloc[0] * 100,
                                number={'suffix': "%"},
                                ))

    fig3.update_layout(margin=dict(l=0, r=0, t=20, b=20), height=90, template='lux')

    return fig3

# Graph 4
@app.callback(
        Output('graph4', 'figure'),
        Input('radio-month', 'value'),
    )
def graph4(month):

    mask = month_filter(month)
    df4 = df.loc[mask]

    df4 = df4.loc[(df4['Tipo_Categoria'] == 'Lucro Líquido')].groupby('Tipo_Categoria')[
        'Vendas', 'Lucro_Líquido'].sum().reset_index()

    fig4 = go.Figure()
    fig4.add_trace(go.Indicator(mode='number',
                                title={
                                    "text": f"<span style='font-size:60%'>Lucro Líquido</span>"},
                                value=df4['Lucro_Líquido'].iloc[0],
                                                                ))

    fig4.update_layout(margin=dict(l=0, r=0, t=20, b=20), height=90, template='lux')

    return fig4


# graph 6
@app.callback(
    Output('graph6','figure'),
    Input('radio-month','value'),
)

def graph6(month):

    mask = month_filter(month)
    df6 = df.loc[mask]

    df6 = df6.loc[(df6['Sub_Categoria'] == 'Receita')|(df6['Sub_Categoria'] == 'Imp Vendas')|(df6['Sub_Categoria'] == 'Desp Adm')|(df6['Sub_Categoria'] == 'Estoque')|(df6['Sub_Categoria'] == 'Impostos')]
    df6 = df6.groupby('Sub_Categoria')['Valor_Total'].sum().reset_index()
    fig6 = go.Figure()
    fig6.add_trace(go.Pie(labels=df6['Sub_Categoria'], values=df6['Valor_Total'], pull=[0, 0, 0, 0, 0.2]))
    fig6.update(layout_showlegend=False)
    fig6.update_layout(margin=dict(l=0, r=0, t=20, b=20), height=170, template='lux')

    return fig6

# graph 7
@app.callback(
    Output('graph7','figure'),
    Input('radio-month','value'),
)

def graph7(month):

    mask = month_filter(month)
    df7 = df.loc[mask]

    df7 = df7.loc[(df7['Sub_Categoria'] == 'CMV') | (df7['Sub_Categoria'] == 'Estoque')].groupby('Sub_Categoria')['Valor_Total'].sum().reset_index()

    fig7 = go.Figure()
    fig7.add_trace(go.Pie(labels=df7['Sub_Categoria'], values=df7['Valor_Total'], hole=.5))
    fig7.update(layout_showlegend=False)
    fig7.update_layout(main_config, height=170, template='lux')

    return fig7

# graph 8
@app.callback(
    Output('graph8','figure'),
    Input('radio-month','value'),
)

def graph8(month):

    mask = month_filter(month)
    df7 = df.loc[mask]

    df7 = df7.loc[(df7['Sub_Categoria'] == 'CMV') | (df7['Sub_Categoria'] == 'Estoque')].groupby('Sub_Categoria')[
        'Valor_Total'].sum().reset_index().round(2)

    fig8 = go.Figure(go.Bar(
        x=df7['Sub_Categoria'],
        y=df7['Valor_Total'],
        orientation='v',
        textposition='auto',
        text=df7['Valor_Total'],
        insidetextfont=dict(family='Times', size=12)))
    fig8.update_layout(main_config, height=170, template='lux')

    return fig8

# graph 9
@app.callback(
    Output('graph9','figure'),
    Input('radio-month','value'),
)

def graph9(month):

    mask = month_filter(month)
    df8 = df.loc[mask]

    df8 = df8.loc[(df8['Sub_Categoria'] == 'Desp Adm')]
    df8 = df8.groupby('Tipo_Categoria')['Valor_Total'].sum().reset_index().sort_values('Valor_Total', axis=0,
                                                                                       ascending=False)
    fig9 = go.Figure()
    fig9.add_trace(go.Pie(labels=df8['Tipo_Categoria'], values=df8['Valor_Total'], hole=.5))
    fig9.update(layout_showlegend=False)
    fig9.update_layout(main_config, height=170, template='lux')

    return fig9

# graph 10
@app.callback(
    Output('graph10','figure'),
    Input('radio-month','value'),
)

def graph10(month):

    mask = month_filter(month)
    df8 = df.loc[mask]

    df8 = df8.loc[(df8['Sub_Categoria'] == 'Desp Adm')]
    df8 = df8.groupby('Tipo_Categoria')['Valor_Total'].sum().reset_index().sort_values('Valor_Total', axis=0,
                                                                                       ascending=False)

    fig10 = go.Figure(go.Bar(
        x=df8['Tipo_Categoria'],
        y=df8['Valor_Total'],
        orientation='v',
        textposition='auto',
        text=df8['Valor_Total'],
        insidetextfont=dict(family='Times', size=12)))
    fig10.update_layout(main_config, height=170, template='lux')

    return fig10

# graph 11
@app.callback(
    Output('graph11','figure'),
    Input('radio-month','value'),
)

def graph11(month):

    mask = month_filter(month)
    df9 = df.loc[mask]

    df9 = df9.loc[(df9['Sub_Categoria'] == 'Imp Vendas') | (df9['Sub_Categoria'] == 'Receita')]
    df9 = df9.groupby('Tipo_Categoria')['Valor_Total'].sum().reset_index()

    fig11 = go.Figure()
    fig11.add_trace(go.Pie(labels=df9['Tipo_Categoria'], values=df9['Valor_Total'], hole=.5))
    fig11.update(layout_showlegend=False)
    fig11.update_layout(main_config, height=170, template='lux')

    return fig11

# graph 12
@app.callback(
    Output('graph12','figure'),
    Input('radio-month','value'),
)

def graph12(month):

    mask = month_filter(month)
    df9 = df.loc[mask]

    df9 = df9.loc[(df9['Sub_Categoria'] == 'Imp Vendas') | (df9['Sub_Categoria'] == 'Receita')]
    df9 = df9.groupby('Tipo_Categoria')['Valor_Total'].sum().reset_index()
    df9['Perc'] = ''
    df9.iloc[[1], [2]] = 100

    df9.loc[0, 'Perc'] = ((df9.loc[0, 'Valor_Total'] / df9.loc[1, 'Valor_Total']) * 100).round(2)
    df9.loc[2, 'Perc'] = ((df9.loc[2, 'Valor_Total'] / df9.loc[1, 'Valor_Total']) * 100).round(2)
    df9.loc[3, 'Perc'] = ((df9.loc[3, 'Valor_Total'] / df9.loc[1, 'Valor_Total']) * 100).round(2)
    df9.loc[4, 'Perc'] = ((df9.loc[4, 'Valor_Total'] / df9.loc[1, 'Valor_Total']) * 100).round(2)

    df9['Perc'] = df9['Perc'].astype(float)
    df9 = df9.loc[(df9['Tipo_Categoria'] != 'Faturamento')]
    df9 = df9.sort_values('Perc', axis=0, ascending=False)

    fig12 = go.Figure(go.Bar(
        x=df9['Tipo_Categoria'],
        y=df9['Valor_Total'],
        orientation='v',
        textposition='auto',
        text=df9['Perc'],
        insidetextfont=dict(family='Times', size=12)))
    fig12.update_layout(main_config, height=170, template='lux')

    return fig12

# graph 13
@app.callback(
    Output('graph13','figure'),
    Input('radio-month','value'),
)

def graph13(month):

    mask = month_filter(month)
    df10 = df.loc[mask]

    df10 = df10.loc[(df10['Tipo_Categoria'] == 'Faturamento')].groupby(['ID_Produto', 'Faturamento'])['Valor_Total'].sum().reset_index()

    fig13 = px.scatter(df10, x="ID_Produto", y="Faturamento")
    fig13.update_layout(margin=dict(l=0, r=0, t=20, b=20), height=240, template='lux')

    return fig13


# Download dados produtos
@app.callback(
    Output('download-xlsx', 'data'),
    Input('btn_xlsx', 'n_clicks'),
    prevent_initial_call=True,
)

def func(n_clicks):

    if n_clicks == None:
        raise PreventUpdate

    df10 = df.loc[(df['Tipo_Categoria'] == 'Faturamento')]

    df10 = df10.groupby(['Mes','ID_Produto'])['Valor_Total'].sum().reset_index()

    return dcc.send_data_frame(df10.to_excel, "Vendas.xlsx", sheet_name="Sheet_name_1")

# graph 14
@app.callback(
    Output('graph14','figure'),
    Input('radio-month','value'),
)

def graph14(month):

    mask = month_filter(month)
    df11 = df.loc[mask]

    df11 = df11.loc[(df11['Sub_Categoria'] == 'Receita') | (df11['Sub_Categoria'] == 'Lucro Bruto') |
                    (df11['Sub_Categoria'] == 'Lucro Operacional') | (df11['Sub_Categoria'] == 'Lucro Líquido')]
    df11 = df11.groupby(['Mes', 'Sub_Categoria'])['Valor_Total'].sum().reset_index()
    df11.rename(columns={'Sub_Categoria': 'Indicador'}, inplace=True)


    fig14 = px.line(df11, x='Mes', y='Valor_Total', color='Indicador', markers=True)
    fig14.update(layout_showlegend=False)
    fig14.update_layout(main_config, height=240, template='lux')

    return fig14

# Graph 15
@app.callback(
    Output('graph15','figure'),
    Input('radio-month','value'),
)

def graph15(month):

    mask = month_filter(month)
    df6 = df.loc[mask]

    df6 = df6.loc[(df6['Sub_Categoria'] == 'Receita') | (df6['Sub_Categoria'] == 'Imp Vendas') | (
                df6['Sub_Categoria'] == 'Desp Adm') | (df6['Sub_Categoria'] == 'Estoque') | (
                              df6['Sub_Categoria'] == 'Impostos')]
    df6 = df6.groupby('Sub_Categoria')['Valor_Total'].sum().reset_index()

    fig15 = go.Figure(go.Bar(
        x=df6['Sub_Categoria'],
        y=df6['Valor_Total'],
        orientation='v',
        textposition='auto',
        text=df6['Valor_Total'],
        insidetextfont=dict(family='Times', size=12)))


    fig15.update_layout(main_config, height=170, template='lux')

    return fig15



# Run server
if __name__ == '__main__':
    app.run_server(debug=True)