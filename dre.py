from dash import html, dcc
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import openpyxl
from dash.exceptions import PreventUpdate
from dash_bootstrap_templates import load_figure_template

load_figure_template('morph')

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
                html.Img(src=r'assets/logo.png',style={'height': '100%','width': '100%',
                                                               'margin-top':'7px'})
                    ]),
            ],sm=2, md=2,lg=1),
        dbc.Col([
            dbc.Card([
               dbc.CardBody([
                    dbc.Col([
                        html.H6('Dashboard - Demonstração de Resultados'),
                    ])
                ], style=tab_card)
            ])
        ],sm=6, md=6,lg=7),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                        dbc.Row([
                           dbc.Col([
                                html.I(className='fa-brands fa-whatsapp',style={'font-size':'150%'})
                            ],sm=2,md=2,lg=2),
                            dbc.Col([
                                html.H6('(16) 9.9791-7818'),
                            ],sm=10,md=10,lg=10),
                    ])
            ], style=tab_card)
        ]),
        ],sm=4, md=4, lg=4)
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
                    dbc.Row([
                        dbc.Col([
                            dbc.Button('Relatório Análise Horizontal e Vertical',color='primary', id='btn_ahv', n_clicks=0,style={'margin-top': '10px'}),
                            dcc.Download(id='download-ahv'),
                        ])
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

    fig1.update_layout(margin=dict(l=0, r=0, t=20, b=20), height=90, template='morph')

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

    fig2.update_layout(margin=dict(l=0, r=0, t=20, b=20), height=90, template='morph')

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

    fig3.update_layout(margin=dict(l=0, r=0, t=20, b=20), height=90, template='morph')

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

    fig4.update_layout(margin=dict(l=0, r=0, t=20, b=20), height=90, template='morph')

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
    fig6.update_layout(margin=dict(l=0, r=0, t=20, b=20), height=170, template='morph')

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
    fig7.update_layout(main_config, height=170, template='morph')

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
    fig8.update_layout(main_config, height=170, template='morph')

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
    fig9.update_layout(main_config, height=170, template='morph')

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
    fig10.update_layout(main_config, height=170, template='morph')

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
    fig11.update_layout(main_config, height=170, template='morph')

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
    fig12.update_layout(main_config, height=170, template='morph')

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
    fig13.update_layout(margin=dict(l=0, r=0, t=20, b=20), height=240, template='morph')

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

# Download dados Analise Horizontal
@app.callback(
    Output('download-ahv', 'data'),
    Input('btn_ahv', 'n_clicks'),
    prevent_initial_call=True,
)

def func(n_clicks):

    if n_clicks == None:
        raise PreventUpdate

    df12 = pd.read_csv('datasets/dados.csv',index_col=0)
    meses = {'Jan': 1, 'Fev': 2, 'Mar': 3, 'Abr': 4, 'Maio': 5, 'Jun': 6, 'Jul': 7, 'Ago': 8, 'Set': 9, 'Out': 10,
             'Nov': 11, 'Dez': 12}

    df12['Mes'] = df12['Mes'].map(meses)

    df12 = df12.groupby('Mes')[
        'Faturamento', 'Total_PIS', 'Total_COFINS', 'Total_ICMS', 'Total_ISS', 'Total_CMV', 'Lucro_Bruto',
        'Água', 'Energia', 'Internet', 'Aluguel', 'Telefone', 'Salários', 'Encargos',
        'Lucro_Operacional', 'Total_CSLL', 'Total_IRPJ', 'Total_IRPJ_Ad', 'Lucro_Líquido'].sum().reset_index()


    df12['Despesas Operacionais'] = df12['Água'] + df12['Energia'] + df12['Internet'] + df12['Aluguel'] + df12['Telefone'] + \
                                   df12['Salários'] + df12['Encargos']

    df12['Receita/Desp não Operacionais'] = 0

    df12['Impostos'] = df12['Total_PIS'] + df12['Total_COFINS'] + df12['Total_ICMS'] + df12['Total_ISS']

    df12['Receita Líquida'] = df12['Faturamento'] - df12['Impostos']

    df12['IRPJ'] = df12['Total_IRPJ'] + df12['Total_IRPJ_Ad']

    df12['EBTIDA'] = df12['Lucro_Operacional'] - df12['Receita/Desp não Operacionais']

    df12 = df12[['Mes', 'Faturamento', 'Impostos', 'Total_PIS', 'Total_COFINS', 'Total_ICMS',
                 'Total_ISS', 'Receita Líquida', 'Total_CMV', 'Lucro_Bruto', 'Água',
                 'Energia', 'Internet', 'Aluguel', 'Telefone', 'Salários', 'Encargos',
                 'Despesas Operacionais', 'Lucro_Operacional', 'Receita/Desp não Operacionais', 'EBTIDA', 'Total_CSLL',
                 'Total_IRPJ', 'Total_IRPJ_Ad', 'IRPJ', 'Lucro_Líquido']]

    df12.drop(columns=['Total_PIS', 'Total_COFINS', 'Total_ICMS', 'Total_ISS', 'Água', 'Energia', 'Internet', 'Aluguel',
                       'Telefone', 'Salários', 'Encargos','Total_IRPJ', 'Total_IRPJ_Ad'], inplace=True)

    df12.rename(columns={'Total_CMV':'CMV','Total_CSLL':'CSLL','Faturamento':'Receita Bruta'},inplace=True)
    df12.columns = df12.columns.str.replace('_', ' ')
    df12.rename(columns={'Mes': '0'}, inplace=True)
    df12 = df12.set_index('0')
    df12 = df12.T
    df12 = df12.reset_index()
    df12.rename(columns={'index': 'Indicador'}, inplace=True)
    df12.rename(columns={1: 'Jan', 2: 'Fev', 3: 'Mar', 4: 'Abr', 5: 'Maio', 6: 'Jun', 7: 'Jul', 8: 'Ago', 9: 'Set',
                        10: 'Out', 11: 'Nov', 12: 'Dez'}, inplace=True)
    df12['AV1'] = 1
    df12['AV2'] = 1
    df12['AV3'] = 1
    df12['AV4'] = 1
    df12['AV5'] = 1
    df12['AV6'] = 1
    df12['AV7'] = 1
    df12['AV8'] = 1
    df12['AV9'] = 1
    df12['AV10'] = 1
    df12['AV11'] = 1
    df12['AV12'] = 1

    df12.loc[1, 'AV1'] = df12.loc[1, 'Jan'] / df12.loc[0, 'Jan']
    df12.loc[2, 'AV1'] = df12.loc[2, 'Jan'] / df12.loc[0, 'Jan']
    df12.loc[3, 'AV1'] = df12.loc[3, 'Jan'] / df12.loc[0, 'Jan']
    df12.loc[4, 'AV1'] = df12.loc[4, 'Jan'] / df12.loc[0, 'Jan']
    df12.loc[5, 'AV1'] = df12.loc[5, 'Jan'] / df12.loc[0, 'Jan']
    df12.loc[6, 'AV1'] = df12.loc[6, 'Jan'] / df12.loc[0, 'Jan']
    df12.loc[7, 'AV1'] = df12.loc[7, 'Jan'] / df12.loc[0, 'Jan']
    df12.loc[8, 'AV1'] = df12.loc[8, 'Jan'] / df12.loc[0, 'Jan']
    df12.loc[9, 'AV1'] = df12.loc[9, 'Jan'] / df12.loc[0, 'Jan']
    df12.loc[10, 'AV1'] = df12.loc[10, 'Jan'] / df12.loc[0, 'Jan']
    df12.loc[11, 'AV1'] = df12.loc[11, 'Jan'] / df12.loc[0, 'Jan']

    df12.loc[1, 'AV2'] = df12.loc[1, 'Fev'] / df12.loc[0, 'Fev']
    df12.loc[2, 'AV2'] = df12.loc[2, 'Fev'] / df12.loc[0, 'Fev']
    df12.loc[3, 'AV2'] = df12.loc[3, 'Fev'] / df12.loc[0, 'Fev']
    df12.loc[4, 'AV2'] = df12.loc[4, 'Fev'] / df12.loc[0, 'Fev']
    df12.loc[5, 'AV2'] = df12.loc[5, 'Fev'] / df12.loc[0, 'Fev']
    df12.loc[6, 'AV2'] = df12.loc[6, 'Fev'] / df12.loc[0, 'Fev']
    df12.loc[7, 'AV2'] = df12.loc[7, 'Fev'] / df12.loc[0, 'Fev']
    df12.loc[8, 'AV2'] = df12.loc[8, 'Fev'] / df12.loc[0, 'Fev']
    df12.loc[9, 'AV2'] = df12.loc[9, 'Fev'] / df12.loc[0, 'Fev']
    df12.loc[10, 'AV2'] = df12.loc[10, 'Fev'] / df12.loc[0, 'Fev']
    df12.loc[11, 'AV2'] = df12.loc[11, 'Fev'] / df12.loc[0, 'Fev']

    df12.loc[1, 'AV3'] = df12.loc[1, 'Mar'] / df12.loc[0, 'Mar']
    df12.loc[2, 'AV3'] = df12.loc[2, 'Mar'] / df12.loc[0, 'Mar']
    df12.loc[3, 'AV3'] = df12.loc[3, 'Mar'] / df12.loc[0, 'Mar']
    df12.loc[4, 'AV3'] = df12.loc[4, 'Mar'] / df12.loc[0, 'Mar']
    df12.loc[5, 'AV3'] = df12.loc[5, 'Mar'] / df12.loc[0, 'Mar']
    df12.loc[6, 'AV3'] = df12.loc[6, 'Mar'] / df12.loc[0, 'Mar']
    df12.loc[7, 'AV3'] = df12.loc[7, 'Mar'] / df12.loc[0, 'Mar']
    df12.loc[8, 'AV3'] = df12.loc[8, 'Mar'] / df12.loc[0, 'Mar']
    df12.loc[9, 'AV3'] = df12.loc[9, 'Mar'] / df12.loc[0, 'Mar']
    df12.loc[10, 'AV3'] = df12.loc[10, 'Mar'] / df12.loc[0, 'Mar']
    df12.loc[11, 'AV3'] = df12.loc[11, 'Mar'] / df12.loc[0, 'Mar']

    df12.loc[1, 'AV4'] = df12.loc[1, 'Abr'] / df12.loc[0, 'Abr']
    df12.loc[2, 'AV4'] = df12.loc[2, 'Abr'] / df12.loc[0, 'Abr']
    df12.loc[3, 'AV4'] = df12.loc[3, 'Abr'] / df12.loc[0, 'Abr']
    df12.loc[4, 'AV4'] = df12.loc[4, 'Abr'] / df12.loc[0, 'Abr']
    df12.loc[5, 'AV4'] = df12.loc[5, 'Abr'] / df12.loc[0, 'Abr']
    df12.loc[6, 'AV4'] = df12.loc[6, 'Abr'] / df12.loc[0, 'Abr']
    df12.loc[7, 'AV4'] = df12.loc[7, 'Abr'] / df12.loc[0, 'Abr']
    df12.loc[8, 'AV4'] = df12.loc[8, 'Abr'] / df12.loc[0, 'Abr']
    df12.loc[9, 'AV4'] = df12.loc[9, 'Abr'] / df12.loc[0, 'Abr']
    df12.loc[10, 'AV4'] = df12.loc[10, 'Abr'] / df12.loc[0, 'Abr']
    df12.loc[11, 'AV4'] = df12.loc[11, 'Abr'] / df12.loc[0, 'Abr']

    df12.loc[1, 'AV5'] = df12.loc[1, 'Maio'] / df12.loc[0, 'Maio']
    df12.loc[2, 'AV5'] = df12.loc[2, 'Maio'] / df12.loc[0, 'Maio']
    df12.loc[3, 'AV5'] = df12.loc[3, 'Maio'] / df12.loc[0, 'Maio']
    df12.loc[4, 'AV5'] = df12.loc[4, 'Maio'] / df12.loc[0, 'Maio']
    df12.loc[5, 'AV5'] = df12.loc[5, 'Maio'] / df12.loc[0, 'Maio']
    df12.loc[6, 'AV5'] = df12.loc[6, 'Maio'] / df12.loc[0, 'Maio']
    df12.loc[7, 'AV5'] = df12.loc[7, 'Maio'] / df12.loc[0, 'Maio']
    df12.loc[8, 'AV5'] = df12.loc[8, 'Maio'] / df12.loc[0, 'Maio']
    df12.loc[9, 'AV5'] = df12.loc[9, 'Maio'] / df12.loc[0, 'Maio']
    df12.loc[10, 'AV5'] = df12.loc[10, 'Maio'] / df12.loc[0, 'Maio']
    df12.loc[11, 'AV5'] = df12.loc[11, 'Maio'] / df12.loc[0, 'Maio']

    df12.loc[1, 'AV6'] = df12.loc[1, 'Jun'] / df12.loc[0, 'Jun']
    df12.loc[2, 'AV6'] = df12.loc[2, 'Jun'] / df12.loc[0, 'Jun']
    df12.loc[3, 'AV6'] = df12.loc[3, 'Jun'] / df12.loc[0, 'Jun']
    df12.loc[4, 'AV6'] = df12.loc[4, 'Jun'] / df12.loc[0, 'Jun']
    df12.loc[5, 'AV6'] = df12.loc[5, 'Jun'] / df12.loc[0, 'Jun']
    df12.loc[6, 'AV6'] = df12.loc[6, 'Jun'] / df12.loc[0, 'Jun']
    df12.loc[7, 'AV6'] = df12.loc[7, 'Jun'] / df12.loc[0, 'Jun']
    df12.loc[8, 'AV6'] = df12.loc[8, 'Jun'] / df12.loc[0, 'Jun']
    df12.loc[9, 'AV6'] = df12.loc[9, 'Jun'] / df12.loc[0, 'Jun']
    df12.loc[10, 'AV6'] = df12.loc[10, 'Jun'] / df12.loc[0, 'Jun']
    df12.loc[11, 'AV6'] = df12.loc[11, 'Jun'] / df12.loc[0, 'Jun']

    df12.loc[1, 'AV7'] = df12.loc[1, 'Jul'] / df12.loc[0, 'Jul']
    df12.loc[2, 'AV7'] = df12.loc[2, 'Jul'] / df12.loc[0, 'Jul']
    df12.loc[3, 'AV7'] = df12.loc[3, 'Jul'] / df12.loc[0, 'Jul']
    df12.loc[4, 'AV7'] = df12.loc[4, 'Jul'] / df12.loc[0, 'Jul']
    df12.loc[5, 'AV7'] = df12.loc[5, 'Jul'] / df12.loc[0, 'Jul']
    df12.loc[6, 'AV7'] = df12.loc[6, 'Jul'] / df12.loc[0, 'Jul']
    df12.loc[7, 'AV7'] = df12.loc[7, 'Jul'] / df12.loc[0, 'Jul']
    df12.loc[8, 'AV7'] = df12.loc[8, 'Jul'] / df12.loc[0, 'Jul']
    df12.loc[9, 'AV7'] = df12.loc[9, 'Jul'] / df12.loc[0, 'Jul']
    df12.loc[10, 'AV7'] = df12.loc[10, 'Jul'] / df12.loc[0, 'Jul']
    df12.loc[11, 'AV7'] = df12.loc[11, 'Jul'] / df12.loc[0, 'Jul']

    df12.loc[1, 'AV8'] = df12.loc[1, 'Ago'] / df12.loc[0, 'Ago']
    df12.loc[2, 'AV8'] = df12.loc[2, 'Ago'] / df12.loc[0, 'Ago']
    df12.loc[3, 'AV8'] = df12.loc[3, 'Ago'] / df12.loc[0, 'Ago']
    df12.loc[4, 'AV8'] = df12.loc[4, 'Ago'] / df12.loc[0, 'Ago']
    df12.loc[5, 'AV8'] = df12.loc[5, 'Ago'] / df12.loc[0, 'Ago']
    df12.loc[6, 'AV8'] = df12.loc[6, 'Ago'] / df12.loc[0, 'Ago']
    df12.loc[7, 'AV8'] = df12.loc[7, 'Ago'] / df12.loc[0, 'Ago']
    df12.loc[8, 'AV8'] = df12.loc[8, 'Ago'] / df12.loc[0, 'Ago']
    df12.loc[9, 'AV8'] = df12.loc[9, 'Ago'] / df12.loc[0, 'Ago']
    df12.loc[10, 'AV8'] = df12.loc[10, 'Ago'] / df12.loc[0, 'Ago']
    df12.loc[11, 'AV8'] = df12.loc[11, 'Ago'] / df12.loc[0, 'Ago']

    df12.loc[1, 'AV9'] = df12.loc[1, 'Set'] / df12.loc[0, 'Set']
    df12.loc[2, 'AV9'] = df12.loc[2, 'Set'] / df12.loc[0, 'Set']
    df12.loc[3, 'AV9'] = df12.loc[3, 'Set'] / df12.loc[0, 'Set']
    df12.loc[4, 'AV9'] = df12.loc[4, 'Set'] / df12.loc[0, 'Set']
    df12.loc[5, 'AV9'] = df12.loc[5, 'Set'] / df12.loc[0, 'Set']
    df12.loc[6, 'AV9'] = df12.loc[6, 'Set'] / df12.loc[0, 'Set']
    df12.loc[7, 'AV9'] = df12.loc[7, 'Set'] / df12.loc[0, 'Set']
    df12.loc[8, 'AV9'] = df12.loc[8, 'Set'] / df12.loc[0, 'Set']
    df12.loc[9, 'AV9'] = df12.loc[9, 'Set'] / df12.loc[0, 'Set']
    df12.loc[10, 'AV9'] = df12.loc[10, 'Set'] / df12.loc[0, 'Set']
    df12.loc[11, 'AV9'] = df12.loc[11, 'Set'] / df12.loc[0, 'Set']

    df12.loc[1, 'AV10'] = df12.loc[1, 'Out'] / df12.loc[0, 'Out']
    df12.loc[2, 'AV10'] = df12.loc[2, 'Out'] / df12.loc[0, 'Out']
    df12.loc[3, 'AV10'] = df12.loc[3, 'Out'] / df12.loc[0, 'Out']
    df12.loc[4, 'AV10'] = df12.loc[4, 'Out'] / df12.loc[0, 'Out']
    df12.loc[5, 'AV10'] = df12.loc[5, 'Out'] / df12.loc[0, 'Out']
    df12.loc[6, 'AV10'] = df12.loc[6, 'Out'] / df12.loc[0, 'Out']
    df12.loc[7, 'AV10'] = df12.loc[7, 'Out'] / df12.loc[0, 'Out']
    df12.loc[8, 'AV10'] = df12.loc[8, 'Out'] / df12.loc[0, 'Out']
    df12.loc[9, 'AV10'] = df12.loc[9, 'Out'] / df12.loc[0, 'Out']
    df12.loc[10, 'AV10'] = df12.loc[10, 'Out'] / df12.loc[0, 'Out']
    df12.loc[11, 'AV10'] = df12.loc[11, 'Out'] / df12.loc[0, 'Out']

    df12.loc[1, 'AV11'] = df12.loc[1, 'Nov'] / df12.loc[0, 'Nov']
    df12.loc[2, 'AV11'] = df12.loc[2, 'Nov'] / df12.loc[0, 'Nov']
    df12.loc[3, 'AV11'] = df12.loc[3, 'Nov'] / df12.loc[0, 'Nov']
    df12.loc[4, 'AV11'] = df12.loc[4, 'Nov'] / df12.loc[0, 'Nov']
    df12.loc[5, 'AV11'] = df12.loc[5, 'Nov'] / df12.loc[0, 'Nov']
    df12.loc[6, 'AV11'] = df12.loc[6, 'Nov'] / df12.loc[0, 'Nov']
    df12.loc[7, 'AV11'] = df12.loc[7, 'Nov'] / df12.loc[0, 'Nov']
    df12.loc[8, 'AV11'] = df12.loc[8, 'Nov'] / df12.loc[0, 'Nov']
    df12.loc[9, 'AV11'] = df12.loc[9, 'Nov'] / df12.loc[0, 'Nov']
    df12.loc[10, 'AV11'] = df12.loc[10, 'Nov'] / df12.loc[0, 'Nov']
    df12.loc[11, 'AV11'] = df12.loc[11, 'Nov'] / df12.loc[0, 'Nov']

    df12.loc[1, 'AV12'] = df12.loc[1, 'Dez'] / df12.loc[0, 'Dez']
    df12.loc[2, 'AV12'] = df12.loc[2, 'Dez'] / df12.loc[0, 'Dez']
    df12.loc[3, 'AV12'] = df12.loc[3, 'Dez'] / df12.loc[0, 'Dez']
    df12.loc[4, 'AV12'] = df12.loc[4, 'Dez'] / df12.loc[0, 'Dez']
    df12.loc[5, 'AV12'] = df12.loc[5, 'Dez'] / df12.loc[0, 'Dez']
    df12.loc[6, 'AV12'] = df12.loc[6, 'Dez'] / df12.loc[0, 'Dez']
    df12.loc[7, 'AV12'] = df12.loc[7, 'Dez'] / df12.loc[0, 'Dez']
    df12.loc[8, 'AV12'] = df12.loc[8, 'Dez'] / df12.loc[0, 'Dez']
    df12.loc[9, 'AV12'] = df12.loc[9, 'Dez'] / df12.loc[0, 'Dez']
    df12.loc[10, 'AV12'] = df12.loc[10, 'Dez'] / df12.loc[0, 'Dez']
    df12.loc[11, 'AV12'] = df12.loc[11, 'Dez'] / df12.loc[0, 'Dez']

    df12 = df12[
        ['Indicador', 'Jan', 'AV1', 'Fev', 'AV2', 'Mar', 'AV3', 'Abr', 'AV4', 'Maio', 'AV5', 'Jun', 'AV6', 'Jul', 'AV7',
         'Ago', 'AV8', 'Set', 'AV9', 'Out', 'AV10', 'Nov', 'AV11', 'Dez', 'AV12']]
    df12['AH2'] = ''
    df12['AH3'] = ''
    df12['AH4'] = ''
    df12['AH5'] = ''
    df12['AH6'] = ''
    df12['AH7'] = ''
    df12['AH8'] = ''
    df12['AH9'] = ''
    df12['AH10'] = ''
    df12['AH11'] = ''
    df12['AH12'] = ''

    df12.loc[0, 'AH2'] = df12.loc[0, 'Fev'] / df12.loc[0, 'Jan'] - 1
    df12.loc[1, 'AH2'] = df12.loc[1, 'Fev'] / df12.loc[1, 'Jan'] - 1
    df12.loc[2, 'AH2'] = df12.loc[2, 'Fev'] / df12.loc[2, 'Jan'] - 1
    df12.loc[3, 'AH2'] = df12.loc[3, 'Fev'] / df12.loc[3, 'Jan'] - 1
    df12.loc[4, 'AH2'] = df12.loc[4, 'Fev'] / df12.loc[4, 'Jan'] - 1
    df12.loc[5, 'AH2'] = df12.loc[5, 'Fev'] / df12.loc[5, 'Jan'] - 1
    df12.loc[6, 'AH2'] = df12.loc[6, 'Fev'] / df12.loc[6, 'Jan'] - 1
    df12.loc[7, 'AH2'] = df12.loc[7, 'Fev'] / df12.loc[7, 'Jan'] - 1
    df12.loc[8, 'AH2'] = df12.loc[8, 'Fev'] / df12.loc[8, 'Jan'] - 1
    df12.loc[9, 'AH2'] = df12.loc[9, 'Fev'] / df12.loc[9, 'Jan'] - 1
    df12.loc[10, 'AH2'] = df12.loc[10, 'Fev'] / df12.loc[10, 'Jan'] - 1
    df12.loc[11, 'AH2'] = df12.loc[11, 'Fev'] / df12.loc[11, 'Jan'] - 1

    df12.loc[0, 'AH3'] = df12.loc[0, 'Mar'] / df12.loc[0, 'Fev'] - 1
    df12.loc[1, 'AH3'] = df12.loc[1, 'Mar'] / df12.loc[1, 'Fev'] - 1
    df12.loc[2, 'AH3'] = df12.loc[2, 'Mar'] / df12.loc[2, 'Fev'] - 1
    df12.loc[3, 'AH3'] = df12.loc[3, 'Mar'] / df12.loc[3, 'Fev'] - 1
    df12.loc[4, 'AH3'] = df12.loc[4, 'Mar'] / df12.loc[4, 'Fev'] - 1
    df12.loc[5, 'AH3'] = df12.loc[5, 'Mar'] / df12.loc[5, 'Fev'] - 1
    df12.loc[6, 'AH3'] = df12.loc[6, 'Mar'] / df12.loc[6, 'Fev'] - 1
    df12.loc[7, 'AH3'] = df12.loc[7, 'Mar'] / df12.loc[7, 'Fev'] - 1
    df12.loc[8, 'AH3'] = df12.loc[8, 'Mar'] / df12.loc[8, 'Fev'] - 1
    df12.loc[9, 'AH3'] = df12.loc[9, 'Mar'] / df12.loc[9, 'Fev'] - 1
    df12.loc[10, 'AH3'] = df12.loc[10, 'Mar'] / df12.loc[10, 'Fev'] - 1
    df12.loc[11, 'AH3'] = df12.loc[11, 'Mar'] / df12.loc[11, 'Fev'] - 1

    df12.loc[0, 'AH4'] = df12.loc[0, 'Abr'] / df12.loc[0, 'Mar'] - 1
    df12.loc[1, 'AH4'] = df12.loc[1, 'Abr'] / df12.loc[1, 'Mar'] - 1
    df12.loc[2, 'AH4'] = df12.loc[2, 'Abr'] / df12.loc[2, 'Mar'] - 1
    df12.loc[3, 'AH4'] = df12.loc[3, 'Abr'] / df12.loc[3, 'Mar'] - 1
    df12.loc[4, 'AH4'] = df12.loc[4, 'Abr'] / df12.loc[4, 'Mar'] - 1
    df12.loc[5, 'AH4'] = df12.loc[5, 'Abr'] / df12.loc[5, 'Mar'] - 1
    df12.loc[6, 'AH4'] = df12.loc[6, 'Abr'] / df12.loc[6, 'Mar'] - 1
    df12.loc[7, 'AH4'] = df12.loc[7, 'Abr'] / df12.loc[7, 'Mar'] - 1
    df12.loc[8, 'AH4'] = df12.loc[8, 'Abr'] / df12.loc[8, 'Mar'] - 1
    df12.loc[9, 'AH4'] = df12.loc[9, 'Abr'] / df12.loc[9, 'Mar'] - 1
    df12.loc[10, 'AH4'] = df12.loc[10, 'Abr'] / df12.loc[10, 'Mar'] - 1
    df12.loc[11, 'AH4'] = df12.loc[11, 'Abr'] / df12.loc[11, 'Mar'] - 1

    df12.loc[0, 'AH5'] = df12.loc[0, 'Maio'] / df12.loc[0, 'Abr'] - 1
    df12.loc[1, 'AH5'] = df12.loc[1, 'Maio'] / df12.loc[1, 'Abr'] - 1
    df12.loc[2, 'AH5'] = df12.loc[2, 'Maio'] / df12.loc[2, 'Abr'] - 1
    df12.loc[3, 'AH5'] = df12.loc[3, 'Maio'] / df12.loc[3, 'Abr'] - 1
    df12.loc[4, 'AH5'] = df12.loc[4, 'Maio'] / df12.loc[4, 'Abr'] - 1
    df12.loc[5, 'AH5'] = df12.loc[5, 'Maio'] / df12.loc[5, 'Abr'] - 1
    df12.loc[6, 'AH5'] = df12.loc[6, 'Maio'] / df12.loc[6, 'Abr'] - 1
    df12.loc[7, 'AH5'] = df12.loc[7, 'Maio'] / df12.loc[7, 'Abr'] - 1
    df12.loc[8, 'AH5'] = df12.loc[8, 'Maio'] / df12.loc[8, 'Abr'] - 1
    df12.loc[9, 'AH5'] = df12.loc[9, 'Maio'] / df12.loc[9, 'Abr'] - 1
    df12.loc[10, 'AH5'] = df12.loc[10, 'Maio'] / df12.loc[10, 'Abr'] - 1
    df12.loc[11, 'AH5'] = df12.loc[11, 'Maio'] / df12.loc[11, 'Abr'] - 1

    df12.loc[0, 'AH6'] = df12.loc[0, 'Jun'] / df12.loc[0, 'Maio'] - 1
    df12.loc[1, 'AH6'] = df12.loc[1, 'Jun'] / df12.loc[1, 'Maio'] - 1
    df12.loc[2, 'AH6'] = df12.loc[2, 'Jun'] / df12.loc[2, 'Maio'] - 1
    df12.loc[3, 'AH6'] = df12.loc[3, 'Jun'] / df12.loc[3, 'Maio'] - 1
    df12.loc[4, 'AH6'] = df12.loc[4, 'Jun'] / df12.loc[4, 'Maio'] - 1
    df12.loc[5, 'AH6'] = df12.loc[5, 'Jun'] / df12.loc[5, 'Maio'] - 1
    df12.loc[6, 'AH6'] = df12.loc[6, 'Jun'] / df12.loc[6, 'Maio'] - 1
    df12.loc[7, 'AH6'] = df12.loc[7, 'Jun'] / df12.loc[7, 'Maio'] - 1
    df12.loc[8, 'AH6'] = df12.loc[8, 'Jun'] / df12.loc[8, 'Maio'] - 1
    df12.loc[9, 'AH6'] = df12.loc[9, 'Jun'] / df12.loc[9, 'Maio'] - 1
    df12.loc[10, 'AH6'] = df12.loc[10, 'Jun'] / df12.loc[10, 'Maio'] - 1
    df12.loc[11, 'AH6'] = df12.loc[11, 'Jun'] / df12.loc[11, 'Maio'] - 1

    df12.loc[0, 'AH7'] = df12.loc[0, 'Jul'] / df12.loc[0, 'Jun'] - 1
    df12.loc[1, 'AH7'] = df12.loc[1, 'Jul'] / df12.loc[1, 'Jun'] - 1
    df12.loc[2, 'AH7'] = df12.loc[2, 'Jul'] / df12.loc[2, 'Jun'] - 1
    df12.loc[3, 'AH7'] = df12.loc[3, 'Jul'] / df12.loc[3, 'Jun'] - 1
    df12.loc[4, 'AH7'] = df12.loc[4, 'Jul'] / df12.loc[4, 'Jun'] - 1
    df12.loc[5, 'AH7'] = df12.loc[5, 'Jul'] / df12.loc[5, 'Jun'] - 1
    df12.loc[6, 'AH7'] = df12.loc[6, 'Jul'] / df12.loc[6, 'Jun'] - 1
    df12.loc[7, 'AH7'] = df12.loc[7, 'Jul'] / df12.loc[7, 'Jun'] - 1
    df12.loc[8, 'AH7'] = df12.loc[8, 'Jul'] / df12.loc[8, 'Jun'] - 1
    df12.loc[9, 'AH7'] = df12.loc[9, 'Jul'] / df12.loc[9, 'Jun'] - 1
    df12.loc[10, 'AH7'] = df12.loc[10, 'Jul'] / df12.loc[10, 'Jun'] - 1
    df12.loc[11, 'AH7'] = df12.loc[11, 'Jul'] / df12.loc[11, 'Jun'] - 1

    df12.loc[0, 'AH8'] = df12.loc[0, 'Ago'] / df12.loc[0, 'Jul'] - 1
    df12.loc[1, 'AH8'] = df12.loc[1, 'Ago'] / df12.loc[1, 'Jul'] - 1
    df12.loc[2, 'AH8'] = df12.loc[2, 'Ago'] / df12.loc[2, 'Jul'] - 1
    df12.loc[3, 'AH8'] = df12.loc[3, 'Ago'] / df12.loc[3, 'Jul'] - 1
    df12.loc[4, 'AH8'] = df12.loc[4, 'Ago'] / df12.loc[4, 'Jul'] - 1
    df12.loc[5, 'AH8'] = df12.loc[5, 'Ago'] / df12.loc[5, 'Jul'] - 1
    df12.loc[6, 'AH8'] = df12.loc[6, 'Ago'] / df12.loc[6, 'Jul'] - 1
    df12.loc[7, 'AH8'] = df12.loc[7, 'Ago'] / df12.loc[7, 'Jul'] - 1
    df12.loc[8, 'AH8'] = df12.loc[8, 'Ago'] / df12.loc[8, 'Jul'] - 1
    df12.loc[9, 'AH8'] = df12.loc[9, 'Ago'] / df12.loc[9, 'Jul'] - 1
    df12.loc[10, 'AH8'] = df12.loc[10, 'Ago'] / df12.loc[10, 'Jul'] - 1
    df12.loc[11, 'AH8'] = df12.loc[11, 'Ago'] / df12.loc[11, 'Jul'] - 1

    df12.loc[0, 'AH9'] = df12.loc[0, 'Set'] / df12.loc[0, 'Ago'] - 1
    df12.loc[1, 'AH9'] = df12.loc[1, 'Set'] / df12.loc[1, 'Ago'] - 1
    df12.loc[2, 'AH9'] = df12.loc[2, 'Set'] / df12.loc[2, 'Ago'] - 1
    df12.loc[3, 'AH9'] = df12.loc[3, 'Set'] / df12.loc[3, 'Ago'] - 1
    df12.loc[4, 'AH9'] = df12.loc[4, 'Set'] / df12.loc[4, 'Ago'] - 1
    df12.loc[5, 'AH9'] = df12.loc[5, 'Set'] / df12.loc[5, 'Ago'] - 1
    df12.loc[6, 'AH9'] = df12.loc[6, 'Set'] / df12.loc[6, 'Ago'] - 1
    df12.loc[7, 'AH9'] = df12.loc[7, 'Set'] / df12.loc[7, 'Ago'] - 1
    df12.loc[8, 'AH9'] = df12.loc[8, 'Set'] / df12.loc[8, 'Ago'] - 1
    df12.loc[9, 'AH9'] = df12.loc[9, 'Set'] / df12.loc[9, 'Ago'] - 1
    df12.loc[10, 'AH9'] = df12.loc[10, 'Set'] / df12.loc[10, 'Ago'] - 1
    df12.loc[11, 'AH9'] = df12.loc[11, 'Set'] / df12.loc[11, 'Ago'] - 1

    df12.loc[0, 'AH10'] = df12.loc[0, 'Out'] / df12.loc[0, 'Set'] - 1
    df12.loc[1, 'AH10'] = df12.loc[1, 'Out'] / df12.loc[1, 'Set'] - 1
    df12.loc[2, 'AH10'] = df12.loc[2, 'Out'] / df12.loc[2, 'Set'] - 1
    df12.loc[3, 'AH10'] = df12.loc[3, 'Out'] / df12.loc[3, 'Set'] - 1
    df12.loc[4, 'AH10'] = df12.loc[4, 'Out'] / df12.loc[4, 'Set'] - 1
    df12.loc[5, 'AH10'] = df12.loc[5, 'Out'] / df12.loc[5, 'Set'] - 1
    df12.loc[6, 'AH10'] = df12.loc[6, 'Out'] / df12.loc[6, 'Set'] - 1
    df12.loc[7, 'AH10'] = df12.loc[7, 'Out'] / df12.loc[7, 'Set'] - 1
    df12.loc[8, 'AH10'] = df12.loc[8, 'Out'] / df12.loc[8, 'Set'] - 1
    df12.loc[9, 'AH10'] = df12.loc[9, 'Out'] / df12.loc[9, 'Set'] - 1
    df12.loc[10, 'AH10'] = df12.loc[10, 'Out'] / df12.loc[10, 'Set'] - 1
    df12.loc[11, 'AH10'] = df12.loc[11, 'Out'] / df12.loc[11, 'Set'] - 1

    df12.loc[0, 'AH11'] = df12.loc[0, 'Nov'] / df12.loc[0, 'Out'] - 1
    df12.loc[1, 'AH11'] = df12.loc[1, 'Nov'] / df12.loc[1, 'Out'] - 1
    df12.loc[2, 'AH11'] = df12.loc[2, 'Nov'] / df12.loc[2, 'Out'] - 1
    df12.loc[3, 'AH11'] = df12.loc[3, 'Nov'] / df12.loc[3, 'Out'] - 1
    df12.loc[4, 'AH11'] = df12.loc[4, 'Nov'] / df12.loc[4, 'Out'] - 1
    df12.loc[5, 'AH11'] = df12.loc[5, 'Nov'] / df12.loc[5, 'Out'] - 1
    df12.loc[6, 'AH11'] = df12.loc[6, 'Nov'] / df12.loc[6, 'Out'] - 1
    df12.loc[7, 'AH11'] = df12.loc[7, 'Nov'] / df12.loc[7, 'Out'] - 1
    df12.loc[8, 'AH11'] = df12.loc[8, 'Nov'] / df12.loc[8, 'Out'] - 1
    df12.loc[9, 'AH11'] = df12.loc[9, 'Nov'] / df12.loc[9, 'Out'] - 1
    df12.loc[10, 'AH11'] = df12.loc[10, 'Nov'] / df12.loc[10, 'Out'] - 1
    df12.loc[11, 'AH11'] = df12.loc[11, 'Nov'] / df12.loc[11, 'Out'] - 1

    df12.loc[0, 'AH12'] = df12.loc[0, 'Dez'] / df12.loc[0, 'Nov'] - 1
    df12.loc[1, 'AH12'] = df12.loc[1, 'Dez'] / df12.loc[1, 'Nov'] - 1
    df12.loc[2, 'AH12'] = df12.loc[2, 'Dez'] / df12.loc[2, 'Nov'] - 1
    df12.loc[3, 'AH12'] = df12.loc[3, 'Dez'] / df12.loc[3, 'Nov'] - 1
    df12.loc[4, 'AH12'] = df12.loc[4, 'Dez'] / df12.loc[4, 'Nov'] - 1
    df12.loc[5, 'AH12'] = df12.loc[5, 'Dez'] / df12.loc[5, 'Nov'] - 1
    df12.loc[6, 'AH12'] = df12.loc[6, 'Dez'] / df12.loc[6, 'Nov'] - 1
    df12.loc[7, 'AH12'] = df12.loc[7, 'Dez'] / df12.loc[7, 'Nov'] - 1
    df12.loc[8, 'AH12'] = df12.loc[8, 'Dez'] / df12.loc[8, 'Nov'] - 1
    df12.loc[9, 'AH12'] = df12.loc[9, 'Dez'] / df12.loc[9, 'Nov'] - 1
    df12.loc[10, 'AH12'] = df12.loc[10, 'Dez'] / df12.loc[10, 'Nov'] - 1
    df12.loc[11, 'AH12'] = df12.loc[11, 'Dez'] / df12.loc[11, 'Nov'] - 1

    df12 = df12[
        ['Indicador', 'Jan', 'AV1', 'Fev', 'AV2', 'AH2', 'Mar', 'AV3', 'AH3', 'Abr', 'AV4', 'AH4', 'Maio', 'AV5', 'AH5',
         'Jun', 'AV6', 'AH6', 'Jul', 'AV7', 'AH7', 'Ago', 'AV8', 'AH8', 'Set', 'AV9', 'AH9', 'Out', 'AV10', 'AH10',
         'Nov', 'AV11', 'AH11', 'Dez', 'AV12', 'AH12']]

    df12 = df12.fillna(0)

    df12['Jan'] = df12['Jan'].apply(formatar_valor)
    df12['Fev'] = df12['Fev'].apply(formatar_valor)
    df12['Mar'] = df12['Mar'].apply(formatar_valor)
    df12['Abr'] = df12['Abr'].apply(formatar_valor)
    df12['Maio'] = df12['Maio'].apply(formatar_valor)
    df12['Jun'] = df12['Jun'].apply(formatar_valor)
    df12['Jul'] = df12['Jul'].apply(formatar_valor)
    df12['Ago'] = df12['Ago'].apply(formatar_valor)
    df12['Set'] = df12['Set'].apply(formatar_valor)
    df12['Out'] = df12['Out'].apply(formatar_valor)
    df12['Nov'] = df12['Nov'].apply(formatar_valor)
    df12['Dez'] = df12['Dez'].apply(formatar_valor)

    df12['AV1'] = df12['AV1'].apply(formatar_porc)
    df12['AV2'] = df12['AV2'].apply(formatar_porc)
    df12['AV3'] = df12['AV3'].apply(formatar_porc)
    df12['AV4'] = df12['AV4'].apply(formatar_porc)
    df12['AV5'] = df12['AV5'].apply(formatar_porc)
    df12['AV6'] = df12['AV6'].apply(formatar_porc)
    df12['AV7'] = df12['AV7'].apply(formatar_porc)
    df12['AV8'] = df12['AV8'].apply(formatar_porc)
    df12['AV9'] = df12['AV9'].apply(formatar_porc)
    df12['AV10'] = df12['AV10'].apply(formatar_porc)
    df12['AV11'] = df12['AV11'].apply(formatar_porc)
    df12['AV12'] = df12['AV12'].apply(formatar_porc)

    df12['AH2'] = df12['AH2'].apply(formatar_porc)
    df12['AH3'] = df12['AH3'].apply(formatar_porc)
    df12['AH4'] = df12['AH4'].apply(formatar_porc)
    df12['AH5'] = df12['AH5'].apply(formatar_porc)
    df12['AH6'] = df12['AH6'].apply(formatar_porc)
    df12['AH7'] = df12['AH7'].apply(formatar_porc)
    df12['AH8'] = df12['AH8'].apply(formatar_porc)
    df12['AH9'] = df12['AH9'].apply(formatar_porc)
    df12['AH10'] = df12['AH10'].apply(formatar_porc)
    df12['AH11'] = df12['AH11'].apply(formatar_porc)
    df12['AH12'] = df12['AH12'].apply(formatar_porc)

    return dcc.send_data_frame(df12.to_excel, "Análise.xlsx", sheet_name="Sheet_name_1")


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
    fig14.update_layout(main_config, height=240, template='morph')

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


    fig15.update_layout(main_config, height=170, template='morph')

    return fig15



# Run server
if __name__ == '__main__':
    app.run_server(debug=False)