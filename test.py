import dash_bootstrap_components as dbc
import numpy as np
import plotly.express as px
import plotly.io as pio
from dash import Dash, Input, Output, Patch, State, callback, dcc, html
from dash_bootstrap_templates import load_figure_template

# adds templates to plotly.io
load_figure_template(["DARKLY", "MINTY_DARK"])


df = px.data.gapminder()

app = Dash(__name__, external_stylesheets=[dbc.themes.SLATE])

fig = px.scatter(
    df.query("year==2007"),
    x="gdpPercap",
    y="lifeExp",
    size="pop",
    color="continent",
    log_x=True,
    size_max=60,
    template="MINTY_DARK",
    # template="DARKLY",
)

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 50,
    "bottom": 0,
    "width": "24rem",
    "padding": "2rem 1rem",
    # "background-color": "#f8f9fa",
}

PLOT_STYLE = {
    # "position": "fixed",
    # "top": 0,
    # "left": 75,
    # "bottom": 0,
    # "width": "75rem",
    "padding": "2rem 1rem",
    # "margin-left": "15px",
    # "margin-top": "7px",
    # "margin-right": "15px",
}


elements = {"M": "Mirror", "L": "Lens", "A": "Aperture"}


def parse_element(element: str) -> html.Div:
    match element:
        case "M" | "L":
            return html.Div([
                # dbc.Row([f"{elements[element]}"]),
                dbc.Row(
                    [
                        dbc.Col(
                            "z:",
                            style={
                                "display": "inline-block",
                            },
                        ),
                        dbc.Input(
                            type="number",
                            placeholder="mm",
                            min=0,
                            id="pos",
                            style={
                                "width": "5rem",
                                "display": "inline-block",
                            },
                        ),
                        dbc.Col(
                            "f:",
                            style={
                                "display": "inline-block",
                            },
                        ),
                        dbc.Input(
                            type="number",
                            placeholder="mm",
                            min=0,
                            id="focal_length",
                            style={
                                "width": "5rem",
                                "display": "inline-block",
                            },
                        ),
                        dbc.Col(
                            "r:",
                            style={
                                "display": "inline-block",
                            },
                        ),
                        dbc.Input(
                            type="number",
                            placeholder="mm",
                            min=0,
                            id="aperture_radius",
                            style={
                                "width": "5rem",
                                "display": "inline-block",
                            },
                        ),
                    ],
                ),
            ])
        case "A":
            return html.Div([
                # dbc.Row([f"{elements[element]}"]),
                dbc.Row(
                    [
                        dbc.Col(
                            "z:",
                            style={
                                "display": "inline-block",
                            },
                        ),
                        dbc.Input(
                            type="number",
                            placeholder="mm",
                            min=0,
                            id="pos",
                            style={
                                "width": "5rem",
                                "display": "inline-block",
                            },
                        ),
                        dbc.Col(
                            "f:",
                            style={
                                "display": "inline-block",
                            },
                        ),
                        dbc.Input(
                            type="number",
                            placeholder="np.inf",
                            min=0,
                            value=np.inf,
                            id="focal_length",
                            disabled=True,
                            style={
                                "width": "5rem",
                                "display": "inline-block",
                            },
                        ),
                        dbc.Col(
                            "r:",
                            style={
                                "display": "inline-block",
                            },
                        ),
                        dbc.Input(
                            type="number",
                            placeholder="mm",
                            min=0,
                            id="aperture_radius",
                            style={
                                "width": "5rem",
                                "display": "inline-block",
                            },
                        ),
                    ],
                ),
            ])


@app.callback(
    Output("elements-container", "children"),
    Output("element-select", "value"),
    Input("element-select", "value"),
    State("elements-container", "children"),
    prevent_initial_call=True,
)
def make_element(elem: str, children: html.Div) -> html.Div:
    print(children)
    new_element = html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Button(
                            "X",
                            title="Close",
                            id="close",
                            className="me-1",
                            style={
                                # "background-color": "lightgreen",
                                # "height": "5px",
                                "text-align": "center",
                                "display": "inline-block",
                            },
                        ),
                        width=4,
                    ),
                    dbc.Col(
                        f"{elements[elem]} {len(children)}",
                        width=8,
                        className="h-50",
                    ),
                ],
                style={"width": "12rem"},
            ),
            parse_element(elem),
        ],
        id=str(len(children)),
    )
    children.append(new_element)
    return children, None


sidebar = html.Div(
    [
        html.H2("Optics"),
        html.Hr(),
        html.P("Add optical elements", className="lead"),
        dbc.Nav(
            [
                dcc.Dropdown(
                    options=[{"label": elements[key], "value": key} for key in elements],
                    # value=1,
                    placeholder="Choose element",
                    style={"color": "black", "width": "12rem"},
                    id="element-select",
                ),
                html.Br(),
            ],
            vertical=True,
            # pills=True,
        ),
        html.Div(id="elements-container", children=[]),
    ],
    style=SIDEBAR_STYLE,
)

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(sidebar, width=4),
        dbc.Col(
            dcc.Graph(id="graph", figure=fig, className="border"),
            width=8,
            style=PLOT_STYLE,
        ),
    ]),
])


if __name__ == "__main__":
    app.run_server(debug=True)
