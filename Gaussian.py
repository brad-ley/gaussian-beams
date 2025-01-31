import ast
import time

import dash_bootstrap_components as dbc
import numpy as np
import plotly.express as px
from dash import (
    ALL,
    State,
    callback_context,
    dcc,
    html,
)
from dash_bootstrap_templates import load_figure_template
from dash_extensions.enrich import DashProxy, Input, MultiplexerTransform, Output
from gaussian_beams import Aperture, Beam, Lens, Mirror

# adds templates to plotly.io
load_figure_template(["DARKLY", "MINTY_DARK"])

app = DashProxy(
    __name__,
    external_stylesheets=[dbc.themes.SLATE],
    transforms=[MultiplexerTransform()],
    # suppress_callback_exceptions=True,
)

fig = px.scatter(
    # df.query("year==2007"),
    # x="gdpPercap",
    # y="lifeExp",
    # size="pop",
    # color="continent",
    # log_x=True,
    # size_max=60,
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
HEIGHT = 1


elements = {"M": "Mirror", "L": "Lens", "A": "Aperture"}
sim_elements = {"Mirror": Mirror, "Lens": Lens, "Aperture": Aperture}


def parse_element(element: str, t: float) -> html.Div:
    match element:
        case "M" | "L":
            dis = False
            pla = "mm"
            val = 125
        case "A":
            dis = True
            pla = "np.inf"
            val = np.inf
    return dbc.Container(
        [
            # dbc.Row([f"{elements[element]}"]),
            dbc.Row(
                [
                    dbc.Col(
                        "z:",
                        style={"display": "inline-block", "margin-top": "10px"},
                    ),
                    dbc.Input(
                        type="number",
                        placeholder="mm",
                        min=0,
                        value=0,
                        id={"type": "pos", "index": t},
                        style={
                            "width": "5rem",
                            "display": "inline-block",
                        },
                    ),
                    dbc.Col(
                        "f:",
                        style={"display": "inline-block", "margin-top": "10px"},
                    ),
                    dbc.Input(
                        type="number",
                        placeholder=pla,
                        disabled=dis,
                        min=0,
                        value=val,
                        id={"type": "focal_length", "index": t},
                        style={
                            "width": "5rem",
                            "display": "inline-block",
                        },
                    ),
                    dbc.Col(
                        "r:",
                        style={"display": "inline-block", "margin-top": "10px"},
                    ),
                    dbc.Input(
                        type="number",
                        placeholder="25.4 mm",
                        min=0,
                        value=50.8,
                        id={"type": "aperture_radius", "index": t},
                        style={
                            "width": "5rem",
                            "display": "inline-block",
                        },
                    ),
                ],
            ),
        ],
        style={"margin-bottom": "15px"},
    )


@app.callback(
    Input("delete-all", "n_clicks"),
    Output("elements-container", "children"),
    prevent_initial_call=True,
)
def del_all(_: int) -> list:
    return []


@app.callback(
    Output("elements-container", "children"),
    Input({"type": "del-elem", "index": ALL}, "n_clicks"),
    State("elements-container", "children"),
    prevent_initial_call=True,
)
def del_element(_: str, children: list) -> list:
    if callback_context.triggered[0]["value"]:
        # print(callback_context.triggered[0]["prop_id"])
        idx = next(
            i
            for (i, val) in enumerate(children)
            if val["props"]["id"]
            == f"elem-{ast.literal_eval(callback_context.triggered[0]['prop_id'].removesuffix('.n_clicks'))['index']}"
        )
        del children[idx]

    # return update_indices(children)
    return children


@app.callback(
    Output("persist", "data"),
    Output("sim-plot", "figure"),
    Input("elements-container", "children"),
    Input({"type": "pos", "index": ALL}, "value"),
    Input({"type": "focal_length", "index": ALL}, "value"),
    Input({"type": "aperture_radius", "index": ALL}, "value"),
    Input("lam", "value"),
    Input("w0", "value"),
    prevent_initial_call=True,
)
def set_persist_state(
    children: list, _, __, ___, lam, w0,
) -> list:  # stores the current state past browser refresh
    print("=======")
    if children:
        optics = []
        for child in children:
            z, f, r = -1, -1, -1
            elem_type = child["props"]["children"][0]["props"]["children"][1]["props"]["children"][
                "props"
            ]["children"]
            for e in child["props"]["children"][1]["props"]["children"][0]["props"]["children"]:
                print(e)
                if "id" in e["props"] and "value" in e["props"]:
                    match e["props"]["id"]["type"]:
                        case "pos":
                            z = e["props"]["value"]
                        case "focal_length":
                            f = e["props"]["value"]
                        case "aperture_radius":
                            r = e["props"]["value"]
            if z == -1:
                z = 0
            if r == -1:
                r = 25.4
            if f == -1:
                f = np.inf

            kwargs = {"f": f, "pos": z, "r": r}
            print(kwargs)
            optics.append(sim_elements[elem_type](**kwargs))
            print(*optics, sep="\n")

        beam = Beam(lam=lam, w=w0).simulate(elements=optics)

        fig = px.scatter(
            template="MINTY_DARK",
        )
        fig.add_scatter(
            x=beam.z_axis,
            y=beam.w,
            showlegend=False,
            name=r"w(r)",
            marker_color="rgba(100, 255, 193, .9)",
        )
        fig.add_scatter(
            x=beam.z_axis,
            y=-beam.w,
            showlegend=False,
            name=r"-w(r)",
            marker_color="rgba(100, 255, 193, .9)",
        )

        return children, fig
    return children, px.scatter(template="MINTY_DARK")


@app.callback(
    Output("elements-container", "children"),
    Output("element-select", "value"),
    Input("element-select", "value"),
    State("persist", "data"),
    State("elements-container", "children"),
)
def create_element(elem: str, persist: list, children: list) -> list:
    if callback_context.triggered[0]["prop_id"] == ".":  # initial callback
        return persist, None

    t = time.time()
    new_element = html.Div(
        [
            dbc.Row(
                [
                    # html.Div(
                    #     [f"{elements[elem]}"], style={"width": "5%", "display": "inline-block"}
                    # ),
                    dbc.Col(
                        dbc.Button(
                            "X",
                            title="Delete",
                            id={"type": "del-elem", "index": t},
                            color="danger",
                            outline=True,
                            className="sm",
                            size="sm",
                        ),
                        width="auto",
                    ),
                    dbc.Col(
                        html.H5(f"{elements[elem]}", style={"margin-top": "3pt"}),
                        width="auto",
                    ),
                ],
                style={"margin-bottom": "3pt"},
            ),
            parse_element(elem, t),
        ],
        id=f"elem-{t}",
    )
    print(new_element)
    children.append(new_element)
    return children, None


sidebar = html.Div(
    [
        html.H1("Optics"),
        html.Hr(),
        # html.P("Add optical elements", className="lead"),
        dbc.Nav(
            [
                dbc.Row(
                    [
                        dbc.Col(dcc.Markdown(r"$$\lambda$$:", mathjax=True), width="auto"),
                        dbc.Col(
                            dbc.Input(
                                style={"width": "5rem"},
                                id="lam",
                                type="number",
                                placeholder="mm",
                                min=1e-6,
                                value=1.25,
                            ),
                            width="auto",
                        ),
                        dbc.Col(dcc.Markdown(r"$$\omega_{in}$$:", mathjax=True), width="auto"),
                        dbc.Col(
                            dbc.Input(
                                style={"width": "5rem"},
                                id="w0",
                                type="number",
                                placeholder="mm",
                                min=1e-6,
                                value=5.71,
                            ),
                            width="auto",
                        ),
                    ],
                ),
                dcc.Dropdown(
                    options=[{"label": elements[key], "value": key} for key in elements],
                    # value=1,
                    placeholder="Choose element",
                    style={
                        "color": "black",
                        # "width": "12rem",
                        "margin-bottom": "5px",
                        "margin-top": "5px",
                    },
                    id="element-select",
                ),
                dbc.Col(id="elements-container", children=[]),
                dbc.Button(
                    "Delete all",
                    title="Delete all",
                    id="delete-all",
                    className="me-1",
                    color="danger",
                    outline=True,
                    style={
                        # "background-color": "lightgreen",
                        # "height": "5px",
                        "text-align": "center",
                        # "width": "12rem",
                    },
                ),
            ],
            vertical=True,
            # pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

# app.clientside_callback(
#     ClientsideFunction(namespace="clientside", function_name="make_draggable"),
#     Output("elements-container", "data-drag"),
#     Input({"type": "draggable_item", "index": ALL}, "id"),
# )

app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(sidebar, width=4),
                dbc.Col(
                    dcc.Graph(id="sim-plot", figure=fig, className="border"),
                    width=8,
                    style=PLOT_STYLE,
                ),
            ],
        ),
        dcc.Store(id="persist", storage_type="session", data=[]),
    ],
)


if __name__ == "__main__":
    app.run_server(debug=True)
