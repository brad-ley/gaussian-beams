# pyright: reportUnusedImport=false


# import dash_bootstrap_components as dbc

import ast
import re
import time
from io import BytesIO
from pathlib import Path

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, ctx, dcc, html, no_update
from dash.dependencies import ALL, Input, Output, State
from dash_bootstrap_templates import load_figure_template
from gaussian_beams import Aperture, Beam, Lens, Mirror
from matplotlib.figure import Figure

# adds templates to plotly.io
load_figure_template(["DARKLY", "MINTY_DARK"])

external_stylesheets = [
    dbc.icons.BOOTSTRAP,
    dbc.themes.BOOTSTRAP,
]

# app = DashProxy(
app = Dash(__name__, external_stylesheets=external_stylesheets)
# app = dash.Dash(
#     __name__,
#     external_stylesheets=external_stylesheets,
#     # external_scripts=[
#     #     "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML",
#     # ],
#     # transforms=[MultiplexerTransform()],
#     title="Gaussian Beam Simulator",
#     # suppress_callback_exceptions=True,
# )

TEMPLATE = "MINTY_DARK"


def make_figure() -> go.Figure:
    m = 60
    return go.Figure().update_layout(
        # template=TEMPLATE,
        template="plotly_dark",
        xaxis_title="z-distance (mm)",
        yaxis_title="Radius (mm)",
        margin={"l": 5 / 4 * m, "r": m / 2, "t": m / 2, "b": m},
    )


fig = make_figure()


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
    "padding": "2rem 1rem",
}

OPTIC_STYLE = {
    "margin-left": "0pt",
    "margin-top": "10px",
}


HEIGHT = 1


elements = {"M": "Mirror", "L": "Lens", "A": "Aperture"}
sim_elements = {"Mirror": Mirror, "Lens": Lens, "Aperture": Aperture}


def fig_to_png(fig) -> BytesIO:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=600)
    buf.seek(0)
    return buf


def parse_element(element: str, t: float, z: float = -1, r: float = -1, f: float = -1) -> html.Div:
    match element:
        case "M" | "L":
            dis = False
            pla = "inf"
            val = 125
        case "A":
            dis = True
            pla = "inf"
            val = np.inf

    z_input = dbc.Col(
        dbc.Input(
            type="number",
            placeholder="0 mm",
            min=0,
            # value=z,
            id={"type": "pos", "index": t},
            style={
                "width": "5.5rem",
                "margin-right": "10pt",
            },
        ),
        width="auto",
    )

    f_input = dbc.Col(
        dbc.Input(
            type="number",
            placeholder=pla,
            disabled=dis,
            min=0,
            # value=val,
            id={"type": "focal_length", "index": t},
            style={
                "width": "5.5rem",
                "margin-right": "10pt",
            },
        ),
        width="auto",
    )

    r_input = dbc.Col(
        dbc.Input(
            type="number",
            placeholder="25.4 mm",
            min=1e-6,
            # value=25.4,
            id={"type": "aperture_radius", "index": t},
            style={
                "width": "5.5rem",
            },
        ),
        width="auto",
    )

    if z != -1:
        z_input = dbc.Col(
            dbc.Input(
                type="number",
                placeholder="0 mm",
                min=0,
                value=z,
                id={"type": "pos", "index": t},
                style={
                    "width": "5.5rem",
                    "margin-right": "10pt",
                },
            ),
            width="auto",
        )

    if f != -1:
        f_input = dbc.Col(
            dbc.Input(
                type="number",
                placeholder=pla,
                disabled=dis,
                min=0,
                value=f,
                id={"type": "focal_length", "index": t},
                style={
                    "width": "5.5rem",
                    "margin-right": "10pt",
                },
            ),
            width="auto",
        )

    if r != -1:
        r_input = dbc.Col(
            dbc.Input(
                type="number",
                placeholder="25.4 mm",
                min=1e-6,
                value=r,
                id={"type": "aperture_radius", "index": t},
                style={
                    "width": "5.5rem",
                },
            ),
            width="auto",
        )

    return dbc.Container(
        [
            # dbc.Row([f"{elements[element]}"]),
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Markdown(
                            "z:",
                            style=OPTIC_STYLE,
                        ),
                        width="auto",
                    ),
                    z_input,
                    dbc.Col(
                        dcc.Markdown(
                            "f:",
                            style=OPTIC_STYLE,
                        ),
                        width="auto",
                    ),
                    f_input,
                    dbc.Col(
                        dcc.Markdown("r:", style=OPTIC_STYLE),
                        width="auto",
                    ),
                    r_input,
                ],
                className="g-0",
                justify="around",
            ),
        ],
        style={"margin-bottom": "15px"},
    )


# @app.callback(
#     Output("elements-container", "children"),
#     prevent_initial_call=True,
# )
# def del_all(_: int) -> list:
#     return []


@app.callback(
    Output("persist_optics", "data"),
    Output("sim-plot", "figure"),
    Output("insertion-loss", "children"),
    Output("download-data", "data"),
    Output("download-figure", "data"),
    Input("elements-container", "children"),
    # Input("beam-setup", "children"),
    Input({"type": "pos", "index": ALL}, "value"),
    Input({"type": "focal_length", "index": ALL}, "value"),
    Input({"type": "aperture_radius", "index": ALL}, "value"),
    Input("lam", "value"),
    Input("w0", "value"),
    Input("pct_encirc", "value"),
    Input("insertion-loss", "children"),
    Input("total-z", "value"),
    Input("download-data-button", "n_clicks"),
    Input("download-figure-button", "n_clicks"),
    prevent_initial_call=True,
)
def set_persist_state(
    children: list,
    # beam_setup: list,
    _: list,
    __: list,
    ___: list,
    lam: float,
    w0: float,
    encirc: float,
    insertion_loss_children: list,
    total_z: float,
    _dl_data: int,
    _dl_fig: int,
) -> list:  # stores the current state past browser refresh
    try:
        fig = make_figure()
        if len(children) > 0:
            z_tot = 0
            optics = []
            name = ""
            for child in children:
                z, f, r = -1, -1, -1
                elem_type = child["props"]["children"][0]["props"]["children"][0]["props"][
                    "children"
                ]["props"]["children"]
                name += f"_{next(key for key in elements if elements[key] == elem_type)}_"
                for e in child["props"]["children"][1]["props"]["children"][0]["props"][
                    "children"
                ]:
                    ee = e["props"]["children"]
                    if "id" in ee["props"] and "value" in ee["props"]:
                        match ee["props"]["id"]["type"]:
                            case "pos":
                                z = ee["props"]["value"]
                            case "focal_length":
                                f = ee["props"]["value"]
                            case "aperture_radius":
                                r = ee["props"]["value"]
                if z == -1 or z is None:
                    z = 0
                if r == -1 or r is None:
                    r = 25.4
                if f == -1 or f is None:
                    f = np.inf

                name += f"{f=:.2f},{r=:.2f}@{z=:.2f}"

                z_tot += z

                kwargs = {"f": f, "pos": z, "r": r}
                optics.append(sim_elements[elem_type](**kwargs))
                # print(*optics, sep="\n")
                match elem_type:
                    case "Lens":
                        fig.add_trace(
                            go.Scatter(
                                x=[z_tot, z_tot],
                                y=[-r, r],
                                showlegend=False,
                                name=f"f={f:.1f}mm lens",
                                marker_color="rgba(100, 100, 100, .9)",
                                line={"dash": "dash"},
                            ),
                        )
                    case "Mirror":
                        fig.add_trace(
                            go.Scatter(
                                x=[z_tot, z_tot],
                                y=[-r, r],
                                showlegend=False,
                                name=f"f={f:.1f}mm mirror",
                                marker_color="rgba(100, 100, 100, .9)",
                                line={"dash": "dashdot"},
                            ),
                        )
                    case "Aperture":
                        fig.add_trace(
                            go.Scatter(
                                x=[z_tot, z_tot],
                                y=[r, r + 5],
                                showlegend=False,
                                name=f"r={r:.1f}mm aperture",
                                marker_color="rgba(170, 0, 0, .9)",
                                # line={"dash": "dashdot"},
                            ),
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=[z_tot, z_tot],
                                y=[-r, -(r + 5)],
                                showlegend=False,
                                name=f"r={r:.1f}mm aperture",
                                marker_color="rgba(170, 0, 0, .9)",
                                # line={"dash": "dashdot"},
                            ),
                        )

            if total_z is not None:
                sim_z = np.linspace(0, total_z, 1000)
                beam = (
                    Beam(lam=lam, w=w0)
                    .simulate(elements=optics, z_axis=sim_z)
                    .encircled_ratio(encirc)
                )
            else:
                beam = Beam(lam=lam, w=w0).simulate(elements=optics).encircled_ratio(encirc)

            # fig = px.scatter(
            #     template="MINTY_DARK",
            # )
            # fig.add_scatter(
            fig.add_trace(
                go.Scatter(
                    x=beam.z_axis,
                    y=beam.w,
                    showlegend=False,
                    name=r"w(r)",
                    # marker_color="rgba(100, 255, 193, .9)",
                    marker_color="rgba(50, 205, 205, 0.9)",
                ),
            )
            # fig.add_scatter(
            fig.add_trace(
                go.Scatter(
                    x=beam.z_axis,
                    y=-beam.w,
                    showlegend=False,
                    name=r"-w(r)",
                    marker_color="rgba(50, 205, 205, 0.9)",
                    fill="tonexty",
                ),
            )

            fig.add_trace(
                go.Scatter(
                    x=beam.z_axis,
                    y=beam.w * beam.encirc_scale,
                    showlegend=False,
                    name=f"{encirc:.1f}% enc.",
                    # name=f"{encirc:.1f}% enc. ({beam.loss:.1f} dB)",
                    # marker_color="rgba(50, 205, 205, 1)",
                    marker_color="rgba(100, 255, 193, 0.9)",
                ),
            )
            fig.add_trace(
                go.Scatter(
                    x=beam.z_axis,
                    y=-beam.w * beam.encirc_scale,
                    showlegend=False,
                    name=f"{encirc:.1f}% enc.",
                    # name=f"{encirc:.1f}% enc. ({beam.loss:.1f} dB)",
                    # marker_color="rgba(50, 205, 205, 1)",
                    marker_color="rgba(100, 255, 193, 0.9)",
                ),
            )

            data_ret = no_update
            fig_ret = no_update
            name = name.lstrip("_")
            if total_z is not None:
                name += f"_to_{total_z}"

            match ctx.triggered[0]["prop_id"]:
                case "download-data-button.n_clicks":
                    kwargs = {"index": False}
                    d = pd.DataFrame(np.array([beam.z_axis, beam.w]).transpose())
                    data_ret = dcc.send_data_frame(
                        d.to_csv,
                        name + ".txt",
                        **kwargs,
                    )
                case "download-figure-button.n_clicks":
                    f = Figure(figsize=(7, 5))
                    a = f.add_subplot(111)
                    f, a = beam.plot(
                        encircled_energy=encirc,
                        # savepath=name + ".png",
                        fig=f,
                        ax=a,
                    )
                    # canvas = FigureCanvasAgg(f)
                    # canvas.print_figure(name + ".png", dpi=600)
                    png_image = fig_to_png(f)
                    fig_ret = dcc.send_bytes(png_image.getvalue(), name + ".png")

            insertion_loss_children = [insertion_loss_children[0], f"{beam.loss:.2f} dB"]

            return (children, fig, insertion_loss_children, data_ret, fig_ret)

    # except TypeError:
    except ValueError:
        return no_update, fig, no_update, no_update, no_update

    else:
        return no_update, fig, no_update, no_update, no_update


def create_elem(elem: str, **kwargs: dict) -> html.Div:
    t = time.time()
    return html.Div(
        [
            dbc.Row(
                [
                    # html.Div(
                    #     [f"{elements[elem]}"], style={"width": "5%", "display": "inline-block"}
                    # ),
                    dbc.Col(
                        html.H5(f"{elements[elem]}", style={"margin-top": "3pt"}),
                        width="auto",
                    ),
                    dbc.Col(
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Button(
                                        children=html.Span(
                                            [
                                                html.I(
                                                    className="bi bi-trash3",
                                                    style={
                                                        "display": "inline-block",
                                                    },
                                                ),
                                            ],
                                        ),
                                        title="Delete",
                                        id={"type": "del-elem", "index": t},
                                        color="danger",
                                        outline=True,
                                        className="sm",
                                        size="sm",
                                        # leftIcon=[DashIconify(icon="fluent:folder-mail-16-filled")],
                                    ),
                                    width="auto",
                                ),
                                dbc.Col(
                                    [
                                        dbc.Button(
                                            children=html.Span(
                                                [
                                                    html.I(
                                                        className="bi bi-arrow-up",
                                                        style={
                                                            "display": "inline-block",
                                                        },
                                                    ),
                                                ],
                                            ),
                                            # "X",
                                            title="Up",
                                            id={"type": "raise-elem", "index": t},
                                            color="secondary",
                                            outline=True,
                                            className="sm",
                                            size="sm",
                                            style={"margin-right": "3pt"},
                                        ),
                                        dbc.Button(
                                            children=html.Span(
                                                [
                                                    html.I(
                                                        className="bi bi-arrow-down",
                                                        style={
                                                            "display": "inline-block",
                                                        },
                                                    ),
                                                ],
                                            ),
                                            # "X",
                                            title="Down",
                                            id={"type": "lower-elem", "index": t},
                                            color="secondary",
                                            outline=True,
                                            className="sm",
                                            size="sm",
                                            style={"margin-right": "3pt"},
                                        ),
                                    ],
                                    width="auto",
                                ),
                            ],
                        ),
                        width="auto",
                    ),
                ],
                style={"margin-bottom": "3pt"},
                justify="between",
            ),
            parse_element(elem, t, **kwargs),
        ],
        id=f"elem-{t}",
    )


@app.callback(
    Output("elements-container", "children"),
    Output("element-select", "value"),
    Output("total-z", "value"),
    Output("upload-fig", "filename"),
    Input("element-select", "value"),
    Input({"type": "del-elem", "index": ALL}, "n_clicks"),
    Input("delete-all", "n_clicks"),
    Input({"type": "raise-elem", "index": ALL}, "n_clicks"),
    Input({"type": "lower-elem", "index": ALL}, "n_clicks"),
    Input("upload-fig", "filename"),
    State("persist_optics", "data"),
    State("elements-container", "children"),
    # prevent_initial_call=True, # initial call is useful to re-set optics if page refreshed
)
def handle_elements(
    elem: str,
    _: str,
    __: int,
    ___: int,
    ____: int,
    filename: str,
    persist: list,
    children: list,
) -> list:
    t_z = no_update

    if ctx.triggered[0]["prop_id"] == ".":  # initial callback
        children = persist

    elif ctx.triggered_id == "element-select":
        children.append(create_elem(elem))
        # return children, None
        # return None

    elif ctx.triggered_id == "upload-fig":
        children = []
        matches = p.findall(Path(filename).stem)

        for mat in matches:
            element = mat[0]
            for i, k in enumerate(mat[1:]):
                v = float(k) if k != "np.inf" else k

                if i == 0:
                    f = v
                elif i == 1:
                    r = v
                elif i == 2:  # noqa: PLR2004
                    z = v

            kwargs = {"z": z, "f": f, "r": r}
            children.append(create_elem(element, **kwargs))

        t_z_matches = p_t_z.findall(Path(filename).stem)

        t_z = float(t_z_matches[0]) if t_z_matches else None

    elif "del-elem" in ctx.triggered[0]["prop_id"]:
        idx = next(
            i
            for (i, val) in enumerate(children)
            if val["props"]["id"]
            == f"elem-{ast.literal_eval(ctx.triggered[0]['prop_id'].removesuffix('.n_clicks'))['index']}"
        )
        del children[idx]

    elif ctx.triggered[0]["prop_id"] == "delete-all.n_clicks":
        children = []

    elif "raise-elem" in ctx.triggered[0]["prop_id"]:
        idx = next(
            i
            for (i, val) in enumerate(children)
            if val["props"]["id"]
            == f"elem-{ast.literal_eval(ctx.triggered[0]['prop_id'].removesuffix('.n_clicks'))['index']}"
        )
        children.insert(idx - 1, children.pop(idx))

    elif "lower-elem" in ctx.triggered[0]["prop_id"]:
        idx = next(
            i
            for (i, val) in enumerate(children)
            if val["props"]["id"]
            == f"elem-{ast.literal_eval(ctx.triggered[0]['prop_id'].removesuffix('.n_clicks'))['index']}"
        )
        children.insert(idx + 1, children.pop(idx))

    return children, None, t_z, None


sidebar = html.Div(
    [
        html.H1("Optics"),
        html.Hr(),
        # html.P("Add optical elements", className="lead"),
        dbc.Nav(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.Markdown(
                                r"$$\lambda$$:",
                                mathjax=True,
                                style={"margin-left": "3pt", "margin-top": "10px"},
                            ),
                            width="auto",
                        ),
                        dbc.Col(
                            dbc.Input(
                                style={"width": "5rem"},
                                id="lam",
                                type="number",
                                placeholder="mm",
                                min=1e-6,
                                value=1.25,
                                persistence=True,
                                persistence_type="local",
                            ),
                            width="auto",
                        ),
                        dbc.Col(
                            dcc.Markdown(
                                r"$\omega_{in}$:",
                                mathjax=True,
                                style={"margin-left": "3pt", "margin-top": "10px"},
                            ),
                            width="auto",
                        ),
                        dbc.Col(
                            dbc.Input(
                                style={"width": "5rem"},
                                id="w0",
                                type="number",
                                placeholder="mm",
                                min=1e-6,
                                value=5.71,
                                persistence=True,
                                persistence_type="local",
                            ),
                            width="auto",
                        ),
                        dbc.Col(
                            dcc.Markdown(
                                r"% enc:",
                                mathjax=True,
                                style={"margin-left": "3pt", "margin-top": "10px"},
                            ),
                            width="auto",
                        ),
                        dbc.Col(
                            dbc.Input(
                                style={"width": "5rem"},
                                id="pct_encirc",
                                type="number",
                                placeholder="mm",
                                min=0,
                                max=99.99999,
                                value=np.round((1 - np.exp(-2)) * 100, 1),
                                persistence=True,
                                persistence_type="local",
                            ),
                            width="auto",
                        ),
                    ],
                    className="g-0",
                    # id="beam-setup",
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
                    children=html.Span(
                        [
                            html.I(
                                className="bi bi-trash3-fill",
                                style={"display": "inline-block", "margin-right": "5px"},
                            ),
                            html.Div(
                                "Delete all",
                                style={"paddingRight": "0.5vw", "display": "inline-block"},
                            ),
                        ],
                    ),
                    # "Delete all",
                    title="Delete all",
                    id="delete-all",
                    # className="me-1",
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


REGEX = r"([M,L,A])_f=([\d]*[a-z]*[\.]?[\d]*[a-z]*),r=([\d]*[a-z]*[\.]?[\d]*[a-z]*)@z=([\d]*[a-z]*[\.]?[\d]*[a-z]*)"
p = re.compile(REGEX)

T_Z_REGEX = r"to_([\d]*)"
p_t_z = re.compile(T_Z_REGEX)

app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(sidebar, width=4),
                dbc.Col(
                    [
                        dcc.Graph(id="sim-plot", figure=fig, className="border"),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    dcc.Markdown(
                                                        r"z-distance end (mm):",
                                                        mathjax=True,
                                                        style={
                                                            "margin-left": "3pt",
                                                            "margin-top": "10px",
                                                        },
                                                    ),
                                                    width="auto",
                                                ),
                                                dbc.Col(
                                                    dbc.Input(
                                                        type="number",
                                                        placeholder="auto",
                                                        min=0,
                                                        # value=-1,
                                                        id="total-z",
                                                        style={
                                                            "width": "5.5rem",
                                                        },
                                                        persistence=True,
                                                        persistence_type="local",
                                                    ),
                                                    width="auto",
                                                ),
                                            ],
                                        ),
                                    ],
                                    width="auto",
                                ),
                                dbc.Col(
                                    dcc.Markdown(
                                        children=["Insertion loss: "],
                                        id="insertion-loss",
                                        style={"margin-top": "10px"},
                                    ),
                                    width="auto",
                                ),
                            ],
                            style={"margin-top": "10px"},
                            id="under-graph",
                            justify="between",
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Button(
                                        children=html.Span(
                                            [
                                                html.I(
                                                    className="bi bi-download",
                                                    style={
                                                        "display": "inline-block",
                                                        "margin-right": "5px",
                                                    },
                                                ),
                                                html.Div(
                                                    "Download data",
                                                    style={
                                                        "paddingRight": "0.5vw",
                                                        "display": "inline-block",
                                                    },
                                                ),
                                                dcc.Download(id="download-data"),
                                            ],
                                        ),
                                        # "Delete all",
                                        title="Download data",
                                        id="download-data-button",
                                        # className="me-1",
                                        color="secondary",
                                        outline=True,
                                        style={
                                            # "background-color": "lightgreen",
                                            # "height": "5px",
                                            "text-align": "center",
                                            "width": "100%",
                                        },
                                    ),
                                    width=5,
                                ),
                                dbc.Col(
                                    dbc.Button(
                                        children=html.Span(
                                            [
                                                html.I(
                                                    className="bi bi-file-earmark-arrow-down",
                                                    style={
                                                        "display": "inline-block",
                                                        "margin-right": "5px",
                                                    },
                                                ),
                                                html.Div(
                                                    "Download figure",
                                                    style={
                                                        "paddingRight": "0.5vw",
                                                        "display": "inline-block",
                                                    },
                                                ),
                                                dcc.Download(id="download-figure"),
                                            ],
                                        ),
                                        # "Delete all",
                                        title="Download figure",
                                        id="download-figure-button",
                                        # className="me-1",
                                        color="secondary",
                                        outline=True,
                                        style={
                                            # "background-color": "lightgreen",
                                            # "height": "5px",
                                            "text-align": "center",
                                            "width": "100%",
                                        },
                                    ),
                                    width=5,
                                ),
                            ],
                            justify="evenly",
                        ),
                        dbc.Row(
                            dbc.Col(
                                dcc.Upload(
                                    children=html.Span(
                                        [
                                            html.I(
                                                className="bi bi-file-earmark-arrow-up",
                                                style={
                                                    "display": "inline-block",
                                                    "margin-right": "5px",
                                                },
                                            ),
                                            html.Div(
                                                "Drag-and-drop or click to upload configuration plot",
                                                style={
                                                    "paddingRight": "0.5vw",
                                                    "display": "inline-block",
                                                },
                                            ),
                                        ],
                                    ),
                                    # "Delete all",
                                    # title="Download figure",
                                    id="upload-fig",
                                    # className="me-1",
                                    # color="secondary",
                                    # outline=True,
                                    style={
                                        # "background-color": "lightgreen",
                                        # "height": "5px",
                                        # "margin-top": "3pt",
                                        # "text-align": "center",
                                        # "width": "34rem",
                                        # "width": "89%",
                                        "height": "50px",
                                        "lineHeight": "50px",
                                        "borderWidth": "1px",
                                        "borderStyle": "dashed",
                                        "borderRadius": "5px",
                                        "textAlign": "center",
                                        "margin-top": "10px",
                                        "margin-bottom": "10px",
                                        "margin-left": "55px",
                                        "margin-right": "55px",
                                    },
                                ),
                                # width="auto",
                            ),
                            justify="center",
                        ),
                        dbc.Row(
                            dbc.Col(
                                dcc.Markdown("""
                                             Code hosted at: [https://github.com/brad-ley/gaussian-beams](https://github.com/brad-ley/gaussian-beams)
                                             """),
                                width="auto",
                            ),
                            justify="center",
                        ),
                    ],
                    width=8,
                    style=PLOT_STYLE,
                ),
            ],
            className="g-0",
        ),
        dcc.Store(id="persist_optics", storage_type="local", data=[]),
        dcc.Store(
            id="beam-waist-data",
            storage_type="local",
        ),
        dcc.Store(id="filename-stem", storage_type="local", data=""),
        # dbc.Input(id="input-box", persistence=True, persistence_type="local"),
        # dbc.Button("Clear Input", id="clear-button", className="mt-3"),
    ],
    className="p-5",
)


# @app.callback(Output("input-box", "value"), [Input("clear-button", "n_clicks")])
# def clear_input(n):
#     if n:
#         return ""
#     return no_update


if __name__ == "__main__":
    # app.run_server(debug=True)

    app.run_server(
        debug=False,
        host="0.0.0.0",
        dev_tools_ui=False,
        dev_tools_props_check=False,
    )
    server = app.server
