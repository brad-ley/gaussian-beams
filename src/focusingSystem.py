import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rc
from pathlib import Path as P

from dash import (
    Dash,
    dcc,
    html,
    Input,
    Output,
    State,
    callback,
    ctx,
    no_update,
    ALL,
)
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template

# from dash_extensions.enrich import BlockingCallbackTransform, DashProxy
from plotly import graph_objects as go
import plotly.express as px

from gaussian_beams import Mirror, Lens, Aperture, Beam

# if __name__ == "__main__":
#     plt.style.use(["science"])
#     rc("text.latex", preamble=r"\usepackage{cmbright}")
#     rcParams = [
#         ["font.family", "sans-serif"],
#         ["font.size", 14],
#         ["axes.linewidth", 1],
#         ["lines.linewidth", 2],
#         ["xtick.major.size", 5],
#         ["xtick.major.width", 1],
#         ["xtick.minor.size", 2],
#         ["xtick.minor.width", 1],
#         ["ytick.major.size", 5],
#         ["ytick.major.width", 1],
#         ["ytick.minor.size", 2],
#         ["ytick.minor.width", 1],
#     ]
#     plt.rcParams.update(dict(rcParams))

# margin_vert = "0px"
# margin = dict(l=40, r=40, t=20, b=20)
# # graph_height = "325px"

# dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"
# app = Dash(
#     __name__,
#     # external_stylesheets=["https://codepen.io/chriddyp/pen/bWLwgP.css"],
#     # external_stylesheets=[dbc.themes.DARKLY],
#     external_stylesheets=[dbc.themes.BOOTSTRAP, dbc_css],  # type: ignore
#     suppress_callback_exceptions=True,
# )


# def update_fig(fig):
#     # colors = {"background": "#272b30", "text": "#AAAAAA"}
#     fig.update_layout({"uirevision": "foo"}, overwrite=True)
#     fig.update_layout(margin=margin)
#     fig.update_layout(
#         {
#             # "plot_bgcolor": colors["background"],
#             # "paper_bgcolor": colors["background"],
#             # "font": {"color": colors["text"]},
#         }
#     )

#     return fig


# def make_fig():
#     return px.line()


# fig = update_fig(make_fig())


# app.layout = html.Div(
#     [
#         html.Div(
#             [
#                 dcc.Graph(
#                     id="deconvolved",
#                     figure=fig,
#                     # style={"height": graph_height},
#                     className="dbc",
#                 )
#             ],
#             style={
#                 # "display": "inline-block",
#                 # "width": "49%",
#                 "horizontal-align": "middle",
#             },
#         ),
#         html.Div(
#             [
#                 html.Div(
#                     [
#                         html.Div(
#                             [
#                                 "Signal \u03d5 (rad):",
#                             ],
#                             style={
#                                 "display": "inline-block",
#                                 "margin": "10px 0px 0px 30px",
#                                 "width": "120px",
#                             },
#                         ),
#                         html.Div(
#                             [
#                                 dcc.Slider(
#                                     0,
#                                     np.pi / 2,
#                                     id="sigphase",
#                                     value=np.pi / 4,
#                                     marks=None,
#                                     tooltip={
#                                         "placement": "right",
#                                         "always_visible": True,
#                                     },
#                                     persistence=True,
#                                 )
#                             ],
#                             style={
#                                 "width": "24.5%",
#                                 "display": "inline-block",
#                                 "margin": "0px 0px -25px 0px",
#                             },
#                         ),
#                         html.Button(
#                             id="findphase",
#                             n_clicks=0,
#                             children=("Auto"),
#                             style={
#                                 "background-color": "lightgreen",
#                                 "margin": "0px 10px -25px 0px",
#                                 "display": "inline-block",
#                                 "text-align": "center",
#                             },
#                         ),
#                         html.Button(
#                             id="addpi",
#                             n_clicks=0,
#                             children=("+\u03C0/2"),
#                             style={
#                                 "background-color": "lightblue",
#                                 "margin": "0px 10px -25px 0px",
#                                 "display": "inline-block",
#                                 "text-align": "center",
#                             },
#                         ),
#                         html.Button(
#                             id="fitbutton",
#                             n_clicks=0,
#                             children="Fit",
#                             style={
#                                 "background-color": "orange",
#                                 "margin": "0px 0px -25px 0px",
#                                 "display": "inline-block",
#                                 "text-align": "center",
#                             },
#                         ),
#                     ]
#                 ),
#                 html.Div(
#                     [
#                         html.Div(
#                             [
#                                 "Fit range (\u03bcs):",
#                             ],
#                             style={
#                                 "display": "inline-block",
#                                 "margin": "10px 0px 0px 30px",
#                                 "width": "120px",
#                             },
#                         ),
#                         html.Div(
#                             [
#                                 dcc.RangeSlider(
#                                     min=0,
#                                     max=20,
#                                     id="timerange",
#                                     value=[5, 15],
#                                     marks=None,
#                                     tooltip={
#                                         "placement": "right",
#                                         "always_visible": True,
#                                     },
#                                     persistence=True,
#                                     className="dbc",
#                                 )
#                             ],
#                             style={
#                                 "width": "70%",
#                                 "display": "inline-block",
#                                 "margin": "0px 0px -25px 0px",
#                             },
#                         ),
#                     ],
#                 ),
#                 html.Div(
#                     [
#                         html.Div(
#                             ["Averages:"],
#                             style={
#                                 "width": "11.5%",
#                                 "display": "inline-block",
#                             },
#                         ),
#                         dcc.Input(
#                             id="averages",
#                             value=500,
#                             type="number",
#                             style={
#                                 "width": "15%",
#                                 "height": "50px",
#                                 # 'lineHeight': '50px',
#                                 "borderWidth": "1px",
#                                 "borderStyle": "line",
#                                 "borderRadius": "5px",
#                                 "textAlign": "left",
#                                 "margin": "0px 0px 10px 0%",
#                                 "display": "inline-block",
#                             },
#                         ),
#                         html.Div(id="fileout", children="Enter file above"),
#                         html.Div(id="fit_params", children="Fit parameters"),
#                         html.Button(
#                             id="save",
#                             n_clicks=0,
#                             children="Save deconvolved",
#                             style={
#                                 "background-color": "lightgreen",
#                                 "margin": "5px 10px 0px 0px",
#                                 "text-align": "center",
#                                 "display": "inline-block",
#                             },
#                         ),
#                         html.Button(
#                             id="batch",
#                             n_clicks=0,
#                             children="Deconvolve batch",
#                             style={
#                                 "background-color": "lightblue",
#                                 "margin": "5px 0px 0px 0px",
#                                 "text-align": "center",
#                                 "display": "inline-block",
#                             },
#                         ),
#                     ],
#                     style={"margin": "10px 0px 0px 30px"},
#                 ),
#             ],
#             style={
#                 # "width": "49%",
#                 # "display": "inline-block",
#                 # "verticalAlign": "top",
#             },
#         ),
#         html.Div(id="plots-on"),
#         dcc.Store(id="plots-shown-dict", storage_type="memory"),
#         dcc.Store(id="just-started", data=True, storage_type="memory"),
#     ],
# )


# @app.callback(Output("graph", "figure"), Input("dropdown", "value"))
# def display_color(color):
#     fig = go.Figure(
#         data=go.Bar(
#             y=[2, 3, 1],  # replace with your own data source
#             marker_color=color,
#         )
#     )
#     return fig


# if __name__ == "__main__":
#     app.run_server(debug=True, threaded=True, port=1028)
