import dash
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container(
    [
        dbc.Input(id="input-box", persistence=True, persistence_type="local"),
        dbc.Button("Clear Input", id="clear-button", className="mt-3"),
    ],
    className="p-5",
)


@app.callback(Output("input-box", "value"), [Input("clear-button", "n_clicks")])
def clear_input(n):
    if n:
        return ""
    return dash.no_update


if __name__ == "__main__":
    app.run_server(debug=True, port=1027)
