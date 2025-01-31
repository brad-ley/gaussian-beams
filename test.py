import dash
from dash import MATCH, Patch, html
from dash_extensions import EventListener
from dash.dependencies import Input, Output, State, ClientsideFunction


app = dash.Dash(
    __name__,
    external_scripts=["https://cdnjs.cloudflare.com/ajax/libs/dragula/3.7.2/dragula.min.js"],
    prevent_initial_callbacks=True,
)

app.layout = html.Div(
    id="main",
    children=[
        EventListener(
            html.Div(id={"type": "drag_container", "idx": 0}, className="container", children=[]),
            events=[
                {
                    "event": "dropcomplete",
                    "props": ["detail.name", "detail.children"],
                }
            ],
            logging=True,
            id={"type": "el_drag_container", "idx": 0},
        ),
        html.Button(id={"type": "add_btn", "idx": 0}, children="Add"),
        html.Div(id={"type": "test_div", "idx": 0}),
    ],
)

app.clientside_callback(
    ClientsideFunction(namespace="clientside", function_name="make_draggable"),
    Output({"type": "drag_container", "idx": MATCH}, "data-drag"),
    [
        Input({"type": "drag_container", "idx": MATCH}, "id"),
        Input({"type": "add_btn", "idx": MATCH}, "n_clicks"),
    ],
)


@app.callback(
    Output({"type": "drag_container", "idx": MATCH}, "children"),
    Input({"type": "add_btn", "idx": MATCH}, "n_clicks"),
)
def add_element(n_clicks):
    patched_children = Patch()
    patched_children.append(
        html.Div(id={"type": "div", "idx": 0, "idx2": n_clicks}, children=f"Text {n_clicks}")
    )
    return patched_children


@app.callback(
    Output({"type": "test_div", "idx": MATCH}, "children"),
    Input({"type": "el_drag_container", "idx": MATCH}, "n_events"),
    State({"type": "el_drag_container", "idx": MATCH}, "event"),
)
def get_new_order(n_events, event):
    """Get new order of elements - can be used to synchronize children"""
    print(f"New order is: {event['detail.children']}")
    return str(event["detail.children"])


if __name__ == "__main__":
    app.run_server(debug=True, port=1027)
