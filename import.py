import traceback
import networkx as nx
import osmnx as ox
import dash
import dash_leaflet as dl
from dash import (
    html,
    dcc,
    Input,
    Output,
    State,
    ctx,
)
from dash.exceptions import PreventUpdate

# DASH APP
app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    prevent_initial_callbacks="initial_duplicate",
)
server = app.server

# GLOBALS
GLOBAL_GRAPH = None
GLOBAL_GRAPH_TYPE = "fallback"
GLOBAL_LANDMARKS = {}
AVERAGE_WALK_SPEED = 1.4

# CAMPUSES
CAMPUSES = {
    "rvce": {
        "label": "RV College of Engineering",
        "point": (12.9237, 77.4987),
        "dist": 550,
        "landmarks": {
            "main gate": (12.9231, 77.4981),
            "library": (12.9241, 77.4990),
            "admin": (12.9246, 77.4997),
            "canteen": (12.9242, 77.4995),
            "hostel": (12.9252, 77.4979),
            "mechanical": (12.9249, 77.5002),
            "ground": (12.9229, 77.4975),
        },
    },
    "bms": {
        "label": "BMS College of Engineering",
        "point": (12.9416, 77.5669),
        "dist": 500,
        "landmarks": {
            "main gate": (12.9411, 77.5663),
            "library": (12.9420, 77.5671),
            "admin": (12.9425, 77.5676),
            "canteen": (12.9419, 77.5675),
            "hostel": (12.9430, 77.5665),
            "ground": (12.9408, 77.5668),
        },
    },
    "msrit": {
        "label": "MS Ramaiah Institute of Technology",
        "point": (13.0345, 77.5573),
        "dist": 600,
        "landmarks": {
            "main gate": (13.0338, 77.5569),
            "library": (13.0348, 77.5578),
            "admin": (13.0351, 77.5582),
            "canteen": (13.0347, 77.5571),
            "hostel": (13.0360, 77.5567),
            "ground": (13.0335, 77.5570),
        },
    },
}

# DEMO GRAPH
def fallback_graph():
    G = nx.Graph()
    nodes = {
        "Library": (12.9716, 77.5946),
        "Cafeteria": (12.9720, 77.5950),
        "Lab": (12.9725, 77.5960),
        "Hostel": (12.9730, 77.5955),
        "Admin": (12.9735, 77.5965),
    }
    edges = [
        ("Library", "Cafeteria", 200),
        ("Library", "Lab", 400),
        ("Cafeteria", "Hostel", 300),
        ("Hostel", "Admin", 200),
        ("Lab", "Admin", 100),
        ("Cafeteria", "Lab", 250),
        ("Library", "Hostel", 900),
    ]
    for name, (lat, lon) in nodes.items():
        G.add_node(
            name,
            y=lat,
            x=lon,
        )
    for u, v, length in edges:
        G.add_edge(
            u,
            v,
            length=length,
        )
    return G

# HELPERS
def get_campus(name):
    name = name.lower().strip()
    for key, value in CAMPUSES.items():
        if key in name:
            return key, value
        if value["label"].lower() in name:
            return key, value
    return None, None

def build_osm_graph(campus_name):
    try:
        key, campus = get_campus(campus_name)
        if campus is None:
            return fallback_graph(), "fallback", {}, "Campus not found."
        point = campus["point"]
        G = ox.graph_from_point(
            point,
            dist=campus["dist"],
            network_type="walk",
            simplify=True,
        )
        for _, _, _, data in G.edges(keys=True, data=True):
            data.setdefault("length", 1)
        largest_cc = max(
            nx.weakly_connected_components(G),
            key=len,
        )
        G = G.subgraph(largest_cc).copy()
        return (
            G,
            "osm",
            campus["landmarks"],
            None,
        )
    except Exception as e:
        print(traceback.format_exc())
        return fallback_graph(), "fallback", {}, str(e)

def graph_to_simple(G):
    UG = nx.Graph()
    for u, v, data in G.edges(data=True):
        weight = data.get("length", 1)
        if UG.has_edge(u, v):
            if weight < UG[u][v]["length"]:
                UG[u][v]["length"] = weight
        else:
            UG.add_edge(
                u,
                v,
                length=weight,
            )
    for node, data in G.nodes(data=True):
        UG.add_node(node, **data)
    return UG

def geocode_place(place_name):
    if GLOBAL_GRAPH_TYPE == "fallback":
        return place_name
    place = place_name.lower().strip()
    for key, value in GLOBAL_LANDMARKS.items():
        if key in place:
            return value
    raise ValueError(
        f"""
Location '{place_name}' not found.
Available landmarks:
{', '.join(GLOBAL_LANDMARKS.keys())}
"""
    )

def get_edge(G, u, v):
    edge_data = G.get_edge_data(u, v)
    if edge_data is None:
        edge_data = G.get_edge_data(v, u)
    if edge_data is None:
        return None
    if isinstance(edge_data, dict):
        if "length" in edge_data:
            return edge_data
        return min(
            edge_data.values(),
            key=lambda x: x.get("length", 1),
        )
    return None

# DISTANCE + STATS
def path_distance(G, path):
    total = 0
    for u, v in zip(path[:-1], path[1:]):
        edge = get_edge(G, u, v)
        if edge:
            total += edge.get("length", 1)
    return round(total, 2)

def walking_time(distance):
    return round(
        (distance / AVERAGE_WALK_SPEED) / 60,
        2,
    )

def route_stats(G, path):
    dist = path_distance(G, path)
    turns = max(0, len(path) - 2)
    return {
        "distance": dist,
        "time": walking_time(dist),
        "turns": turns,
    }

# HEURISTIC
def heuristic(G, a, b):
    lat1 = G.nodes[a]["y"]
    lon1 = G.nodes[a]["x"]
    lat2 = G.nodes[b]["y"]
    lon2 = G.nodes[b]["x"]
    return ox.distance.great_circle(
        lat1,
        lon1,
        lat2,
        lon2,
    )

# ROUTING
def compute_routes(G, start_place, end_place):
    try:
        if GLOBAL_GRAPH_TYPE == "fallback":
            start_node = start_place.strip()
            end_node = end_place.strip()
            if start_node not in G.nodes:
                return None, None, None, None, None, None, "Invalid start node."
            if end_node not in G.nodes:
                return None, None, None, None, None, None, "Invalid destination node."
            routing_graph = G
        else:
            start_geo = geocode_place(start_place)
            end_geo = geocode_place(end_place)
            start_node = ox.distance.nearest_nodes(
                G,
                X=start_geo[1],
                Y=start_geo[0],
            )
            end_node = ox.distance.nearest_nodes(
                G,
                X=end_geo[1],
                Y=end_geo[0],
            )
            routing_graph = graph_to_simple(G)
        
        # DIJKSTRA
        dijkstra_path = nx.shortest_path(
            routing_graph,
            source=start_node,
            target=end_node,
            weight="length",
        )
        
        # A STAR
        astar_path = nx.astar_path(
            routing_graph,
            start_node,
            end_node,
            heuristic=lambda a, b: heuristic(routing_graph, a, b),
            weight="length",
        )
        
        # FLOYD WARSHALL
        if len(routing_graph.nodes) <= 150:
            predecessors, _ = nx.floyd_warshall_predecessor_and_distance(
                routing_graph,
                weight="length",
            )
            floyd_path = nx.reconstruct_path(
                start_node,
                end_node,
                predecessors,
            )
        else:
            floyd_path = dijkstra_path
        
        # ALTERNATIVE PATHS
        alt_paths = list(
            nx.shortest_simple_paths(
                routing_graph,
                source=start_node,
                target=end_node,
                weight="length",
            )
        )[:3]
        
        return (
            dijkstra_path,
            floyd_path,
            astar_path,
            alt_paths,
            start_node,
            end_node,
            None,
        )
    except Exception as e:
        print(traceback.format_exc())
        return None, None, None, None, None, None, str(e)

# DIRECTIONS
def generate_directions(path):
    directions = []
    for i, node in enumerate(path):
        if i == 0:
            directions.append(f"Start from {node}")
        elif i == len(path) - 1:
            directions.append(f"Reach destination at {node}")
        else:
            directions.append(f"Continue via {node}")
    return directions

# DECISION EXPLANATION ENGINE
def explain_route_decision(
    G,
    selected_path,
    alternative_paths,
    algorithm,
):
    best_distance = path_distance(G, selected_path)
    stats = route_stats(G, selected_path)
    explanation = []
    
    explanation.append(html.H2(f"{algorithm} Decision Analysis"))
    explanation.append(html.Div("The algorithm selected this route because its cumulative traversal cost was lower than competing alternatives."))
    explanation.append(html.Br())
    
    explanation.append(html.Div(f"Distance: {best_distance} meters"))
    explanation.append(html.Div(f"Estimated Walking Time: {stats['time']} minutes"))
    explanation.append(html.Div(f"Turns Required: {stats['turns']}"))
    explanation.append(html.Br())
    
    explanation.append(html.H3("Chosen Route"))
    explanation.append(html.Div(" → ".join(map(str, selected_path))))
    explanation.append(html.Br())
    
    explanation.append(html.H3("Directions"))
    directions = generate_directions(selected_path)
    explanation.append(html.Ul([html.Li(step) for step in directions]))
    explanation.append(html.Br())
    
    explanation.append(html.H3("Rejected Alternatives"))
    for i, route in enumerate(alternative_paths[1:], start=2):
        dist = path_distance(G, route)
        extra = round(dist - best_distance, 2)
        turns = len(route) - 2
        reasons = []
        if dist > best_distance:
            reasons.append(f"{extra}m longer")
        if turns > stats["turns"]:
            reasons.append("more turns")
        if turns < stats["turns"]:
            reasons.append("fewer turns but longer")
        explanation.append(
            html.Div([
                html.B(f"Alternative {i}"),
                html.Div(" → ".join(map(str, route))),
                html.Div(f"Distance: {dist}m"),
                html.Div(f"Reason Rejected: {', '.join(reasons)}"),
                html.Br(),
            ])
        )
        
    explanation.append(html.H3("Algorithm Logic"))
    if algorithm == "Dijkstra":
        explanation.append(html.Div("Dijkstra explores routes in increasing order of cumulative path cost and guarantees the globally optimal shortest route."))
    elif algorithm == "A*":
        explanation.append(html.Div("A* uses both current travel cost and estimated remaining distance to guide search toward the destination efficiently."))
    elif algorithm == "Floyd Warshall":
        explanation.append(html.Div("Floyd Warshall computes shortest paths between all node pairs using dynamic programming and retrieves the minimum-cost route."))
    
    explanation.append(html.Br())
    explanation.append(html.H3("Final Conclusion"))
    explanation.append(html.Div(f"{algorithm} determined that the selected route provides the most efficient path between the chosen locations based on distance and traversal cost."))
    return html.Div(explanation)

# PATH TO COORDS
def path_to_coordinates(G, path):
    coords = []
    if not path:
        return coords
    for u, v in zip(path[:-1], path[1:]):
        edge = get_edge(G, u, v)
        if edge is None:
            continue
        if "geometry" in edge:
            xs, ys = edge["geometry"].xy
            segment = [(y, x) for x, y in zip(xs, ys)]
            if coords and coords[-1] == segment[0]:
                coords.extend(segment[1:])
            else:
                coords.extend(segment)
        else:
            if not coords:
                coords.append((G.nodes[u]["y"], G.nodes[u]["x"]))
            coords.append((G.nodes[v]["y"], G.nodes[v]["x"]))
    return coords

# MAP
def draw_map(
    G,
    dijkstra_path=None,
    floyd_path=None,
    astar_path=None,
    alt_paths=None,
    start_node=None,
    end_node=None,
):
    coords = [
        [d["y"], d["x"]]
        for _, d in G.nodes(data=True)
        if "y" in d and "x" in d
    ]
    center = coords[0] if coords else [12.9237, 77.4987]
    children = [dl.TileLayer()]
    
    if alt_paths:
        for route in alt_paths[1:]:
            children.append(
                dl.Polyline(
                    positions=path_to_coordinates(G, route),
                    color="gray",
                    weight=4,
                    opacity=0.5,
                )
            )
    if dijkstra_path:
        children.append(dl.Polyline(positions=path_to_coordinates(G, dijkstra_path), color="orange", weight=8))
    if floyd_path:
        children.append(dl.Polyline(positions=path_to_coordinates(G, floyd_path), color="blue", weight=5))
    if astar_path:
        children.append(dl.Polyline(positions=path_to_coordinates(G, astar_path), color="green", weight=5))
        
    if start_node in G.nodes:
        children.append(dl.Marker(position=[G.nodes[start_node]["y"], G.nodes[start_node]["x"]], children=[dl.Tooltip("Start")]))
    if end_node in G.nodes:
        children.append(dl.Marker(position=[G.nodes[end_node]["y"], G.nodes[end_node]["x"]], children=[dl.Tooltip("Destination")]))
        
    return dl.Map(
        center=center,
        zoom=17,
        children=children,
        style={"width": "100%", "height": "650px", "borderRadius": "15px"},
    )

# STYLES
CARD = {
    "backgroundColor": "white",
    "padding": "30px",
    "borderRadius": "18px",
    "boxShadow": "0px 4px 15px rgba(0,0,0,0.1)",
    "marginBottom": "20px",
}
INPUT_STYLE = {
    "width": "100%",
    "height": "60px",
    "padding": "10px 15px",
    "fontSize": "22px",
    "borderRadius": "12px",
    "border": "2px solid #d1d5db",
    "marginTop": "8px",
    "marginBottom": "25px",
}
BUTTON_STYLE = {
    "backgroundColor": "#2563eb",
    "color": "white",
    "padding": "14px 22px",
    "border": "none",
    "borderRadius": "12px",
    "fontSize": "16px",
    "fontWeight": "bold",
    "marginRight": "10px",
    "marginTop": "10px",
    "cursor": "pointer",
}

# LAYOUT
app.layout = html.Div(
    style={"backgroundColor": "#f4f6f9", "padding": "30px", "fontFamily": "Arial"},
    children=[
        html.H1("Smart Campus Navigation System", style={"textAlign": "center", "fontSize": "42px"}),
        html.Div(
            style=CARD,
            children=[
                html.H2("Navigation Controls"),
                html.Label("Campus Name"),
                dcc.Input(id="campus-input", value="RVCE", style=INPUT_STYLE),
                html.Label("Start Location"),
                dcc.Input(id="start-input", value="library", style=INPUT_STYLE),
                html.Label("Destination"),
                dcc.Input(id="end-input", value="main gate", style=INPUT_STYLE),
                html.Button("Load Campus", id="btn-load", n_clicks=0, style=BUTTON_STYLE),
                html.Button("Run Dijkstra", id="btn-dijkstra", n_clicks=0, style=BUTTON_STYLE),
                html.Button("Run Floyd", id="btn-floyd", n_clicks=0, style=BUTTON_STYLE),
                html.Button("Run A*", id="btn-astar", n_clicks=0, style=BUTTON_STYLE),
                html.Button("Compare All", id="btn-compare", n_clicks=0, style=BUTTON_STYLE),
            ],
        ),
        html.Div(id="analytics", style=CARD),
        dcc.Loading(type="circle", children=[html.Div(id="map-container")]),
    ],
)

# LOAD GRAPH
@app.callback(
    Output("analytics", "children"),
    Output("map-container", "children"),
    Input("btn-load", "n_clicks"),
    State("campus-input", "value"),
    prevent_initial_call=True,
)
def load_graph(_, campus_name):
    global GLOBAL_GRAPH
    global GLOBAL_GRAPH_TYPE
    global GLOBAL_LANDMARKS
    if not campus_name:
        raise PreventUpdate
    if campus_name.lower().strip() == "demo":
        G = fallback_graph()
        GLOBAL_GRAPH = G
        GLOBAL_GRAPH_TYPE = "fallback"
        GLOBAL_LANDMARKS = {}
        return (
            html.Div([
                html.H2("Demo Graph Loaded"),
                html.Div(f"Nodes: {len(G.nodes)}"),
                html.Div(f"Edges: {len(G.edges)}"),
            ]),
            draw_map(G),
        )
    G, graph_type, landmarks, error = build_osm_graph(campus_name)
    GLOBAL_GRAPH = G
    GLOBAL_GRAPH_TYPE = graph_type
    GLOBAL_LANDMARKS = landmarks
    if graph_type == "fallback":
        return (html.Div(f"Error: {error}"), draw_map(G))
    return (
        html.Div([
            html.H2(f"{campus_name} Loaded Successfully"),
            html.Div(f"Nodes: {len(G.nodes)}"),
            html.Div(f"Edges: {len(G.edges)}"),
        ]),
        draw_map(G),
    )


# --- VS CODE EXECUTION ENTRY POINT ---
if __name__ == "__main__":
    # debug=True allows the app to auto-reload in VS Code when you make code changes
    app.run_server(debug=True, port=8050)