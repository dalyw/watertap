import pyomo.environ as pyo
from pyomo.network import Arc

from dash import Dash, html
import dash_cytoscape as cyto


def plot_network(
    m,
    path_to_save=None,
):
    """
    Interactive flowsheet plot using Dash Cytoscape.
    - Launches a Dash app with a flowsheet diagram.
    - Unit nodes and port nodes are shown, arcs are edges.
    - Nodes are draggable in the browser.
    - Requires: pip install dash dash-cytoscape
    """
    internal_block_names = [
        "process_flow",
        "spent_regenerant",
        "fresh_regenerant",
        "properties_in",
        "properties_out",
    ]

    # Collect unit nodes
    unit_nodes = []
    for unit in m.fs.component_objects(pyo.Block, descend_into=True):
        node_name = unit.name.split("fs.")[-1]
        if (
            "properties" in unit.name
            or "costing" in unit.name
            or unit.name.endswith("_expanded")
            or any(name in unit.name for name in internal_block_names)
        ):
            continue
        unit_nodes.append(node_name)
    print("Unit nodes found:", unit_nodes)

    # Collect ports and all edges (unit-port, port-unit, port-port)
    used_ports = dict()  # (unit_name, port_name) -> set(['inlet', 'outlet'])
    arc_edges = []
    for arc in m.fs.component_objects(Arc, descend_into=True):
        source_block = arc.source.parent_block()
        dest_block = arc.destination.parent_block()
        if (
            "properties" in source_block.name
            or "costing" in source_block.name
            or "properties" in dest_block.name
            or "costing" in dest_block.name
            or any(name in source_block.name for name in internal_block_names)
            or any(name in dest_block.name for name in internal_block_names)
        ):
            continue
        source_unit = source_block.name.split("fs.")[-1]
        dest_unit = dest_block.name.split("fs.")[-1]
        source_port_name = arc.source.local_name
        dest_port_name = arc.destination.local_name
        used_ports.setdefault((source_unit, source_port_name), set()).add("outlet")
        used_ports.setdefault((dest_unit, dest_port_name), set()).add("inlet")
        arc_edges.append(
            (
                f"{source_unit}_{source_port_name}",
                f"{dest_unit}_{dest_port_name}",
                source_unit,
                dest_unit,
            )
        )
    print("Arc edges found:", arc_edges)

    port_nodes = [f"{unit}_{port}" for (unit, port) in used_ports]

    # Build Cytoscape elements: units, ports, and edges
    elements = []
    internal_state_blocks = []
    # Add unit nodes
    for idx, unit in enumerate(unit_nodes):
        elements.append(
            {
                "data": {"id": unit, "label": unit, "type": "unit"},
                "grabbable": True,
                "selectable": True,
            }
        )
    # Add port nodes
    for idx, port in enumerate(port_nodes):
        elements.append(
            {
                "data": {"id": port, "label": port, "type": "port"},
                "grabbable": True,
                "selectable": True,
            }
        )
    # Add internal state block node for IX unit (only fresh_regenerant)
    for unit in unit_nodes:
        if unit == "ion_exchange":
            internal_id = f"{unit}_fresh_regenerant"
            elements.append(
                {
                    "data": {
                        "id": internal_id,
                        "label": "fresh_regenerant",
                        "type": "internal_state",
                    },
                    "grabbable": False,
                    "selectable": False,
                }
            )
            # Connect to the IX unit
            elements.append({"data": {"source": internal_id, "target": unit}})
    # Add edges for arcs
    for src, dst, src_unit, dst_unit in arc_edges:
        elements.append(
            {
                "data": {"source": src, "target": dst},
            }
        )
    # Add edges for port-to-unit and unit-to-port
    for (unit, port), directions in used_ports.items():
        port_node = f"{unit}_{port}"
        if unit in unit_nodes:
            if "inlet" in directions:
                elements.append({"data": {"source": port_node, "target": unit}})
            if "outlet" in directions:
                elements.append({"data": {"source": unit, "target": port_node}})

    # Cytoscape stylesheet for node/edge appearance
    stylesheet = [
        {
            "selector": 'node[type="unit"]',
            "style": {
                "background-color": "#7fc7ff",
                "shape": "rectangle",
                "width": 60,
                "height": 40,
                "label": "data(label)",
                "font-size": 18,
            },
        },
        {
            "selector": 'node[type="port"]',
            "style": {
                "background-color": "#b6e3b6",
                "shape": "ellipse",
                "width": 30,
                "height": 30,
                "label": "data(label)",
                "font-size": 12,
            },
        },
        {
            "selector": 'node[type="internal_state"]',
            "style": {
                "background-color": "#ffa500",
                "shape": "diamond",
                "width": 40,
                "height": 40,
                "label": "data(label)",
                "font-size": 14,
                "border-width": 2,
                "border-color": "#b36b00",
            },
        },
        {
            "selector": "edge",
            "style": {
                "curve-style": "bezier",
                "target-arrow-shape": "triangle",
                "arrow-scale": 1.5,
                "width": 3,
                "line-color": "#888",
                "target-arrow-color": "#888",
                "font-size": 10,
            },
        },
    ]

    app = Dash(__name__)
    app.layout = html.Div(
        [
            html.H2("Interactive Flowsheet Network (Cytoscape)"),
            html.Div(
                [
                    html.Span(
                        style={
                            "display": "inline-block",
                            "width": "20px",
                            "height": "20px",
                            "background": "#7fc7ff",
                            "margin-right": "5px",
                            "border": "2px solid #333",
                        }
                    ),
                    html.Span("Unit", style={"margin-right": "20px"}),
                    html.Span(
                        style={
                            "display": "inline-block",
                            "width": "20px",
                            "height": "20px",
                            "background": "#b6e3b6",
                            "border-radius": "50%",
                            "margin-right": "5px",
                            "border": "2px solid #333",
                        }
                    ),
                    html.Span("Port", style={"margin-right": "20px"}),
                    html.Span(
                        style={
                            "display": "inline-block",
                            "width": "20px",
                            "height": "20px",
                            "background": "#ffa500",
                            "clip-path": "polygon(50% 0%, 100% 50%, 50% 100%, 0% 50%)",
                            "margin-right": "5px",
                            "border": "2px solid #b36b00",
                        }
                    ),
                    html.Span("Internal State", style={"margin-right": "20px"}),
                ],
                style={"margin-bottom": "10px"},
            ),
            cyto.Cytoscape(
                id="flowsheet-cytoscape",
                elements=elements,
                layout={"name": "cose"},  # auto-layout for better visibility
                style={
                    "width": "1200px",
                    "height": "800px",
                    "background-color": "#e6f0fa",
                },
                stylesheet=stylesheet,
                userPanningEnabled=True,
                userZoomingEnabled=True,
                boxSelectionEnabled=True,
                autoungrabify=False,
                autounselectify=False,
                minZoom=0.2,
                maxZoom=2,
            ),
        ]
    )
    print("Launching Dash Cytoscape app on a random available port (port=0)...")
    app.run(debug=True, port=0)
