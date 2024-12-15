import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pyomo.environ as pyo
from pyomo.network import Arc



def plot_network(m, stream_table, path_to_save=None):
    """
    Plot the network of the flowsheet.

    Args:
        m (pyomo.core.base.PyomoModel): The Pyomo model to plot.
        stream_table (pandas.DataFrame): The stream table to plot.
        path_to_save (str): The path to save the plot.

    Returns:
        None. Saves the plot to the path_to_save.
    """
    plt.figure(figsize=(8, 6))
    G = nx.DiGraph()

    # specify types of nodes
    included_types = [
        '_ScalarMixer', '_ScalarCSTR', '_ScalarCSTR_Injection', '_ScalarClarifier', 
        '_ScalarSeparator', '_ScalarProduct', '_ScalarAD', '_ScalarFeed', 
        '_ScalarPressureChanger', 'ScalarParam', '_ScalarDewateringUnit', 
        '_ScalarElectroNPZO',  
        '_ScalarThickener', 'OrderedScalarSet', '_ScalarTranslator_ADM1_ASM2D', 
        '_ScalarTranslator_ASM2d_ADM1'
    ]
    
    for unit in m.fs.component_objects(pyo.Block, descend_into=True):
        if type(unit).__name__ in included_types:
            G.add_node(unit.name.split('fs.')[-1])

    # edges with mass concentrations as labels
    for arc in m.fs.component_objects(Arc, descend_into=True):
        source_name = arc.source.parent_block().name.split('fs.')[-1]
        dest_name = arc.destination.parent_block().name.split('fs.')[-1]
        G.add_edge(source_name, dest_name)
        
        source_key = source_name
        column_mapping = {
            "thickener outlet": "thickener",
            "ADM-ASM translator outlet": "translator_asm2d_adm1",
            "dewater outlet": "dewater",
            "electroNP treated": "electroNP",
            "electroNP byproduct": "electroNP",
            "Treated water": "Treated",
            "Sludge": "Sludge",
            "Feed": "FeedWater"
        }

        # Check if source_key is one of the values in column_mapping, and if so, update source_key to the corresponding key
        source_key = next((k for k, v in column_mapping.items() if v == source_key), source_key)

        if source_key in stream_table.columns:
            S_N2 = np.round(stream_table.loc['Mass Concentration S_N2', source_key], 3)
            S_NO3 = np.round(stream_table.loc['Mass Concentration S_NO3', source_key], 3)
            S_NH4 = np.round(stream_table.loc['Mass Concentration S_NH4', source_key], 3)
            label = f"N2: {S_N2}\nNO3: {S_NO3}\nNH4: {S_NH4}"
            G.edges[source_name, dest_name]['label'] = label

    pos = nx.kamada_kawai_layout(G, weight='weight', scale=3, center=None, dim=2)  # Increased scale for more spread
    # Adjust node positions for better layout
    position_adjustments = {
        'MX3': [0, -0.1],
        'MX4': [0, -0.1], 
        'R1': [0, -0.1],
        'R2': [-0.2, 0.1],
        'R5': [0, 0.3],
        'R6': [0, 0.3],
        'R7': [0, -0.2],
        'SP1': [0.1, 0],
        'thickener': [-0.1, 0.2],
        'dewater': [-0.2, -0.1],
        'translator_adm1_asm2d': [0, -0.1],
        'translator_asm2d_adm1': [-0.4, 0]
    }
    
    for node, [dx, dy] in position_adjustments.items():
        if node in pos:
            pos[node][0] += dx
            pos[node][1] += dy
    nx.draw(G, pos, with_labels=True, node_size=500, node_color="skyblue", font_size=10, font_weight="bold", arrows=True)
    
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    
    plt.title("BSM2 + electroNP Flowsheet")
    if path_to_save is not None:
        plt.savefig(path_to_save, dpi=300)
    plt.show()
