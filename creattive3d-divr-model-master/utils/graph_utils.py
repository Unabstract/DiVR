import ast

# Define a safe eval function to avoid directly using eval()
def safe_eval(x):
    try:
        return ast.literal_eval(x)
    except ValueError:
        return None

# Define the Node class with all required properties
class Node:
    def __init__(self, id, interactable, localization, movable, color, presence,
                 x_min, y_min, x_max, y_max, x_cent, y_cent):
        self.id = id
        self.interactable = interactable
        self.localization = localization
        self.movable = movable
        self.color = color
        self.presence = presence
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
        self.x_cent = x_cent
        self.y_cent = y_cent

class Edge:
    def __init__(self, node_a, node_b, dynamic_type, adjacent_type, location_type, interaction_type, active, distance):
        self.node_a = node_a
        self.node_b = node_b
        self.dynamic_type = dynamic_type
        self.adjacent_type = adjacent_type
        self.location_type = location_type
        self.interaction_type = interaction_type
        self.active = active
        self.distance = distance

# function to convert Node objects to a list of features
def node_to_features(node, x_norm, y_norm):
    # # Handle '?' by setting a default value for now, this should be dynamically calculated
    # # Convert the xpos to a binary value, assuming 'Norm' is 0.0 and 'YPos' is 1.0
    # xpos_value = 0.0 if node.xpos == 'Norm' else 1.0

    return [
        float(node.interactable),
        float(node.localization),
        float(node.movable),
        float(node.color),
        float(node.presence),
        float(node.x_min/x_norm),
        float(node.y_min/y_norm),
        float(node.x_max/x_norm),
        float(node.y_max/y_norm),
        float(node.x_cent/x_norm),
        float(node.y_cent/y_norm)
    ]


def edge_to_features(edge, dist_norm):

    if edge.distance != -100:
        norm_distance = float(edge.distance/dist_norm)
    else:
        norm_distance = -10.0
    return [
        float(edge.dynamic_type),
        float(edge.adjacent_type),
        float(edge.interaction_type),
        float(edge.location_type),
        float(edge.active),
        norm_distance
    ]

def edge_to_index(edge):

    source_node = int(edge.node_a)
    target_node = int(edge.node_b)

    return source_node,target_node
