from torch_geometric.data import Data

class SceneData(Data):
    def __init__(self, num_nodes=None, **kwargs):
        super(SceneData, self).__init__()
        self.num_nodes = num_nodes
        for key, item in kwargs.items():
            self[key] = item

    def __inc__(self, key, value, *args, **kwargs):
        if key.startswith('edge_index'):
            return self.num_nodes
        else:
            return super(SceneData, self).__inc__(key, value)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key.startswith('edge_index'):
            return 1
        else:
            return super(SceneData, self).__cat_dim__(key, value, *args, **kwargs)

class SceneData_het(Data):
    def __init__(self, num_nodes=None, **kwargs):
        super(SceneData_het, self).__init__()
        self.num_nodes = num_nodes
        for key, item in kwargs.items():
            self[key] = item

    def __inc__(self, key, value, *args, **kwargs):
        if key.startswith('edge_index') or key == 'edge_features':
            # Handle both edge indices and edge features similarly
            # For 'edge_index', this ensures indices are incremented to point to the correct nodes
            # For 'edge_features', this could be used to manage feature index increments if applicable
            return self.num_nodes
        else:
            return super(SceneData_het, self).__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key.startswith('edge_index'):
            # Edge indices should be concatenated along the second dimension
            return 1
        elif key == 'edge_features':
            # Decide on the concatenation dimension for edge features
            # Typically, this might be 0 for stacking features from different graphs
            # Adjust this based on how your edge features are structured
            return 0
        else:
            return super(SceneData_het, self).__cat_dim__(key, value, *args, **kwargs)


