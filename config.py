import numpy as np
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, TensorSpec

HYPERPARAMETERS = {
    "model_aggregation": ["SAGEConv"],
    "model_edge_dim": [1],
    "batch_size": [1],
    "model_concat": [False],
    "learning_rate": [0.1, 0.05, 0.01, 0.001],
    "weight_decay": [0.0001, 0.00001, 0.001],
    "sgd_momentum": [0.9, 0.8, 0.5],
    "scheduler_gamma": [0.995, 0.9, 0.8, 0.5, 1],
    "model_out_features": [8, 16],
    "model_normalize": [False]
}

input_schema = Schema([TensorSpec(np.dtype(np.float32), (-1, 30), name="x"),
                       TensorSpec(np.dtype(np.float32), (-1, 11), name="edge_attr"),
                       TensorSpec(np.dtype(np.int32), (2, -1), name="edge_index")])

output_schema = Schema([TensorSpec(np.dtype(np.float32), (-1, 1))])

SIGNATURE = ModelSignature(inputs=input_schema, outputs=output_schema)

