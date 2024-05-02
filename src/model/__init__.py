from .base_gcn import ResGCNModule
from .TPGCN import TPGCN
from .DoubleTPGCN import DoubleTPGCN


__models = {
    'TPGCN': TPGCN,
    'DoubleTPGCN': DoubleTPGCN,
}


# def create_model(A):
#     return TPGCN(ResGCNModule, structure=[1, 2, 3, 3], spatial_block='Basic', temporal_block='Basic', data_shape=(1, 3, 1, 1, 1), num_class=2, A=A)

def create(model_type, **kwargs):
    model_name = model_type.split('-')[0]
    return __models[model_name].create(model_type, **kwargs)
