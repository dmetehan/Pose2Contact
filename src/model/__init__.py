from .base_gcn import TPGCN, ResGCNModule


def create_model(A):
    return TPGCN(ResGCNModule, structure=[1, 2, 3, 3], spatial_block='Basic', temporal_block='Basic', data_shape=(1, 3, 1, 1, 1), num_class=2, A=A)


# __models = {
#     '2PGCN': TPGCN,
# }


# def create(model_type, **kwargs):
#     model_name = model_type.split('-')[0]
#     return __models[model_name].create(model_type, **kwargs)
