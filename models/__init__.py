from .p2pnet import build, build_for_pipeline

# build the P2PNet model
# set training to 'True' during training
def build_model(args, training=False):
    return build(args, training)

def build_model_for_pipeline():
    return build_for_pipeline()