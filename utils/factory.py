
def get_model(model_name, args):
    name = model_name.lower()
    if name == "icarl":
        from models.icarl import iCaRL
        return iCaRL(args)
    elif name == "bic":
        from models.bic import BiC
        return BiC(args)
    elif name == "replay":
        from models.replay import Replay
        return Replay(args)
    elif name == "foster":
        from models.foster import FOSTER
        return FOSTER(args)
    elif name =="kindroid":
        from models.kindroid import KINDROIDNet
        return KINDROIDNet(args)
    else:
        assert 0
