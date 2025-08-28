from .models import radarnet, radarnet_v2


def get_model(arch, expansion=2):

    if arch == "simple":
        model = radarnet.SimpleNet()
    elif arch == "DevModel1":
        model = radarnet.DevModel1(expansion=expansion)
    elif arch == "DevModel2":
        model = radarnet.DevModel2(expansion=expansion)
    elif arch == "DevModel3":
        model = radarnet.DevModel3(expansion=expansion)
    elif arch == "DevModel6":
        model = radarnet.DevModel6(expansion=expansion)
    elif arch == "DevModel7":
        model = radarnet.DevModel7(expansion=expansion)
    elif arch == "DevModel8":
        model = radarnet.DevModel7(expansion=expansion)
    elif arch == "DevModel9":
        model = radarnet_v2.DevModel9(expansion=expansion)
    elif arch == "DevModel10":
        model = radarnet_v2.DevModel10(expansion=expansion)

    return model
