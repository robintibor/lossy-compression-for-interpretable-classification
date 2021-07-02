def get_all_activations(net, X,):
    activations = []
    def append_activations(module, input, output):
        activations.append(output)
    handles = []
    for module in net.modules():
        handle = module.register_forward_hook(append_activations)
        handles.append(handle)
    try:
        _ = net(X)
    finally:
        for h in handles:
            h.remove()
    return activations
