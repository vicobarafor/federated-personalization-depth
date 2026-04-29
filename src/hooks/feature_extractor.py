
class FeatureExtractor:
    def __init__(self, model, layers):
        self.model = model
        self.layers = layers
        self.features = {}
        self.handles = []

        self._register_hooks()

    def _get_hook(self, name):
        def hook(module, input, output):
            self.features[name] = output.detach()
        return hook

    def _register_hooks(self):
        for name, layer in self.layers.items():
            handle = layer.register_forward_hook(self._get_hook(name))
            self.handles.append(handle)

    def clear(self):
        self.features = {}

    def remove(self):
        for handle in self.handles:
            handle.remove()
