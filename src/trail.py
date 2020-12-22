import sys
import torch
sys.path.append('/home/wanghongwei/WorkSpace/source/tools/pytorchviz/')
from torchvision.models import AlexNet
from torchviz import make_dot_from_trace, make_dot
model = AlexNet()
x = torch.randn(1, 3, 227, 227).requires_grad_(True)

with torch.onnx.set_training(model, False):
    trace, _ = torch.jit.get_trace_graph(model, args=(x,))
make_dot_from_trace(trace)


# y = model(x)
# make_dot(y, params=dict(list(model.named_parameters()) + [('x', x)]))
