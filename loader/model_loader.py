import settings
import torch
import torchvision
from models import resnet
from models.base import BasicBlock
import torch.nn as nn
def loadmodel(hook_fn):

    if settings.MODEL == 'resnet26':
        num_classes = [1000]
        layer_config = [4, 4, 4]

        rnet = resnet.FlatResNet26(BasicBlock, layer_config, num_classes)

        source = settings.MODEL_FILE
            # load pretrained weights into flat ResNet
        print source
        checkpoint = load_weights_to_flatresnet(source, rnet, num_classes, 'submit')
        print checkpoint
        for name, m in checkpoint.named_modules():
            print name
        #checkpoint = torch.load(settings.MODEL_FILE)
#         if type(checkpoint).__name__ == 'OrderedDict' or type(checkpoint).__name__ == 'dict':
#             model = torchvision.models.__dict__[settings.MODEL](num_classes=settings.NUM_CLASSES)
#             if settings.MODEL_PARALLEL:
#                 state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint[
#                     'state_dict'].items()}  # the data parallel layer will add 'module' before each layer name
#             else:
#                 state_dict = checkpoint
#             model.load_state_dict(state_dict)
#         else:
        model = checkpoint
    for name in settings.FEATURE_NAMES:
        #print name
        #print model._modules.get('conv1')._modules.get('depthwise')[0].register_forward_hook(hook_fn)
        print model._modules.get('blocks')._modules.get('2')._modules.get('3')._modules.get('SeparableConv2d1')._modules.get('depthwise')[0].register_forward_hook(hook_fn)
        #model._modules.get(name).register_forward_hook(hook_fn)
    if settings.GPU:
        model.cuda()
    model.eval()
    return model

def load_weights_to_flatresnet(source, net, num_classes,  mode):
    checkpoint = torch.load(source)
    net_old = checkpoint['net']

    store_data = []
    for name, m in net_old.named_modules():
        #if isinstance(m, nn.Conv2d) and (m.kernel_size[0]==1):
        if isinstance(m, nn.Conv2d):
            store_data.append(m.weight.data)

    element = 0
    for name, m in net.named_modules():
        #if isinstance(m, nn.Conv2d) and (m.kernel_size[0]==1):
        if isinstance(m, nn.Conv2d):
            m.weight.data = torch.nn.Parameter(store_data[element])
            element += 1

    store_data = []
    store_data_bias = []
    store_data_rm = []
    store_data_rv = []
    for name, m in net_old.named_modules():
        if isinstance(m, nn.BatchNorm2d): # and 'bn' in name:
            store_data.append(m.weight.data)
            store_data_bias.append(m.bias.data)
            store_data_rm.append(m.running_mean)
            store_data_rv.append(m.running_var)

    element = 0
    for name, m in net.named_modules():
        if isinstance(m, nn.BatchNorm2d): # and 'bn' in name:
                m.weight.data = torch.nn.Parameter(store_data[element].clone())
                m.bias.data = torch.nn.Parameter(store_data_bias[element].clone())
                m.running_var = store_data_rv[element].clone()
                m.running_mean = store_data_rm[element].clone()
                element += 1

    if mode == 'submit':
        #if dataset is 'imagenet12' or dataset is 'daimlerpedcls':
        net.fcs[0].weight.data = torch.nn.Parameter(net_old.fcs[0].weight.data)
        net.fcs[0].bias.data = torch.nn.Parameter(net_old.fcs[0].bias.data)
    '''
        else:

            net.fcs[0].weight.data = torch.nn.Parameter(net_old.module.fcs[0].weight.data)
            net.fcs[0].bias.data = torch.nn.Parameter(net_old.module.fcs[0].bias.data)
    '''
    del net_old
    return net
