import torch
from config import config
from bert import BertModel


def set_learned_params(net, weights_path="./weights/pytorch_model.bin"):
    loaded_state_dict = torch.load(weights_path)

    # set model
    net = BertModel(config)
    # set model as evaluation mode
    net.eval()

    # save parameter names of current model
    param_names = []
    for name, _ in net.named_parameters():
        param_names.append(name)

    # put state dict in order from the front due to different names
    # set new state dict to put
    new_state_dict = net.state_dict().copy()

    # put pre-trained state dict to new one
    for index, (key_name, value) in enumerate(loaded_state_dict.items()):
        # get parameter names of current model
        name = param_names[index]
        # replace new state dict to parameters of pre-trained model
        new_state_dict[name] = value
        # notify information of replacement
        print(str(key_name) + " -> " + str(name))

        # make break in case of out of range of index
        if (index + 1) >= len(param_names):
            break

    # transplant new state dict into current state dict
    net.load_state_dict(new_state_dict)

    return net
