import torch
import torch.nn as nn
# import torch.nn.init as init

def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)

    Note: .data shouldn’t be used anymore
    Using the inplace init methods directly passing the parameters
    '''

    if isinstance(m, nn.Linear):             # Check instance of m
        #nn.init.xavier_normal_(m.weight)    # Use inplace initialization
        #nn.init.xavier_uniform_(m.weight, gain=0.8)    # Use inplace initialization
        #nn.init.normal_(m.weight, mean=1, std=0.02)
        nn.init.uniform_(m.weight, 0.0, 0.001)

        if m.bias is not None:               # Make sure bias exists
            #init.normal_(m.bias)            # Set the bias to zeros, if it’s available
            nn.init.constant_(m.bias, 0)     # Set the bias to zeros, if it’s available
            #nn.init.zero_(m.bias)








# # takes in a module and applies the specified weight initialization
# def weights_init_uniform_rule(m):
#     classname = m.__class__.__name__
#     # for every Linear layer in a model..
#     if classname.find('Linear') != -1:
#         # get the number of the inputs
#         n = m.in_features
#         y = 1.0/np.sqrt(n)
#         m.weight.data.uniform_(-y, y)
#         m.bias.data.fill_(0)
#
#
#
#
#
# # https://github.com/pytorch/examples/blob/master/dcgan/main.py#L95-L102
# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         m.weight.data.normal_(0.0, 0.02)
#     elif classname.find('BatchNorm') != -1:
#         m.weight.data.normal_(1.0, 0.02)
#         m.bias.data.fill_(0)
#
#
# # https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py#L46-L59
# def _initialize_weights(self):
#     for m in self.modules():
#         if isinstance(m, nn.Conv2d):
#             n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#             m.weight.data.normal_(0, math.sqrt(2. / n))
#             if m.bias is not None:
#                 m.bias.data.zero_()
#         elif isinstance(m, nn.BatchNorm2d):
#             m.weight.data.fill_(1)
#             m.bias.data.zero_()
#         elif isinstance(m, nn.Linear):
#             m.weight.data.normal_(0, 0.01)
#             m.bias.data.zero_()
#
#
#
#
#
# def weight_init(m):
#     '''
#     Usage:
#         model = Model()
#         model.apply(weight_init)
#     '''
#     if isinstance(m, nn.Conv1d):
#         init.normal_(m.weight.data)
#         if m.bias is not None:
#             init.normal_(m.bias.data)
#     elif isinstance(m, nn.Conv2d):
#         init.xavier_normal_(m.weight.data)
#         if m.bias is not None:
#             init.normal_(m.bias.data)
#     elif isinstance(m, nn.Conv3d):
#         init.xavier_normal_(m.weight.data)
#         if m.bias is not None:
#             init.normal_(m.bias.data)
#     elif isinstance(m, nn.ConvTranspose1d):
#         init.normal_(m.weight.data)
#         if m.bias is not None:
#             init.normal_(m.bias.data)
#     elif isinstance(m, nn.ConvTranspose2d):
#         init.xavier_normal_(m.weight.data)
#         if m.bias is not None:
#             init.normal_(m.bias.data)
#     elif isinstance(m, nn.ConvTranspose3d):
#         init.xavier_normal_(m.weight.data)
#         if m.bias is not None:
#             init.normal_(m.bias.data)
#     elif isinstance(m, nn.BatchNorm1d):
#         init.normal_(m.weight.data, mean=1, std=0.02)
#         init.constant_(m.bias.data, 0)
#     elif isinstance(m, nn.BatchNorm2d):
#         init.normal_(m.weight.data, mean=1, std=0.02)
#         init.constant_(m.bias.data, 0)
#     elif isinstance(m, nn.BatchNorm3d):
#         init.normal_(m.weight.data, mean=1, std=0.02)
#         init.constant_(m.bias.data, 0)
#     elif isinstance(m, nn.Linear):
#         init.xavier_normal_(m.weight.data)
#         init.normal_(m.bias.data)
#     elif isinstance(m, nn.LSTM):
#         for param in m.parameters():
#             if len(param.shape) >= 2:
#                 init.orthogonal_(param.data)
#             else:
#                 init.normal_(param.data)
#     elif isinstance(m, nn.LSTMCell):
#         for param in m.parameters():
#             if len(param.shape) >= 2:
#                 init.orthogonal_(param.data)
#             else:
#                 init.normal_(param.data)
#     elif isinstance(m, nn.GRU):
#         for param in m.parameters():
#             if len(param.shape) >= 2:
#                 init.orthogonal_(param.data)
#             else:
#                 init.normal_(param.data)
#     elif isinstance(m, nn.GRUCell):
#         for param in m.parameters():
#             if len(param.shape) >= 2:
#                 init.orthogonal_(param.data)
#             else:
#                 init.normal_(param.data)


# def initialize_weights(m):
#   if isinstance(m, nn.Conv2d):
#       nn.init.kaiming_uniform_(m.weight.data,nonlinearity='relu')
#       if m.bias is not None:
#           nn.init.constant_(m.bias.data, 0)
#   elif isinstance(m, nn.BatchNorm2d):
#       nn.init.constant_(m.weight.data, 1)
#       nn.init.constant_(m.bias.data, 0)
#   elif isinstance(m, nn.Linear):
#       nn.init.kaiming_uniform_(m.weight.data)
#       nn.init.constant_(m.bias.data, 0)



# https://gist.github.com/jeasinema/ed9236ce743c8efaf30fa2ff732749f5
# https://gist.github.com/jojonki/be1e8af97dfa12c983446391c3640b68
# https://github.com/udacity/deep-learning-v2-pytorch/blob/master/weight-initialization/weight_initialization_solution.ipynb
# https://discuss.pytorch.org/t/how-are-layer-weights-and-biases-initialized-by-default/13073/9
# https://androidkt.com/initialize-weight-bias-pytorch/

# Initialization for a specific layer using .Parameter:
# https://stackoverflow.com/questions/59467473/create-a-new-model-in-pytorch-with-custom-initial-value-for-the-weights
