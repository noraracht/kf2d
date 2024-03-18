import torch
import torch.nn as nn
import logging

import pandas as pd

# Custom loss function
class Loss(nn.Module):

    def __init__(self):
        super(Loss, self).__init__()

    def my_mse_loss(self, model_dist, true_dist):
        assert model_dist.shape == true_dist.shape
        #print('Train tensor')
        #print(model_dist)
        #print('True tensor')
        #print(true_dist)
        weight = 1 / (true_dist + 1e-6) #** 2   square_root_fm weighting
        #print('Weights')
        #print(weight)
        #print(model_dist - true_dist)

        true_dist = torch.sqrt(true_dist) # need to multiply by 100 and then take sqrt

        #print(torch.max(model_dist - true_dist))
        #print(torch.min(model_dist - true_dist))

        #weight = 1 / (true_dist + 1e-8) ** 2  # square_root_fm weighting
        #loss = ((model_dist - true_dist) ** 2 * weight).mean()
        v = ((model_dist - true_dist) ** 2 * weight)


        # fixed value 10^4

        # pd.DataFrame(model_dist.detach().numpy()).to_csv('model_dist.csv', index=False, header=False, sep=',')
        # pd.DataFrame(true_dist.detach().numpy()).to_csv('true_dist.csv', index=False, header=False, sep=',')

        #print('loss')
        #loss = v.sum()/(torch.numel(v)-torch.numel(torch.diag(v, 0)))  # computes loss without diameter elements
        #loss = v.sum()/model_dist.size(dim=0)

        loss = v

        #print(loss)
        #print((model_dist - true_dist) ** 2 * weight)
        #loss = ((model_dist/(true_dist+ 1e-8)-1) ** 2).mean() #exclude the diameter

        return loss.mean()


    def forward(self, model_dist, true_dist):
        loss = self.my_mse_loss(model_dist, true_dist)

        return loss


class Loss_for_contigs(nn.Module):

    def __init__(self):
        super(Loss_for_contigs, self).__init__()

    def my_mse_loss(self, model_dist, true_dist, ma_dist):
        assert model_dist.shape == true_dist.shape
        #print('Train tensor')
        #print(model_dist)
        #print('True tensor')
        #print(true_dist)

        true_dist = torch.sqrt(true_dist) # need to multiply by 100 and then take sqrt

        mask_within_genome = true_dist==0.0
        #mask_within_genome.fill_diagonal_(False)

        mask_between_genomes = ~mask_within_genome

        A = 0.0  # fixed value between 1-10^4
        weight = 1 / (true_dist + 1e-6) * mask_between_genomes + A /(ma_dist + 1e-6) * mask_within_genome  # ** 2   square_root_fm weighting
        # weight = 1 / (true_dist + 1e-6)
        # print('Weights')
        # print(weight)
        # print(model_dist - true_dist)


        #print(torch.max(model_dist - true_dist))
        #print(torch.min(model_dist - true_dist))

        #weight = 1 / (true_dist + 1e-8) ** 2  # square_root_fm weighting
        #loss = ((model_dist - true_dist) ** 2 * weight).mean()
        #loss = ((model_dist - true_dist) ** 2 * weight).mean() # computes loss with diameter elements
        loss = ((model_dist - true_dist) ** 2 * weight)

        lt = torch.Tensor.float(loss*mask_between_genomes).mean()
        rt = torch.Tensor.float(loss*mask_within_genome).mean()


        #lt_mean = lt.sum() / (lt != 0).sum()

        #logging.info('Loss left: {:.20f}, right: {:.20f}, total: {:.20f}'.format(lt, rt, loss.mean()))


        # pd.DataFrame(model_dist.detach().numpy()).to_csv('model_dist.csv', index=False, header=False, sep=',')
        # pd.DataFrame(true_dist.detach().numpy()).to_csv('true_dist.csv', index=False, header=False, sep=',')

        #print('loss')
        #loss = v.sum()/(torch.numel(v)-torch.numel(torch.diag(v, 0)))  # computes loss without diameter elements


        #loss = v.sum()/model_dist.size(dim=0)

        #print(loss)
        #print((model_dist - true_dist) ** 2 * weight)
        #loss = ((model_dist/(true_dist+ 1e-8)-1) ** 2).mean() #exclude the diameter

        return loss.mean()

    def forward(self, model_dist, true_dist, ma_dist):
        loss = self.my_mse_loss(model_dist, true_dist, ma_dist)

        return loss

class Loss_wlambda(nn.Module):

    def __init__(self):
        super(Loss_wlambda, self).__init__()

        # self.lambda_i = torch.nn.Parameter(torch.ones(n_dims, dtype=torch.float32, requires_grad=True, device=device))
        # self.lambda_j = torch.nn.Parameter(torch.ones(n_dims, dtype=torch.float32, requires_grad=True, device=device))
        #self.a_loss_const = a

    def my_mse_loss(self, model_dist, true_dist, lambda_i_reduced):
        assert model_dist.shape == true_dist.shape

        # Define constant A as batch size
        #a_loss_const = 1.0/lambda_i_reduced.size(dim=0) # labels is type tensor and a_loss_const is type int
        # print(a_loss_const)

        # Compute loss

        # print('Train tensor')
        # print(model_dist)
        # print('True tensor')
        # print(true_dist)
        weight = 1 / (true_dist + 1e-6)  # ** 2   square_root_fm weighting
        # print('Weights')
        # print(weight)
        # print(model_dist - true_dist)

        true_dist = torch.sqrt(true_dist)  # need to multiply by 100 and then take sqrt

        # print(torch.max(model_dist - true_dist))
        # print(torch.min(model_dist - true_dist))

        # weight = 1 / (true_dist + 1e-8) ** 2  # square_root_fm weighting
        # loss = ((model_dist - true_dist) ** 2 * weight).mean()
        v = ((model_dist - true_dist) ** 2 * weight)
        s = (lambda_i_reduced * v).sum(dim=1)
        left_t = (lambda_i_reduced * s).sum(dim=0)

        #right_t = ((torch.nn.functional.logsigmoid(lambda_i_reduced))).sum()

        # b=((torch.log(lambda_i_reduced))**2).sum()
        # print(torch.mul(b, a_loss_const))

        print("Left term: {}".format(left_t))
        #print("Right term: {}".format(right_t))

        loss = left_t
        #loss = left_t - self.a_loss_const * right_t

        #print("A constant: {}".format(self.a_loss_const))
        print("Loss: {}".format(loss))

        # print('loss')
        # loss = v.sum()/(torch.numel(v)-torch.numel(torch.diag(v, 0)))  # computes loss without diameter elements

        loss = loss/(torch.numel(v)-torch.numel(torch.diag(v, 0)))
        print("Loss mean reduced: {}".format(loss))
        print()
        #loss = loss / model_dist.size(dim=0)

        # print(loss)
        # print((model_dist - true_dist) ** 2 * weight)
        # loss = ((model_dist/(true_dist+ 1e-8)-1) ** 2).mean() #exclude the diameter

        return loss

    def forward(self, model_dist, true_dist, lambda_i_reduced):
        loss = self.my_mse_loss(model_dist, true_dist, lambda_i_reduced)

        return loss





# Custom loss function with lambda
# -----------------------------------
# class Loss_wlambda(nn.Module):
#
#     def __init__(self, a):
#         super(Loss_wlambda, self).__init__()
#
#         # self.lambda_i = torch.nn.Parameter(torch.ones(n_dims, dtype=torch.float32, requires_grad=True, device=device))
#         # self.lambda_j = torch.nn.Parameter(torch.ones(n_dims, dtype=torch.float32, requires_grad=True, device=device))
#         self.a_loss_const = a
#
#     def my_mse_loss(self, model_dist, true_dist, lambda_i_reduced):
#         assert model_dist.shape == true_dist.shape
#
#         # Define constant A as batch size
#         #a_loss_const = 1.0/lambda_i_reduced.size(dim=0) # labels is type tensor and a_loss_const is type int
#         # print(a_loss_const)
#
#         # Compute loss
#
#         # print('Train tensor')
#         # print(model_dist)
#         # print('True tensor')
#         # print(true_dist)
#         weight = 1 / (true_dist + 1e-6)  # ** 2   square_root_fm weighting
#         # print('Weights')
#         # print(weight)
#         # print(model_dist - true_dist)
#
#         true_dist = torch.sqrt(true_dist)  # need to multiply by 100 and then take sqrt
#
#         # print(torch.max(model_dist - true_dist))
#         # print(torch.min(model_dist - true_dist))
#
#         # weight = 1 / (true_dist + 1e-8) ** 2  # square_root_fm weighting
#         # loss = ((model_dist - true_dist) ** 2 * weight).mean()
#         v = ((model_dist - true_dist) ** 2 * weight)
#         s = (torch.sigmoid(lambda_i_reduced) * v).sum(dim=1)
#         left_t = (torch.sigmoid(lambda_i_reduced) * s).sum(dim=0)
#         right_t = ((torch.nn.functional.logsigmoid(lambda_i_reduced))).sum()
#
#         # b=((torch.log(lambda_i_reduced))**2).sum()
#         # print(torch.mul(b, a_loss_const))
#
#         print("Left term: {}".format(left_t))
#         print("Right term: {}".format(right_t))
#
#         loss = left_t - self.a_loss_const * right_t
#
#         #print("A constant: {}".format(self.a_loss_const))
#         print("Loss: {}".format(loss))
#
#         # print('loss')
#         # loss = v.sum()/(torch.numel(v)-torch.numel(torch.diag(v, 0)))  # computes loss without diameter elements
#
#         loss = loss/(torch.numel(v)-torch.numel(torch.diag(v, 0)))
#         print("Loss mean reduced: {}".format(loss))
#         print()
#         #loss = loss / model_dist.size(dim=0)
#
#         # print(loss)
#         # print((model_dist - true_dist) ** 2 * weight)
#         # loss = ((model_dist/(true_dist+ 1e-8)-1) ** 2).mean() #exclude the diameter
#
#         return loss
#
#     def forward(self, model_dist, true_dist, lambda_i_reduced):
#         loss = self.my_mse_loss(model_dist, true_dist, lambda_i_reduced)
#
#         return loss




# # Custom loss function with lambda
# # -----------------------------------
# class Loss_wlambda(nn.Module):
#
#     def __init__(self):
#         super(Loss_wlambda, self).__init__()
#
#         # self.lambda_i = torch.nn.Parameter(torch.ones(n_dims, dtype=torch.float32, requires_grad=True, device=device))
#         # self.lambda_j = torch.nn.Parameter(torch.ones(n_dims, dtype=torch.float32, requires_grad=True, device=device))
#
#     def my_mse_loss(self, model_dist, true_dist, lambda_i_reduced):
#         assert model_dist.shape == true_dist.shape
#
#         # Define constant A as batch size
#         a_loss_const = lambda_i_reduced.size(dim=0) # labels is type tensor and a_loss_const is type int
#         # print(a_loss_const)
#
#         # Compute loss
#
#         # print('Train tensor')
#         # print(model_dist)
#         # print('True tensor')
#         # print(true_dist)
#         weight = 1 / (true_dist + 1e-6)  # ** 2   square_root_fm weighting
#         # print('Weights')
#         # print(weight)
#         # print(model_dist - true_dist)
#
#         true_dist = torch.sqrt(true_dist)  # need to multiply by 100 and then take sqrt
#
#         # print(torch.max(model_dist - true_dist))
#         # print(torch.min(model_dist - true_dist))
#
#         # weight = 1 / (true_dist + 1e-8) ** 2  # square_root_fm weighting
#         # loss = ((model_dist - true_dist) ** 2 * weight).mean()
#         v = ((model_dist - true_dist) ** 2 * weight)
#         s = (lambda_i_reduced * v).sum(dim=1)
#         t = (lambda_i_reduced * s).sum(dim=0)
#
#         # b=((torch.log(lambda_i_reduced))**2).sum()
#         # print(torch.mul(b, a_loss_const))
#
#         loss = t + a_loss_const * (((torch.log(lambda_i_reduced)) ** 2).sum())
#
#         # print('loss')
#         # loss = v.sum()/(torch.numel(v)-torch.numel(torch.diag(v, 0)))  # computes loss without diameter elements
#
#         loss = loss/(torch.numel(v)-torch.numel(torch.diag(v, 0)))
#
#         #loss = loss / model_dist.size(dim=0)
#
#         # print(loss)
#         # print((model_dist - true_dist) ** 2 * weight)
#         # loss = ((model_dist/(true_dist+ 1e-8)-1) ** 2).mean() #exclude the diameter
#
#         return loss
#
#     def forward(self, model_dist, true_dist, lambda_i_reduced):
#         loss = self.my_mse_loss(model_dist, true_dist, lambda_i_reduced)
#
#         return loss

# -----------------------------------
# Attempt 3.
# class Loss_wlambda(nn.Module):
#
#     def __init__(self, n_dims, device):
#         super(Loss_wlambda, self).__init__()
#
#         self.lambda_i = torch.nn.Parameter(torch.ones(n_dims, dtype=torch.float32, requires_grad=True, device=device))
#         self.lambda_j = torch.nn.Parameter(torch.ones(n_dims, dtype=torch.float32, requires_grad=True, device=device))
#
#
#     def my_mse_loss(self, model_dist, true_dist, labels, device):
#         assert model_dist.shape == true_dist.shape
#
#
#         # Define constant A as batch size
#         a_loss_const = labels.size(dim=0) # labels is type tensor and a_loss_const is type int
#
#         # Create mask
#         #index_array = labels.type(torch.LongTensor)
#         index_array = labels
#         index_array.to(device)
#
#         n = self.lambda_i.size(dim=0)
#         mask_array = torch.zeros(n, dtype=torch.bool, device=device)
#         mask_array[index_array] = True
#         mask_array.to(device)
#
#         # Create mask based on labels
#         lambda_i_reduced = torch.masked_select(self.lambda_i, mask_array)
#         lambda_j_reduced = torch.masked_select(self.lambda_j, mask_array)
#
#         torch.set_printoptions(precision=10)
#         print("Lambda i: {}".format(lambda_i_reduced))
#         print("Lambda j: {}".format(lambda_j_reduced))
#
#         print("Lambda i gradient: {}".format(lambda_i_reduced.grad))
#         print("Lambda j gradient: {}".format(lambda_j_reduced.grad))
#
#
#         # Compute loss
#
#         #print('Train tensor')
#         #print(model_dist)
#         #print('True tensor')
#         #print(true_dist)
#         weight = 1 / (true_dist + 1e-6) #** 2   square_root_fm weighting
#         #print('Weights')
#         #print(weight)
#         #print(model_dist - true_dist)
#
#         true_dist = torch.sqrt(true_dist) # need to multiply by 100 and then take sqrt
#
#         #print(torch.max(model_dist - true_dist))
#         #print(torch.min(model_dist - true_dist))
#
#         #weight = 1 / (true_dist + 1e-8) ** 2  # square_root_fm weighting
#         #loss = ((model_dist - true_dist) ** 2 * weight).mean()
#         v = ((model_dist - true_dist) ** 2 * weight)
#         s = (lambda_j_reduced*v).sum(dim = 1)
#         t = (lambda_i_reduced*s).sum(dim = 0)
#
#         # b=((torch.log(lambda_i_reduced))**2).sum()
#         # print(torch.mul(b, a_loss_const))
#
#         loss = t + a_loss_const*(((torch.log(lambda_i_reduced))**2).sum())
#         #print('loss')
#         #loss = v.sum()/(torch.numel(v)-torch.numel(torch.diag(v, 0)))  # computes loss without diameter elements
#         loss = loss/(torch.numel(v)-torch.numel(torch.diag(v, 0)))
#         #print(loss)
#         #print((model_dist - true_dist) ** 2 * weight)
#         #loss = ((model_dist/(true_dist+ 1e-8)-1) ** 2).mean() #exclude the diameter
#
#         return loss
#
#     def forward(self, model_dist, true_dist, labels, device):
#         loss = self.my_mse_loss(model_dist, true_dist, labels, device)
#
#         return loss
#
#


