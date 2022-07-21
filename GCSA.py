from __future__ import print_function
import argparse
import sys

from sklearn.metrics import euclidean_distances

sys.path.append("..")
import os
import random
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import util
import classifier_embed_contras
import model
import losses
import torch.nn.functional as F
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='CUB', help='FLO')
parser.add_argument('--dataroot', default='./data', help='path to dataset')
parser.add_argument('--matdataset', default=True, help='Data in matlab format')
parser.add_argument('--image_embedding', default='res101')
parser.add_argument('--class_embedding', default='sent', help='att or sent')
parser.add_argument('--syn_num', type=int, default=500, help='number features to generate per class')
parser.add_argument('--gzsl', type=bool, default=True, help='enable generalized zero-shot learning')
parser.add_argument('--preprocessing', type=bool, default=True, help='enbale MinMaxScaler on visual features')
parser.add_argument('--standardization', action='store_true', default=False)
parser.add_argument('--validation', action='store_true', default=False, help='enable cross validation mode')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batch_size', type=int, default=2048, help='input batch size')
parser.add_argument('--resSize', type=int, default=2048, help='size of visual features')
parser.add_argument('--attSize', type=int, default=1024, help='size of semantic features')
parser.add_argument('--nz', type=int, default=1024, help='noise for generation')
parser.add_argument('--embedSize', type=int, default=2048, help='size of embedding h')
parser.add_argument('--outzSize', type=int, default=512, help='size of non-liner projection z')

## network architechure
parser.add_argument('--ngh', type=int, default=4096, help='size of the hidden units in generator G')
parser.add_argument('--ndh', type=int, default=4096, help='size of the hidden units in discriminator D')
parser.add_argument('--nhF', type=int, default=2048, help='size of the hidden units comparator network F')

parser.add_argument('--ins_weight_pos', type=float, default=0.001, help='weight of the classification loss when learning G')
parser.add_argument('--ins_weight_neg', type=float, default=0.001, help='weight of the classification loss when learning G')
parser.add_argument('--cls_weight', type=float, default=0.001, help='weight of the score function when learning G')
parser.add_argument('--ins_temp', type=float, default=0.1, help='temperature of visual features')
parser.add_argument('--cls_temp', type=float, default=0.1, help='temperature in class-level supervision')
parser.add_argument('--ins_tempa', type=float, default=40, help='temperature of semantics')
parser.add_argument('--nepoch', type=int, default=1300, help='number of epochs to train for')
parser.add_argument('--critic_iter', type=int, default=5, help='critic iteration, following WGAN-GP')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate to training')
parser.add_argument('--lr_decay_epoch', type=int, default=100,
                    help='conduct learning rate decay after every 100 epochs')
parser.add_argument('--lr_dec_rate', type=float, default=0.99, help='learning rate decay rate')
parser.add_argument('--lambda1', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
parser.add_argument('--classifier_lr', type=float, default=0.001, help='learning rate to train softmax classifier')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
parser.add_argument('--manualSeed', type=int, default=3483, help='manual seed')
parser.add_argument('--nclass_all', type=int, default=200, help='number of all classes')
parser.add_argument('--nclass_seen', type=int, default=150, help='number of all classes')
parser.add_argument('--use_classify', type=bool, default=False, help='use classify or not')
parser.add_argument('--gpus', default='0', help='the number of the GPU to use')
parser.add_argument('--k', type=int, default=30, help='k for knn')
parser.add_argument('--reg', type=float, default=1.0, help="the weight of regression loss")
parser.add_argument('--mask', type=int, default=1, help="negative sample mask")
opt = parser.parse_args()
print(opt)
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# load data
data = util.DATA_LOADER(opt)
print("# of training samples: ", data.ntrain)

netG = model.MLP_G(opt)
netMap = model.Embedding_Net(opt)
netD = model.MLP_CRITIC(opt)
F_ha = model.Dis_Embed_Att(opt)

RNet1 = model.RNet1(opt)

model_path = './models/' + opt.dataset
if not os.path.exists(model_path):
    os.makedirs(model_path)

if len(opt.gpus.split(',')) > 1:
    netG = nn.DataParallel(netG)
    netD = nn.DataParallel(netD)
    netMap = nn.DataParallel(netMap)
    F_ha = nn.DataParallel(F_ha)


if(opt.mask==1):
    contras_criterion= losses.SupConLoss_clear_new(opt)
else:
    contras_criterion = losses.SupConLoss_clear_mask0(opt)

input_res = torch.FloatTensor(opt.batch_size, opt.resSize)
input_att = torch.FloatTensor(opt.batch_size, opt.attSize)
noise_gen = torch.FloatTensor(opt.batch_size, opt.nz)
input_label = torch.LongTensor(opt.batch_size)


if opt.cuda:
    netG.cuda()
    netD.cuda()
    netMap.cuda()
    F_ha.cuda()
    # F_aa.cuda()
    RNet1.cuda()
    input_res = input_res.cuda()
    noise_gen, input_att = noise_gen.cuda(), input_att.cuda()
    input_label = input_label.cuda()

    # derta.cuda()

def sample():
    batch_feature, batch_label, batch_att = data.next_batch(opt.batch_size)
    input_res.copy_(batch_feature)
    input_att.copy_(batch_att)
    input_label.copy_(util.map_label(batch_label, data.seenclasses))


def generate_syn_feature(netG, classes, attribute, num):
    nclass = classes.size(0)
    syn_feature = torch.FloatTensor(nclass * num, opt.resSize)
    syn_label = torch.LongTensor(nclass * num)
    syn_att = torch.FloatTensor(num, opt.attSize)
    syn_noise = torch.FloatTensor(num, opt.nz)
    if opt.cuda:
        syn_att = syn_att.cuda()
        syn_noise = syn_noise.cuda()

    for i in range(nclass):
        iclass = classes[i]
        iclass_att = attribute[iclass]
        syn_att.copy_(iclass_att.repeat(num, 1))
        syn_noise.normal_(0, 1)
        with torch.no_grad():
            output = netG(syn_noise, syn_att)
        syn_feature.narrow(0, i * num, num).copy_(output.data.cpu())
        syn_label.narrow(0, i * num, num).fill_(iclass)
    return syn_feature, syn_label


# setup optimizer
import itertools
optimizerD = optim.Adam(itertools.chain(netD.parameters(), netMap.parameters(), F_ha.parameters()), lr=opt.lr,
                        betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(),lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerR = optim.Adam(RNet1.parameters(), lr=0.0001, betas=(opt.beta1, 0.999))

def calc_gradient_penalty(netD, real_data, fake_data, input_att):
    alpha = torch.rand(opt.batch_size, 1)
    alpha = alpha.expand(real_data.size())
    if opt.cuda:
        alpha = alpha.cuda()
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    if opt.cuda:
        interpolates = interpolates.cuda()
    interpolates = Variable(interpolates, requires_grad=True)
    disc_interpolates = netD(interpolates, input_att)
    ones = torch.ones(disc_interpolates.size())
    if opt.cuda:
        ones = ones.cuda()
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[
        0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lambda1
    return gradient_penalty



# use the for-loop to save the GPU-memory
def class_scores_for_loop(embed, input_label, relation_net):
    all_scores = torch.FloatTensor(embed.shape[0], opt.nclass_seen).cuda()
    for i, i_embed in enumerate(embed):
        expand_embed = i_embed.repeat(opt.nclass_seen,
                                      1)
        all_scores[i] = (torch.div(relation_net(torch.cat((expand_embed, data.attribute_seen.cuda()), dim=1)),
                                   opt.cls_temp).squeeze())
    score_max, _ = torch.max(all_scores, dim=1, keepdim=True)
    scores_norm = all_scores - score_max.detach()
    mask = F.one_hot(input_label, num_classes=opt.nclass_seen).float().cuda()
    exp_scores = torch.exp(scores_norm)
    log_scores = scores_norm - torch.log(exp_scores.sum(1, keepdim=True))
    cls_loss = -((mask * log_scores).sum(1) / mask.sum(1)).mean()
    return cls_loss


# It is much faster to use the matrix, but it cost much GPU memory.
def class_scores_in_matrix(embed, input_label, relation_net):
    expand_embed = embed.unsqueeze(dim=1).repeat(1, opt.nclass_seen, 1).reshape(embed.shape[0] * opt.nclass_seen, -1)
    expand_att = data.attribute_seen.unsqueeze(dim=0).repeat(embed.shape[0], 1, 1).reshape(
        embed.shape[0] * opt.nclass_seen, -1).cuda()
    all_scores = torch.div(relation_net(torch.cat((expand_embed, expand_att), dim=1)), opt.cls_temp).reshape(
        embed.shape[0],
        opt.nclass_seen)
    score_max, _ = torch.max(all_scores, dim=1, keepdim=True)
    scores_norm = all_scores - score_max.detach()
    mask = F.one_hot(input_label, num_classes=opt.nclass_seen).float().cuda()
    exp_scores = torch.exp(scores_norm)
    log_scores = scores_norm - torch.log(exp_scores.sum(1, keepdim=True))
    cls_loss = -((mask * log_scores).sum(1) / mask.sum(1)).mean()
    return cls_loss

def KNNPredict(X_train, y_train, X_test, k=5):
    sim = -1 * euclidean_distances(X_test.cpu().data.numpy(), X_train.cpu().data.numpy())
    idx_mat = np.argsort(-1 * sim, axis=1)[:, 0: k]
    preds = np.array([np.argmax(np.bincount(item)) for item in y_train[idx_mat]])
    return preds

def compute_per_class_acc_gzsl(test_label, predicted_label, target_classes):
    acc_per_class = 0
    for i in target_classes:
        idx = (test_label == i)
        if torch.sum(idx) == 0:
            acc_per_class += 0
        else:
            acc_per_class += float(torch.sum(test_label[idx] == predicted_label[idx])) / float(torch.sum(idx))# 第i个类的准确率
    acc_per_class /= float(target_classes.size(0))
    return acc_per_class

for epoch in range(opt.nepoch):  # 2000
    FP = 0
    mean_lossD = 0
    mean_lossG = 0
    for i in range(0, data.ntrain, opt.batch_size):
        ############################
        # (1) Update D network: optimize WGAN-GP objective, Equation (2):(1)更新D（判别器）网络:优化WGAN-GP目标，式(2)
        ###########################
        for p in netD.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to False below in netG update
        for p in netMap.parameters():  # reset requires_grad
            p.requires_grad = True
        for p in F_ha.parameters():  # reset requires_grad
            p.requires_grad = True

        for iter_d in range(opt.critic_iter):
            sample()
            netD.zero_grad()
            netMap.zero_grad()
            #
            # train with realG
            # sample a mini-batch
            sparse_real = opt.resSize - input_res[1].gt(0).sum()
            embed_real, outz_real = netMap(input_res)
            criticD_real = netD(input_res, input_att)
            criticD_real = criticD_real.mean()

            # CONTRASITVE LOSS
            real_ins_contras_loss_pos,real_ins_contras_loss_neg = contras_criterion(outz_real, input_att, input_label)

            # train with fakeG
            noise_gen.normal_(0, 1)
            fake = netG(noise_gen, input_att)  # x~ = G(att,noise)
            fake_norm = fake.data[0].norm()
            sparse_fake = fake.data[0].eq(0).sum()
            criticD_fake = netD(fake.detach(), input_att)
            criticD_fake = criticD_fake.mean()

            # gradient penalty
            gradient_penalty = calc_gradient_penalty(netD, input_res, fake.data, input_att)
            Wasserstein_D = criticD_real - criticD_fake

            cls_loss_real = class_scores_in_matrix(embed_real, input_label, F_ha)
            # cls_loss_real = class_scores_for_loop(embed_real, input_label, F_ha)
            D_cost = criticD_fake - criticD_real + gradient_penalty + real_ins_contras_loss_pos  + real_ins_contras_loss_neg + cls_loss_real

            D_cost.backward()
            optimizerD.step()
        ############################
        # (2) Update G（生成器） network: optimize WGAN-GP objective, Equation (2)
        ###########################
        for p in netD.parameters():  # reset requires_grad
            p.requires_grad = False  # avoid computation
        for p in netMap.parameters():  # reset requires_grad
            p.requires_grad = False
        for p in F_ha.parameters():  # reset requires_grad
            p.requires_grad = False


        netG.zero_grad()
        RNet1.zero_grad()
        noise_gen.normal_(0, 1)
        fake = netG(noise_gen,
                    input_att)

        embed_fake, outz_fake = netMap(fake)
        a_g = RNet1(embed_fake)

        criticG_fake = netD(fake, input_att)
        criticG_fake = criticG_fake.mean()
        G_cost = -criticG_fake

        embed_real, outz_real = netMap(input_res)
        a_r = RNet1(embed_real)
        input_attv = Variable(input_att)


        R_loss = Variable(torch.Tensor([0.0])).cuda()
        countn = 0
        for i in range(data.ntrain_class):
            sample_idx = (input_label == i).data.nonzero().squeeze()
            if sample_idx.numel() == 0:

                R_loss += 0
            else:
                countn = countn + 1
                G_sample_cls = fake[sample_idx, :]


                ag_cls = a_g[sample_idx, :]
                ar_cls = a_r[sample_idx, :]
                a_cls = input_attv[sample_idx, :]
                # R_loss += torch.sqrt((ar_cls.mean(dim=0) - a_cls.mean(dim=0)).pow(
                #     2).sum() + 1e-10) + torch.sqrt((ag_cls.mean(dim=0) - a_cls.mean(dim=0)).pow(
                #     2).sum() + 1e-10) #SUN

                R_loss += (ar_cls.mean(dim=0) - a_cls.mean(dim=0)).pow(
                    2).sum().sqrt() + (ag_cls.mean(dim=0) - a_cls.mean(dim=0)).pow(
                    2).sum().sqrt()

        R_loss = R_loss / countn

        all_outz = torch.cat((outz_fake, outz_real.detach()), dim=0)

        fake_ins_contras_loss_pos, fake_ins_contras_loss_neg= contras_criterion(all_outz,torch.cat((input_att,input_att), dim=0),
                                                  torch.cat((input_label, input_label), dim=0))

        cls_loss_fake = class_scores_in_matrix(embed_fake, input_label, F_ha)
        # cls_loss_fake = class_scores_for_loop(embed_fake, input_label, F_ha)

        errG =  G_cost + opt.ins_weight_pos *fake_ins_contras_loss_pos + opt.ins_weight_neg*fake_ins_contras_loss_neg + opt.cls_weight * cls_loss_fake  + R_loss * opt.reg
        errG.backward()
        optimizerG.step()
        optimizerR.step()# delete for CUB

    F_ha.zero_grad()
    # F_aa.zero_grad()
    if (epoch + 1) % opt.lr_decay_epoch == 0:
        for param_group in optimizerD.param_groups:
            param_group['lr'] = param_group['lr'] * opt.lr_dec_rate
        for param_group in optimizerG.param_groups:
            param_group['lr'] = param_group['lr'] * opt.lr_dec_rate

    mean_lossG /= data.ntrain / opt.batch_size  # 无用
    mean_lossD /= data.ntrain / opt.batch_size  # 无用
    print(
        '[%d/%d] Loss_D: %.4f Loss_G: %.4f, Wasserstein_dist: %.4f, real_ins_contras_loss_pos:%.4f, real_ins_contras_loss_neg:%.4f, fake_ins_contras_loss_pos:%.4f, fake_ins_contras_loss_neg:%.4f, cls_loss_real: %.4f, cls_loss_fake: %.4f, R_loss:%.4f'
        % (
        epoch, opt.nepoch, D_cost, G_cost, Wasserstein_D, real_ins_contras_loss_pos ,  real_ins_contras_loss_neg, fake_ins_contras_loss_pos, fake_ins_contras_loss_neg,cls_loss_real,cls_loss_fake,R_loss.item()))

    # evaluate the model, set G to evaluation mode
    netG.eval()

    for p in netMap.parameters():
        p.requires_grad = False

    if opt.gzsl:  # Generalized zero-shot learning

        syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data.attribute, opt.syn_num)
        train_X = torch.cat((data.train_feature, syn_feature), 0)
        train_Y = torch.cat((data.train_label, syn_label), 0)
        train_z = train_X.cuda()
        test_z_seen = data.test_seen_feature.cuda()
        nclass = opt.nclass_all
        if opt.use_classify == True:
            cls = classifier_embed_contras.CLASSIFIER(train_X, train_Y, netMap, opt.embedSize, data, nclass, opt.cuda,
                                                      opt.classifier_lr, 0.5, 25, opt.syn_num,
                                                      True)
            print('unseen=%.4f, seen=%.4f, h=%.4f' % (cls.acc_unseen, cls.acc_seen, cls.H))

        else:
            train_z, _ = netMap(train_z)
            test_z_seen, _ = netMap(test_z_seen)
            test_z_unseen = data.test_unseen_feature.cuda()
            test_z_unseen, _ = netMap(test_z_unseen)
            pred_Y_s = torch.from_numpy(KNNPredict(train_z, train_Y, test_z_seen, k=opt.k))
            pred_Y_u = torch.from_numpy(KNNPredict(train_z, train_Y, test_z_unseen, k=opt.k))
            acc_seen = compute_per_class_acc_gzsl(data.test_seen_label, pred_Y_s, data.seenclasses)
            acc_unseen = compute_per_class_acc_gzsl(data.test_unseen_label, pred_Y_u, data.unseenclasses)
            H = 2 * acc_seen * acc_unseen / (acc_seen + acc_unseen)
            print('unseen=%.4f, seen=%.4f, h=%.4f' % (acc_unseen, acc_seen, H))

    else:  # conventional zero-shot learning
        syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data.attribute, opt.syn_num)
        cls = classifier_embed_contras.CLASSIFIER(syn_feature, util.map_label(syn_label, data.unseenclasses), netMap,
                                                  opt.embedSize, data,
                                                  data.unseenclasses.size(0), opt.cuda, opt.classifier_lr, 0.5, 100,
                                                  opt.syn_num,
                                                  False)
        acc = cls.acc
        print('unseen class accuracy=%.4f ' % acc)

    # reset G to training mode
    netG.train()
    for p in netMap.parameters():  # reset requires_grad
        p.requires_grad = True
