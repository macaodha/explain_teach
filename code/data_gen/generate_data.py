from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import im_folder
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
import os
from scipy.ndimage import zoom
import torchvision
from scipy.spatial.distance import squareform, pdist
from sklearn.preprocessing import normalize
from scipy.ndimage.filters import gaussian_filter
import custom_augs
import cv2


def gen_mask(ip_mask, im, blur=False):
    # assume im in [0,1] and RGB
    mask = ip_mask.copy()
    mask -= np.min(mask)
    mask /= np.max(mask)

    mask = np.exp(3*mask)  # scale it
    mask /= np.max(mask)

    if blur:
        mask = gaussian_filter(mask.copy(), 5)
    mask = mask*0.99 + 0.01

    mask = np.tile(mask[..., np.newaxis], (1, 1, 3))

    if im.dtype == np.uint8:
        im = im.astype(np.float)/255

    if len(im.shape) == 3:
        im = im[:,:,:3]
    if len(im.shape) == 2:
        im = np.tile(im[..., np.newaxis], (1, 1, 3))

    op = im*mask
    op[op>1] = 1
    op[op<0] = 0
    return op


def plot_proj(feats_op, gt, classes, title_txt, fid):
    cols = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    plt.figure(fid)
    for cc in np.unique(gt):
        inds = np.where(gt==cc)[0]
        plt.plot(feats_op[inds, 0], feats_op[inds, 1], cols[cc]+'.', label=classes[cc])

    plt.legend()
    plt.title(title_txt)
    plt.axis('equal')
    plt.show()


class FTNet(torch.nn.Module):
    def __init__(self, backbone, num_classes, compute_bias):
        super(FTNet, self).__init__()
        op_chns = 64
        self.backbone = backbone
        self.conv1 = nn.Conv2d(128, op_chns, kernel_size=3, padding=1)
        self.fc = torch.nn.Linear(op_chns, num_classes, bias=compute_bias)

    def forward(self, data):
        x = self.backbone(data)
        x_feat = F.relu(self.conv1(x))
        x = F.adaptive_avg_pool2d(x_feat, 1)
        x_flat = x.view(x.size(0), -1)
        cls_op = self.fc(x_flat)
        return x_feat, x_flat, cls_op


def cam_mapper(act, cam_weight, cam_bias):
    cam_map = np.zeros((act.shape[0], act.shape[-1],act.shape[-1], num_classes))
    for cc in range(cam_weight.shape[1]):
        cam_map[:,:,:,cc] = (act*cam_weight[:, cc][..., np.newaxis, np.newaxis] + cam_bias[cc]).sum(1)
    cam_map[cam_map<0] = 0
    return cam_map


def resize_cam(cam, orig_size, crop_size):
    # assumes square
    diff = (orig_size - crop_size) / 2
    resize_fact = crop_size / cam.shape[0]
    cam_op = np.ones((orig_size, orig_size, cam.shape[-1]))*cam.min()
    for cc in range(cam.shape[-1]):
        zm = zoom(cam[:,:,cc], (resize_fact, resize_fact), order=1)
        cam_op[diff:-diff,diff:-diff, cc] = zm.copy()
        cam_op[:,:,cc] = gaussian_filter(cam_op[:,:,cc], sigma=1.5)
    return cam_op


# Training settings
lr = 0.0002
weight_decay = 1e-4
momentum = 0.9
batch_size = 64
test_batch_size = 64
log_interval = 100
compute_bias = False

dataset = 'chinese_chars'  # 'oct', 'butterflies_crop', 'chinese_chars'
base_dir = '../../data/'
save_op = False
save_debug_ims = False

if save_op == False:
    print('***\nNot saving outputs\n***')

if dataset == 'oct':
    root_dir = 'oct/images/'
    explain_dir = 'oct/explanations/'
    op_file_name = 'oct'
    orig_size = 144
    crop_size = 128
    epochs = 60
elif dataset == 'chinese_chars':
    root_dir = 'chinese_chars/images/'
    explain_dir = 'chinese_chars/explanations/'
    op_file_name = 'chinese_chars'
    orig_size = 144
    crop_size = 128
    epochs = 60
elif dataset == 'butterflies_crop':
    root_dir = 'butterflies_crop/images/'
    explain_dir = 'butterflies_crop/explanations/'
    op_file_name = 'butterflies_crop'
    orig_size = 144  # assumes square
    crop_size = 128
    epochs = 60


is_cuda = True
kwargs = {'num_workers': 6, 'pin_memory': True} if is_cuda else {}
plt.close('all')

mu_data = [0.485, 0.456, 0.406]
std_data = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        custom_augs.RandomSizedCrop(crop_size),
        custom_augs.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.4,hue=0.25),
        transforms.ToTensor(),
        transforms.Normalize(mean=mu_data, std=std_data)])

test_transform = transforms.Compose([
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize(mean=mu_data, std=std_data)])

# Note this is currently loading all files into both train and test
train_dataset = im_folder.ImageFolder(root=base_dir+root_dir, transform=train_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
all_dataset = im_folder.ImageFolder(root=base_dir+root_dir, transform=test_transform)
all_loader = torch.utils.data.DataLoader(all_dataset, batch_size=batch_size, shuffle=False, **kwargs)

# get dataset details
data, target = next(iter(all_loader))
num_channels = int(data.size()[1])
class_names = all_loader.dataset.classes
num_classes = len(class_names)
gt_labels = np.asarray(all_loader.dataset.class_labels)
imgs = all_loader.dataset.imgs
imgs_files = [ii[len(base_dir):] for ii in imgs]
explain_files = [explain_dir + ii[len(base_dir+root_dir):] for ii in imgs]
print('class names', class_names)

for cc in class_names:
    if not os.path.isdir(base_dir + explain_dir + cc):
        os.makedirs(base_dir + explain_dir + cc)

# use resnet BB
resnet = torchvision.models.resnet18(pretrained=False)
resnetbb = nn.Sequential(*list(resnet.children())[:-4])
model = FTNet(resnetbb, num_classes, compute_bias)

if is_cuda:
    model.cuda()


# train CNN
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
l1_loss_weight = nn.L1Loss()
l1_target_weight = torch.autograd.Variable(torch.zeros(model.fc.weight.size()).cuda())

for epoch in range(1, epochs + 1):

    if epoch == (epochs/2):
        print('dropping learning rate to ', lr / 10.0)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr / 10.0

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if is_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        optimizer.zero_grad()
        _, _, output = model(data)
        #weight_loss = 1000*l1_loss_weight(model.fc.weight, l1_target_weight)
        loss = F.cross_entropy(output, target)# + weight_loss

        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print('Train: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

    # running test
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in all_loader:
        if is_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        _, _, output = model(data)
        test_loss += F.cross_entropy(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss /= len(all_loader.dataset)
    print('Test: Loss: {:.4f}, Acc: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(all_loader.dataset),
        100. * correct / len(all_loader.dataset)))


model.eval()
cam_weight = model.fc.weight.cpu().data.numpy().T
if compute_bias:
    cam_bias = model.fc.bias.cpu().data.numpy()
else:
    cam_bias = np.zeros(num_classes)


# extract features
feats_op = []
cam_op = []
logits = []
cnt = 0
for ii, (data, target) in enumerate(all_loader):
    if is_cuda:
        data, target = data.cuda(), target.cuda()
    data, target = Variable(data, volatile=True), Variable(target)
    feats, feats_flat, output = model(data)

    cam = cam_mapper(feats.cpu().data.numpy(), cam_weight, cam_bias)
    cam_op.append(cam)
    feats_op.append(feats_flat.cpu().data.numpy())
    logits.append(output.cpu().data.numpy())

    for bb in range(cam.shape[0]):

        # save explanation images
        cam_rs = resize_cam(cam[bb, :], orig_size, crop_size)
        if save_op:
            im = plt.imread(imgs[cnt])
            expl_im = gen_mask(cam_rs[:,:,gt_labels[cnt]], im)
            exp_op = base_dir + explain_dir + class_names[gt_labels[cnt]] + '/' + os.path.basename(imgs[cnt])
            cv2.imwrite(exp_op, cv2.cvtColor((expl_im*255).astype(np.uint8), cv2.COLOR_RGB2BGR))

        cnt += 1

feats_op = np.vstack((feats_op))
cam_op = np.vstack((cam_op))
logits = np.vstack((logits))
pred_labels = logits.argmax(1)


# visualize results
if save_debug_ims:
    if not os.path.isdir('im_heat'):
        os.makedirs('im_heat')
    if not os.path.isdir('im_expl'):
        os.makedirs('im_expl')

    print('saving some ims')
    for jj in range(30):
        plt.close('all')
        im_id = np.random.randint(len(imgs))
        im = plt.imread(imgs[im_id])
        cam = resize_cam(cam_op[im_id,:], orig_size, crop_size)

        plt.figure(3)
        plt.gcf().suptitle('%s, GT: %s, Pred: %s' % (os.path.basename(imgs[im_id]), class_names[gt_labels[im_id]], class_names[pred_labels[im_id]]))
        plt.subplot(np.ceil(0.1 + num_classes/2.0), 2, 1)
        plt.imshow(im, cmap='gray', interpolation='bilinear');plt.axis('off')
        plt.title('ip im')
        for ii in range(num_classes):
            plt.subplot(np.ceil(0.01 + num_classes/2.0), 2, ii+2)
            plt.imshow(cam[:,:,ii], vmin=0, vmax=cam.max(), interpolation='bilinear')
            plt.axis('off');plt.title('*'*int(gt_labels[im_id]==ii) + class_names[ii])
        plt.savefig('im_heat/' + str(jj).zfill(3) + '.png')

        expl_im = gen_mask(cam[:,:,gt_labels[im_id]], im)
        plt.figure(4)
        plt.gcf().suptitle('%s, GT: %s, Pred: %s' % (os.path.basename(imgs[im_id]), class_names[gt_labels[im_id]], class_names[pred_labels[im_id]]))
        plt.subplot(1, 2, 1)
        plt.imshow(im, cmap='gray', vmin=0, vmax=1)
        plt.title('ip im')
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(expl_im, cmap='gray', vmin=0, vmax=1)
        plt.axis('off')
        plt.title('expl')
        plt.savefig('im_expl/' + str(jj).zfill(3) + '.png')


# plot
pca = decomposition.PCA(n_components=2)
pca.fit(feats_op, gt_labels)
feats_pca = pca.transform(feats_op)
plot_proj(feats_pca, gt_labels, class_names, 'PCA', 0)

# PW distance
sorted_inds = np.argsort(gt_labels)
feats_op_norm = normalize(feats_op)
dist = squareform(pdist(feats_op_norm[sorted_inds, :], 'cosine'))
print(class_names)

plt.figure(1)
plt.imshow(dist)
plt.show()

# save
if save_op:
    print('saving results ' + op_file_name)
    np.savez(base_dir + op_file_name, im_files=imgs_files, explain_files=explain_files,
             X=feats_op, Y=gt_labels, Y_pred=pred_labels, interp=cam_op,
             class_names=class_names)
else:
    print('not saving output')
