import logging
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import wide_resnet.config as cf
from braindecode.util import set_random_seeds
from tensorboardX import SummaryWriter
from wide_resnet.networks import *

from lossy.datasets import get_dataset

log = logging.getLogger(__name__)



def run_exp(
        first_n,
        split_test_off_train,
        nf_net,
        np_th_seed,
        n_epochs,
        debug,
        output_dir):
    hparams = {k:v for k,v in locals().items() if v is not None}
    writer = SummaryWriter(output_dir)
    writer.add_hparams(hparams, metric_dict={}, name=output_dir)
    writer.flush()

    lr = 0.1
    depth = 28
    widen_factor = 10
    dropout = 0.3
    dataset = 'cifar10'

    set_random_seeds(np_th_seed, True)
    # Hyper Parameter settings
    use_cuda = torch.cuda.is_available()
    start_epoch, batch_size, optim_type = cf.start_epoch, cf.batch_size, cf.optim_type

    channel, im_size, num_classes, class_names, trainloader, train_det_loader, testloader = get_dataset(
        'CIFAR10', '/home/schirrmr/data/pytorch-datasets/data/CIFAR10/', standardize=False, first_n=first_n,
        batch_size=batch_size, eval_batch_size=512,
        split_test_off_train=split_test_off_train)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cf.mean[dataset], cf.std[dataset]),
    ])  # meanstd transformation

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cf.mean[dataset], cf.std[dataset]),
    ])
    train_set = trainloader.dataset
    test_set = testloader.dataset
    if first_n is not None:
        train_set = train_set.dataset
        test_set = test_set.dataset
    if split_test_off_train is True:
        train_set = train_set.dataset
        test_set = test_set.dataset

    train_set.transform = transform_train
    train_set.transforms.transform = transform_train
    test_set.transform = transform_test
    test_set.transforms.transform = transform_test
    if nf_net:
        from wide_resnet.networks.wide_nfnet import conv_init, Wide_NFResNet
        net = Wide_NFResNet(depth, widen_factor, dropout, num_classes).cuda()
        file_name = f'wide-nfresnet-{depth:d}x{widen_factor:d}'
        net.apply(conv_init)
    else:
        from wide_resnet.networks.wide_resnet import conv_init, Wide_ResNet
        net = Wide_ResNet(depth, widen_factor, dropout, num_classes).cuda()
        file_name = f'wide-resnet-{depth:d}x{widen_factor:d}'
        net.apply(conv_init)

    print(net)
    criterion = nn.CrossEntropyLoss()

    def train(epoch):
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        optimizer = optim.SGD(net.parameters(), lr=cf.learning_rate(lr, epoch), momentum=0.9, weight_decay=5e-4)

        print('\n=> Training Epoch #%d, LR=%.4f' % (epoch, cf.learning_rate(lr, epoch)))
        for i_batch, (inputs, targets) in enumerate(trainloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()  # GPU settings
            optimizer.zero_grad()
            outputs = net(inputs)  # Forward Propagation
            loss = criterion(outputs, targets)  # Loss
            loss.backward()  # Backward Propagation
            optimizer.step()  # Optimizer update

            train_loss += loss.item() * targets.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum().item()

            sys.stdout.write('\r')
            sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%'
                             % (epoch, n_epochs, i_batch + 1,
                                len(trainloader), loss.item(), 100. * correct / total))
            sys.stdout.flush()
        epoch_results = dict(train_acc=100. * correct / total,
                             train_loss=train_loss / total)
        for key, val in epoch_results.items():
            writer.add_scalar(key, val, epoch, )
        return epoch_results

    def test(epoch):
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for i_batch, (inputs, targets) in enumerate(testloader):
                if use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item() * targets.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).cpu().sum().item()

            # Save checkpoint when best model
            acc = 100. * correct / total
            avg_loss = test_loss / total
            print("\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%%" % (epoch, avg_loss, acc))
            epoch_results = dict(test_acc=acc,
                                 test_loss=avg_loss)
            for key, val in epoch_results.items():
                writer.add_scalar(key, val, epoch, )
            return epoch_results

    elapsed_time = 0
    for epoch in range(start_epoch, start_epoch + n_epochs):
        start_time = time.time()

        train_results = train(epoch)
        test_results = test(epoch)

        epoch_time = time.time() - start_time
        elapsed_time += epoch_time
        print('| Elapsed time : %d:%02d:%02d' % (cf.get_hms(elapsed_time)))
    results = dict(**train_results, **test_results)
    writer.close()
    return results




