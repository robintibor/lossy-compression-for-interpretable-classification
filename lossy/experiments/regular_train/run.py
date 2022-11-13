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
import os.path
from lossy import wide_nf_net

from lossy.condensation.networks import ConvNet
from lossy.datasets import get_dataset
from lossy.optim import SGD_AGC
from sam_optim.sam import SAM

log = logging.getLogger(__name__)


def run_exp(
    dataset,
    first_n,
    split_test_off_train,
    model_name,
    np_th_seed,
    n_epochs,
    adaptive_gradient_clipping,
    optim_type,
    lr,
    weight_decay,
    depth,
    width,
    dropout,
    activation,
    save_model,
    pooling,
    norm,
    debug,
    output_dir,
):
    assert optim_type in ['sgd', 'adamw', 'sam']
    hparams = {k: v for k, v in locals().items() if v is not None}
    writer = SummaryWriter(output_dir)
    writer.add_hparams(hparams, metric_dict={}, name=output_dir)
    writer.flush()
    assert not ((optim_type == 'adamw') and adaptive_gradient_clipping)

    set_random_seeds(np_th_seed, True)
    # Hyper Parameter settings
    use_cuda = torch.cuda.is_available()
    batch_size = 128

    (
        channel,
        im_size,
        num_classes,
        class_names,
        trainloader,
        train_det_loader,
        testloader,
    ) = get_dataset(
        dataset.upper(),
        "/home/schirrmr/data/pytorch-datasets/",
        standardize=False,
        first_n=first_n,
        batch_size=batch_size,
        eval_batch_size=512,
        split_test_off_train=split_test_off_train,
    )
    if dataset in ['cifar10', 'cifar100', 'stripes']:
        # but won't be applied for stripes actually atm
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(wide_nf_net.mean[dataset], wide_nf_net.std[dataset]),
            ]
        )  # meanstd transformation
    else:
        assert dataset in ['mnist', 'fashionmnist']
        transform_train = transforms.Compose(
            [
                transforms.Resize(
                    32,
                ),  # transforms.InterpolationMode.BILINEAR),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: torch.cat((x, x, x), dim=0)),
                transforms.Normalize(wide_nf_net.mean[dataset], wide_nf_net.std[dataset]),
            ]
        )  # meanstd transformation
    if dataset in ['cifar10', 'cifar100', 'stripes']:
        # but won't be applied for stripes actually atm
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(wide_nf_net.mean[dataset], wide_nf_net.std[dataset]),
            ]
        )
    else:
        transform_test = transforms.Compose(
            [
                transforms.Resize(
                    32,
                ),  # transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: torch.cat((x, x, x), dim=0)),
                transforms.Normalize(wide_nf_net.mean[dataset], wide_nf_net.std[dataset]),
            ]
        )

    train_set = trainloader.dataset
    test_set = testloader.dataset
    if first_n is not None:
        train_set = train_set.dataset
        test_set = test_set.dataset
    if split_test_off_train is True:
        train_set = train_set.dataset
        test_set = test_set.dataset
    if dataset != 'stripes':
        assert hasattr(train_set, 'transform')
        assert hasattr(train_set.transforms, 'transform')
        train_set.transform = transform_train
        train_set.transforms.transform = transform_train
        assert hasattr(test_set, 'transform')
        assert hasattr(test_set.transforms, 'transform')
        test_set.transform = transform_test
        test_set.transforms.transform = transform_testS
    if model_name == 'wide_nf_net':
        from lossy.wide_nf_net import conv_init, Wide_NFResNet

        net = Wide_NFResNet(depth, width, dropout, num_classes,
                            activation=activation).cuda()
        file_name = f"wide-nfresnet-{depth:d}x{width:d}"
        net.apply(conv_init)
    elif model_name == 'wide_bnorm_net':
        from lossy.wide_resnet import conv_init, Wide_ResNet
        assert activation == 'relu'
        net = Wide_ResNet(depth, width, dropout, num_classes).cuda()
        file_name = f"wide-resnet-{depth:d}x{width:d}"
        net.apply(conv_init)
    elif model_name == 'ConvNet':
        net = ConvNet(3, num_classes, width, depth, activation, norm, pooling, (32, 32)).cuda()
    else:
        raise ValueError(f"Unknown model {model_name:s}")

    print(net)
    criterion = nn.CrossEntropyLoss()

    def train(epoch):
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        if optim_type == 'sgd':
            if adaptive_gradient_clipping:
                inner_optim = SGD_AGC(
                    net.parameters(),
                    lr=cf.learning_rate(lr, epoch),
                    momentum=0.9,
                    weight_decay=weight_decay,
                )
            else:
                inner_optim = optim.SGD(
                    net.parameters(),
                    lr=cf.learning_rate(lr, epoch),
                    momentum=0.9,
                    weight_decay=weight_decay,
                )
        #impadd
        elif optim_type == 'sam':
            base_optimizer = torch.optim.SGD  # define an optimizer for the "sharpness-aware" update
            inner_optim = SAM(net.parameters(), base_optimizer, lr=cf.learning_rate(lr, epoch), momentum=0.9)
        else:
            assert optim_type == 'adamw'
            inner_optim = optimizer

        print("\n=> Training Epoch #%d, LR=%.4f" % (epoch, cf.learning_rate(lr, epoch)))
        for i_batch, (inputs, targets) in enumerate(trainloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()  # GPU settings
            inner_optim.zero_grad(set_to_none=True)
            outputs = net(inputs)  # Forward Propagation
            loss = criterion(outputs, targets)  # Loss
            loss.backward()  # Backward Propagation
            if optim_type == 'sam':
                inner_optim.first_step(zero_grad=True)
                outputs = net(inputs)  # Forward Propagation
                loss = criterion(outputs, targets)  # Loss
                loss.backward()
                # second forward-backward pass
                inner_optim.second_step(zero_grad=True)
            else:
                inner_optim.step()  # Optimizer update
            inner_optim.zero_grad(set_to_none=True)

            train_loss += loss.item() * targets.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum().item()

            sys.stdout.write("\r")
            sys.stdout.write(
                "| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%"
                % (
                    epoch,
                    n_epochs,
                    i_batch + 1,
                    len(trainloader),
                    loss.item(),
                    100.0 * correct / total,
                )
            )
            sys.stdout.flush()
        epoch_results = dict(
            train_acc=100.0 * correct / total, train_loss=train_loss / total
        )
        for key, val in epoch_results.items():
            writer.add_scalar(
                key,
                val,
                epoch,
            )
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
            acc = 100.0 * correct / total
            avg_loss = test_loss / total
            print(
                "\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%%"
                % (epoch, avg_loss, acc)
            )
            epoch_results = dict(test_acc=acc, test_loss=avg_loss)
            for key, val in epoch_results.items():
                writer.add_scalar(
                    key,
                    val,
                    epoch,
                )
            return epoch_results

    elapsed_time = 0
    if optim_type == 'adamw':
        optimizer = optim.AdamW(
            net.parameters(), lr=lr, weight_decay=weight_decay, eps=1e-7)

    for epoch in range(0, n_epochs):
        start_time = time.time()

        train_results = train(epoch)
        test_results = test(epoch)

        epoch_time = time.time() - start_time
        elapsed_time += epoch_time
        print("| Elapsed time : %d:%02d:%02d" % (cf.get_hms(elapsed_time)))
        if (not debug) and (save_model):
            torch.save(net.state_dict(), os.path.join(output_dir, "state_dict.th"), )
            #torch.save(net, os.path.join(output_dir, "net.th"), )
    results = dict(**train_results, **test_results)
    writer.close()
    return results
