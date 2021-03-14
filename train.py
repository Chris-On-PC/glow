import argparse
import os
import json
import shutil
import random

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

def check_manual_seed(seed):
    seed = seed or random.randint(1, 10000)
    random.seed(seed)
    torch.manual_seed(seed)

    print("Using seed: {seed}".format(seed=seed))

def check_dataset(dataset, dataroot, augment, download):
    if dataset == "cifar10":
        cifar10 = get_CIFAR10(augment, dataroot, download)
        input_size, num_classes, train_dataset, test_dataset = cifar10
    if dataset == "svhn":
        svhn = get_SVHN(augment, dataroot, download)
        input_size, num_classes, train_dataset, test_dataset = svhn

    return input_size, num_classes, train_dataset, test_dataset

def compute_loss(nll, reduction="mean"):
    if reduction == "mean":
        losses = {"nll": torch.mean(nll)}
    elif reduction == "none":
        losses = {"nll": nll}

    losses["total_loss"] = losses["nll"]

    return losses   

def compute_loss_y(nll, y_logits, y_weight, y, multi_class, reduction="mean"):
    if reduction == "mean":
        losses = {"nll": torch.mean(nll)}
    elif reduction == "none":
        losses = {"nll": nll}

    if multi_class:
        y_logits = torch.sigmoid(y_logits)
        loss_classes = F.binary_cross_entropy_with_logits(
            y_logits, y, reduction=reduction
        )
    else:
        loss_classes = F.cross_entropy(
            y_logits, torch.argmax(y, dim=1), reduction=reduction
        )

    losses["loss_classes"] = loss_classes
    losses["total_loss"] = losses["nll"] + y_weight * loss_classes

    return losses

def main():


    # Some input parameters
    cuda = 
    seed = 1
    output_dir = # Directory to output logs and model checkpoints

    # Training parameters
    epochs =            250 # number of epochs to train for
    lr =                5e-4 # Learning rate
    n_workers =         6 # number of data loading workers
    n_init_batches =    8 # Number of batches to use for Act Norm initialisation
    batch_size =        64 # atch size used during training
    eval_batch_size =   512 # batch size used during evaluation
    warmup =            5 # Use this number of epochs to warmup learning rate linearly from zero to learning rate

    # Model parameters
    hidden_channels =   512 # Number of hidden channels
    K =                 32 # Number of layers per block
    L =                 3 # Number of blocks
    actnorm_scale =     1.0 # Act norm scale
    flow_permutation =  'invconv'  # Type of flow permutation
    flow_coupling =     'affine' # Type of flow coupling
    LU_decomposed =  # Train with LU decomposed 1x1 convs
    learn_top =     # Do not train top layer (prior)
    y_condition = # Train using class condition
    y_weight =          0.01 # Weight for class condition loss
    max_grad_clip =     0.0 # Max gradient value (clip above - for off)
    max_grad_norm =     0.0 # Max norm of gradient (clip above - 0 for off)

    # Init
    device = "cpu" if (not torch.cuda.is_available() or not cuda) else "cuda:0"
    check_manual_seed(seed)

    ds = check_dataset(dataset, dataroot, augment, download)
    image_shape, num_classes, train_dataset, test_dataset = ds

    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers, drop_last=True)
    test_loader = data.DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=n_workers, drop_last=False)


def step(engine, batch):
    model.train()
    optimizer.zero_grad()

    x, y = batch
    x = x.to(device)

    if y_condition:
        y = y.to(device)
        z, nll, y_logits = model(x, y)
        losses = compute_loss_y(nll, y_logits, y_weight, y, multi_class)
    else:
        z, nll, y_logits = model(x, None)
        losses = compute_loss(nll)

    losses["total_loss"].backward()

    if max_grad_clip > 0:
        torch.nn.utils.clip_grad_value_(model.parameters(), max_grad_clip)
    if max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    optimizer.step()

    return losses

def eval_step(engine, batch):
    model.eval()

    x, y = batch
    x = x.to(device)

    with torch.no_grad():
        if y_condition:
            y = y.to(device)
            z, nll, y_logits = model(x, y)
            losses = compute_loss_y(
                nll, y_logits, y_weight, y, multi_class, reduction="none"
            )
        else:
            z, nll, y_logits = model(x, None)
            losses = compute_loss(nll, reduction="none")

    return losses