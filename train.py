import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.data_loading import *
from utils.model import *
import os
import argparse
import datetime, time
import numpy as np
import random
# import timm
def train(model, optimizor, criterion, loader, args):
    model.train() # set model to train mode
    all_loss = 0.0
    for i, (x, y) in enumerate(iter(loader)):
        if args.cuda:
            x = x.cuda()
            y = y.cuda()
        loss = criterion(model(x), y)
        # TODO
        # Write the training process of each step
        # Hint: three components: clear gradients, back propagation, update parameters
        optimizor.zero_grad()
        loss.backward()
        optimizor.step()
        ###################################
        all_loss += loss.item()
    args.training_loss = all_loss
    args.end_time = time.time()
    print('Training loss: {:.2f}'.format(all_loss))
    

def validation(model, loader, args):
    model.eval() # set model to eval mode
    count = 0
    correct_count = 0
    for i, (x, y) in enumerate(iter(loader)):
        if args.cuda:
            x = x.cuda()
            y = y.cuda()
        inference = torch.argmax(model(x), dim=-1)
        correct_count += (inference == y).sum().item()
        count += y.shape[0]
    args.end_time = time.time()
    args.validation_acc = correct_count/count*100
    print("Validation accuracy: {:.2f}%".format(args.validation_acc))

def seed_init(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training timm models.')
    # data
    parser.add_argument('--data_dir', type=str, default='./play_dataset')
    parser.add_argument('--data_name', type=str, default='play_dataset')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes in the dataset')
    # model training
    parser.add_argument('--model', type=str, default='resnet18', help='The name of model in the TIMM library')
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=8)
    # optimizor hyperparameters
    parser.add_argument('--optim', type=str, default='SGD', choices=['SGD', 'Adam'], help='The choice of optimizor, support: SGD, ADAM')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--momentum', type=float, default=0.9)
    # preparation
    parser.add_argument('--no_cuda', action='store_true', help='Keep from using cuda')
    parser.add_argument('--gpu', type=int, default=0, help='Assign cuda no.')
    parser.add_argument('--log', type=str, default='./log', help='Where logs and results are stored')
    parser.add_argument('--seed', type=int, default=2021, help='Initialize random seed')
    
    args = parser.parse_args()
    # PREPARATION: cuda, run name, log, csv, seed
    args.cuda = not args.no_cuda
    if args.cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    # I recommend having a unique name for each 
    args.name = 'DATA{}_MODEL{}_EPOCH{}_BATCH{}_OPTIM{}_LR{}_MOM{}_DECAY{}_SEED{}'.format( \
        args.data_name, args.model, args.epoch, args.batch_size, args.optim, \
        args.lr, args.momentum, args.weight_decay, args.seed \
        )
    # log: txt + csv
    os.system('mkdir -p {}'.format(args.log)) # mkdir: log/model_name
    args.log += '/{}.{}'.format(args.name, 'txt') # log file
    with open(args.log, 'w') as logger:
        logger.write('Run: '+str(args.name)+'\n')
        logger.write('Time of start: '+str(datetime.datetime.now())+ '\n')
        logger.write('Args: ' + str(args) + '\n')
        logger.write('=========================================\n')

    args.csv = args.log[:-4] + '.csv' # csv results
    with open(args.csv, 'w') as logcsv:
        logcsv.write('epoch,training_loss,training_time,validation_acc\n')
    # seed
    seed_init(args)
    print("RUN: ", args.name)

    # LOAD DATA
    _, dset_loader = load_data(args.data_dir, args.batch_size, args.seed)

    # CREATE MODEL & CRITERION
    model = my_model(args)
    # model = timm.create_model(args.model, num_classes=args.num_classes)
    criterion = nn.CrossEntropyLoss()
    if args.cuda:
        model = model.cuda()
        criterion = criterion.cuda()
    if args.optim == 'SGD':
        optimizor = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim == 'Adam':
        optimizor = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


    # TRAIN & VALIDATE
    for epoch in range(args.epoch):
        print("At epoch {}: ".format(epoch))
        args.start_time = time.time()
        train(model, optimizor, criterion, dset_loader['tr'], args)
        print('Training time (Seconds): {:.2f}'.format(args.end_time - args.start_time))
        validation(model, dset_loader['te'], args)


        # Training at this epoch completed; print log
        with open(args.log, 'a+') as logger:
            logger.write("Epoch {:0>3d}/{:0>3d}, Training Loss {:.2f}, Training Time {:.2f}s, Validation Accuracy {:.2f}\n".format(epoch, args.epoch, args.training_loss, (args.end_time - args.start_time), args.validation_acc))
            logger.write('End of this epoch at: '+str(datetime.datetime.now())+ '\n')
        with open(args.csv, 'a+') as logcsv:
            logcsv.write('{},{:.2f},{:.2f},{:.2f}\n'.format(epoch, args.training_loss, (args.end_time - args.start_time), args.validation_acc))