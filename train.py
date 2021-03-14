import os
import sys
import time
import datetime
import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
# from torchvision import datasets
# from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

from utils.utils import weights_init_normal, compute_loss

from mymodel import UltraNet
import test

from data.dataset import DACDataset
from data.augmentation import Augmentation, BaseTransform


wdir = 'weights' + os.sep  # weights dir
last = wdir + 'last.pt'
best = wdir + 'best.pt'
test_best = wdir + 'test_best.pt'
results_file = 'results.txt'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=500)  # 500200 batches at bs 16, 117263 COCO images = 273 epochs
    parser.add_argument('--batch-size', type=int, default=16)  # effective bs = batch_size * accumulate = 16 * 4 = 64
    parser.add_argument('--accumulate', type=int, default=4, help='batches to accumulate before optimizing')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-tiny-1cls_1.cfg', help='*.cfg path')
    parser.add_argument('--data', type=str, default='data/coco2017.data', help='*.data path')
    parser.add_argument('--multi-scale', action='store_true', help='adjust (67% - 150%) img_size every 10 batches')
    parser.add_argument('--img-size', nargs='+', type=int, default=[320], help='train and test image-sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', action='store_true', help='resume training from last.pt')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    # parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--arc', type=str, default='default', help='yolo architecture')  # default, uCE, uBCE
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--adam', action='store_true', help='use adam optimizer')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    parser.add_argument('--var', type=float, help='debug variable')
    opt = parser.parse_args()
    opt = parser.parse_args()
    print(opt)

    device = torch_utils.select_device(opt.device, batch_size=opt.batch_size)

    train()

def train():
    img_size, img_size_test = opt.img_size if len(opt.img_size) == 2 else opt.img_size * 2  # train, test sizes
    epochs = opt.epochs  
    batch_size = opt.batch_size
    accumulate = opt.accumulate  # effective bs = batch_size * accumulate = 16 * 4 = 64
    weights = opt.weights  # initial training weights

    # remove previous results
    for f in glob.glob('*_batch*.png') + glob.glob(results_file):
        os.remove(f)

    # init model
    model = UltraNet().to(device)
    model.apply(weights_init_normal)

    # optimizer 
    optimizer = torch.optim.Adam(model.parameters())

    # cosine lr
    lf = lambda x: (1 + math.cos(x * math.pi / epochs)) / 2 * 0.99 + 0.01  # cosine 
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[round(epochs * x) for x in [0.8, 0.9]], gamma=0.1)
    scheduler.last_epoch = 0

    root = "/share/DAC2020/dataset/"
    dataset = DACDataset(root, "train", BaseTransform(320, 160))

    # Dataloader
    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=nw,
                                             shuffle=not opt.rect,  # Shuffle=True unless rectangular training is used
                                             pin_memory=True,
                                             #collate_fn=dataset.collate_fn
                                             )

    # Testloader
    testloader = torch.utils.data.DataLoader(DACDataset(root, "test", BaseTransform(320, 160)),
                                             batch_size=batch_size * 2,
                                             num_workers=nw,
                                             pin_memory=True,
                                             #collate_fn=dataset.collate_fn
                                             )

    nc = 13
    model.nc = nc  # attach number of classes to model
    model.arc = opt.arc  # attach yolo architecture
    model.hyp = hyp  # attach hyperparameters to model
    #model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device)  # attach class weights
    model.class_weights = torch.ones(13)/13
    maps = np.zeros(nc)  # mAP per class


    for epoch in range(opt.epochs):
        model.train()
        start_time = time.time()
        train_loss = 0
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))  # progress bar
        for batch_i, (_, imgs, targets) in pbar:
            batches_done = len(dataloader) * epoch + batch_i

            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)

            # multi-scale is not used here

            # forward 
            pred = model(imgs)

            # compute loss
            loss, loss_items = compute_loss(pred, targets, model)
            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ', loss_items)
                return results
            loss.backward()
            loss = loss*batch_size/64

            train_loss = (train_loss * batch_i + loss.item()) / (batch_i + 1)

            # optimize every accumulate
            if batch_done % accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()

            s = ('%10s' + '%10.3g' * 3) % ('%g/%g' % (epoch, epochs - 1),  train_loss, len(targets), img_size)
            pbar.set_description(s)


        # end one epoch
        scheduler.step()

        # process data of current epoch
        final_epoch = (epoch + 1 == epochs)
        results = test.test(batch_size=batch_size * 2,
                            img_size=img_size_test,
                            model=model,
                            conf_thres=0.001,  # 0.001 if opt.evolve or (final_epoch and is_coco) else 0.01,
                            iou_thres=0.6,
                            save_json=final_epoch and is_coco,
                            single_cls=opt.single_cls,
                            dataloader=testloader)

        # Write epoch results
        with open(results_file, 'a') as f:
            f.write(s + '%10.3g' * len(results) % results + '\n')  # P, R, mAP, F1, test_losses=(GIoU, obj, cls)

        x = list(train_loss) + list(results)
        titles = ['Train loss',
                    'iou', 'Test_loss', 'Giou loss', 'obj loss']
        for xi, title in zip(x, titles):
            tb_writer.add_scalar(title, xi, epoch)

        # Save training results
        save = (not opt.nosave) or (final_epoch)
        if save:
            with open(results_file, 'r') as f:
                # Create checkpoint
                chkpt = {'epoch': epoch,
                         # 'best_fitness': best_fitness,
                         'training_results': f.read(),
                         'model': model.module.state_dict() if type(
                             model) is nn.parallel.DistributedDataParallel else model.state_dict(),
                         'optimizer': None if final_epoch else optimizer.state_dict()}

            # Save last checkpoint
            torch.save(chkpt, last)

            # Delete checkpoint
            del chkpt

        # end training

        torch.cuda.empty_cache()
        return results

