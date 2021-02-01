#!/usr/bin/env python3
#--Ayan Chakrabarti <ayan@wustl.edu>
import numpy as np
import ctrlc
import time
import sys
import os
import importlib
import argparse
import json

VALITER=1e3
DISPITER=10
WD=1e-4
MAXITER=64e4

def get_lr(niter):
    if niter < 16e4:
        return 1e-1
    elif niter < 32e4:
        return 1e-2
    elif niter < 48e4:
        return 1e-3
    else:
        return 1e-4


def mprint(s):
    sys.stdout.write(time.strftime("%Y-%m-%d %H:%M:%S ")+s+"\n")
    sys.stdout.flush()
    
parser = argparse.ArgumentParser()
parser.add_argument('-save',help='Path to model file name',required=True)
parser.add_argument('-split',help='How many sub-batches to split in per gpu',type=int,default=2)
parser.add_argument('-ngpu',help='How many gpus to use',type=int,default=1)
parser.add_argument('-qtype',type=int,default=0,
                    help='Quantization type: 0: None. 4, 8: Simple quantization to 4 or 8 bits. ' \
                         'Defalut: 0.')
parser.add_argument('-bs', type=int, default=128, help="Batch size")
opts = parser.parse_args()
saveloc = opts.save

BSZ=opts.bs

##########################################################################################
# Load data

BDIR='/home/ubuntu/imagenet/'
train = [f.rstrip().split(',') for f in open(BDIR+'train.txt').readlines()]
val = [f.rstrip().split(',') for f in open(BDIR+'val.txt').readlines()]

train_im = [f[0] for f in train]
train_lb = [int(f[1]) for f in train]

val_im = [f[0] for f in val]
val_lb = [int(f[1]) for f in val]

##########################################################################################

from imResnet152 import Model
g = Model(BSZ//(opts.split*opts.ngpu),opts.qtype,WD,opts.split,opts.ngpu)


ESIZE = len(train_im)//BSZ
VSIZE = len(val_im)//BSZ

rs = np.random.RandomState(0)

origiter = 0
if os.path.isfile(saveloc):
    origiter = g.load(saveloc)
    for k in range( (origiter+ESIZE-1) // ESIZE):
        idx = rs.permutation(len(train_im))
    mprint("Restored to iteration %d" % origiter)    
niter = origiter    

avg_loss = 0.; avg_acc = 0.
train_ips_list = []
while niter < MAXITER+1:
    lr = get_lr(niter)
    
    if niter % VALITER == 0:
        vacc = 0.; vloss = 0.; viter = 0
        for b in range(VSIZE):
            b_im = [val_im[i] for i in range(b*BSZ,(b+1)*BSZ)]
            b_lb = [val_lb[i] for i in range(b*BSZ,(b+1)*BSZ)]
            b_lb = np.reshape(np.int32(b_lb),[-1,1,1,1])

            for s in range(opts.split):
                acc,loss = g.forward([False,b_im[s::opts.split]],b_lb[s::opts.split,...],True)
                viter = viter + 1;vacc = vacc + acc;vloss = vloss + loss

        vloss = vloss / viter; vacc = vacc / viter
        mprint("[%09d] Val_Loss = %.3e, Val_Acc = %.4f" % (niter,vloss,vacc))

    if niter == MAXITER:
        break

    b = niter % ESIZE 
    if b == 0:
        idx = rs.permutation(len(train_lb))

    b_im = [train_im[idx[i]] for i in range(b*BSZ,(b+1)*BSZ)]
    b_lb = [train_lb[idx[i]] for i in range(b*BSZ,(b+1)*BSZ)]
    b_lb = np.reshape(np.int32(b_lb),[-1,1,1,1])

    tic = time.time()
    g.binit()
    for s in range(opts.split):
        acc,loss = g.forward([True,b_im[s::opts.split]],b_lb[s::opts.split,...],True)
        avg_loss = avg_loss + loss; avg_acc = avg_acc + acc;
        g.hback()
    g.cback(lr)
    batch_time = time.time() - tic

    ips = BSZ / batch_time
    train_ips_list.append(ips)
    print("Batch size: %d\tTrain throughput: %.2f ips" % (BSZ, ips))
    if niter >= 3:
        out_file = "speed_results.tsv"
        with open(out_file, "a") as fout:
            val_dict = {
                "network": 'resnet152',
                "algorithm": "blpa-exact" if opts.qtype == 0 else "blpa",
                "batch_size": opts.bs,
                "ips": np.median(train_ips_list),
            }
            fout.write(json.dumps(val_dict) + "\n")
            print(f"save results to {out_file}")
            exit()

    niter = niter+1
    
    if niter % DISPITER == 0:
        avg_loss = avg_loss / (DISPITER*opts.split); avg_acc = avg_acc / (DISPITER*opts.split)
        mprint("[%09d] lr=%.2e, Train_Loss = %.3e, Train_Acc = %.4f" % (niter,lr,avg_loss,avg_acc))
        avg_loss = 0.; avg_acc = 0.;
        if ctrlc.stop:
            break

if niter > origiter:
    g.save(saveloc,niter)
    mprint("Saved model.")
