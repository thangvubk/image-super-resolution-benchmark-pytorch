from __future__ import division
import tensorflow as tf
import scipy
import glob
import numpy as np
from model import *
from utils import *
import time
import os
import argparse
from torch.autograd import Variable
parser = argparse.ArgumentParser(description='SR benchmark')
parser.add_argument('-m', '--model', metavar='M', type=str, default='VDSR',
                    help='network architecture. Default SRCNN')
parser.add_argument('-s', '--scale', metavar='S', type=str, default='3x', 
                    help='interpolation scale. Default 3x')
parser.add_argument('-t', '--test-set', metavar='T', type=str, default='set5')
args = parser.parse_args()

def compute_PSNR(out, lbl):
    diff = out - lbl
    rmse = np.sqrt(np.mean(diff**2))
    psnr = 20*np.log10(255/rmse)
    return psnr

def main(argv=None):
    test_path = os.path.join('data/preprocessed_data/test', args.test_set, args.scale)
    save_path = os.path.join('results', args.model, args.test_set, args.scale)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    lr_paths = glob.glob(os.path.join(test_path, 'low_res', '*.bmp'))
    hr_paths = glob.glob(os.path.join(test_path, 'high_res', '*.bmp'))
    lr_paths.sort()
    hr_paths.sort()
    
    check_point = os.path.join('check_point', args.model, args.scale, 'model.pt')
    if not os.path.exists(check_point):
        raise Exception('Cannot find %s' %check_point)
    model = torch.load(check_point)
    model.cuda()
    psnrs = []
    for lr_path, hr_path in zip(lr_paths, hr_paths):
        # inp and lbl is in [0: 255] range
        inp = scipy.misc.imread(lr_path)
        inp = inp/255 - 0.5
        inp = inp[np.newaxis, np.newaxis, :, :]
        inp = Variable(torch.Tensor(inp).cuda())

        #since = time.time()
        out = model(inp) 
        #print(time.time() - since)
        
        out = (out + 0.5)* 255
        out = out.data.cpu().numpy()
        out = out[0, 0, :, :]
        lbl = scipy.misc.imread(hr_path)
        psnrs.append(compute_PSNR(out, lbl))
        print('%20s: %.3fdB' %(os.path.basename(lr_path), compute_PSNR(out, lbl)))
        scipy.misc.imsave(os.path.join(save_path, os.path.basename(lr_path)), out)
    print('average: %.4fdB' %np.mean(psnrs))

if __name__ == '__main__':
    tf.app.run()
    


