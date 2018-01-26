from __future__ import division
from __future__ import print_function
from data import *
from model import *
from solver import *
from loss import *
from utils import *
import progressbar
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import argparse
parser = argparse.ArgumentParser(description='SR benchmark')
parser.add_argument('-m', '--model', metavar='M', type=str, default='VDSR',
                    help='network architecture. Default SRCNN')
parser.add_argument('-s', '--scale', metavar='S', type=str, default='3x', 
                    help='interpolation scale. Default 3x')
parser.add_argument('-b', '--batch-size', metavar='B', type=int, default=64,
                    help='batch size used for training. Default 64')
parser.add_argument('-l', '--learning-rate', metavar='L', type=float, default=1e-3,
                    help='learning rate used for training. Default 1e-3')
parser.add_argument('-n', '--num-epochs', metavar='N', type=int, default=50,
                    help='number of training epochs. Default 100')
parser.add_argument('-f', '--fine-tune', dest='fine_tune', action='store_true',
                    help='fine tune the model under check_point dir,\
                    instead of training from scratch. Default False')

args = parser.parse_args()

def main(argv=None):
    dataset = Trainset('data/preprocessed_data/train/3x/dataset.h5')
    data_loader = DataLoader(dataset, batch_size=args.batch_size,
                             shuffle=True, num_workers=4)
    model = get_model(args.model)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)
    model.cuda()
    optimizer = optim.Adam(model.parameters(),
                           lr=args.learning_rate,
                           weight_decay=1e-6)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    loss_fn = nn.MSELoss()
    for epoch in range(args.num_epochs):
        num_batches = int(math.ceil(len(dataset)/args.batch_size))
        bar = progressbar.ProgressBar(max_value=num_batches)
        #print(optimizer.param_groups[0]['lr'])
        running_loss = 0
        for i, (inputs, labels) in enumerate(data_loader):
            inputs, labels = (Variable(inputs.cuda()),
                              Variable(labels.cuda()))
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            running_loss += loss.data[0]
            loss.backward()
            optimizer.step()
            bar.update(i, force=True)
        scheduler.step()
        print('epoch %d: loss %f' %(epoch, running_loss/num_batches))

    # save model
    check_point = os.path.join('check_point', args.model, args.scale)
    if not os.path.exists(check_point):
        os.makedirs(check_point)
    model_path = os.path.join(check_point, 'model.pt')
    torch.save(model, model_path)

if __name__ == '__main__':
    main()
