import time
from math import ceil
from networks import DenoisingDatasets
from networks.archs import UWKAN
import torch.optim as optim
import torch
import os
from options import set_opts

args = set_opts()  # 传入参数

net = UWKAN(2, 2).cuda()

# optimizer
optimizer = optim.Adam(net.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)

datasets = {'train': DenoisingDatasets.SimulateTrain(args.Train_dir, SNR_min=args.SNR_min, SNR_max=args.SNR_max)}

lr_scheduler = scheduler
batch_size = {'train': args.batch_size}
data_loader = {phase: torch.utils.data.DataLoader(datasets[phase], batch_size=batch_size[phase],
                                                  shuffle=True, num_workers=args.num_workers, pin_memory=True) for phase
               in datasets.keys()}
num_data = {phase: len(datasets[phase]) for phase in datasets.keys()}
num_iter_epoch = {phase: ceil(num_data[phase] / batch_size[phase]) for phase in datasets.keys()}
for epoch in range(args.epochs):
    tic = time.time()
    net.train()
    lr = optimizer.param_groups[0]['lr']
    phase = 'train'
    for ii, data in enumerate(data_loader[phase]):
        im_noisy, im_gt, label = [x.cuda() for x in data]
        phi_Z = net(im_noisy)
        loss = torch.mean((phi_Z - im_gt) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        # save model
        if (epoch + 1) % args.save_model_freq == 0 or epoch + 1 == args.epochs:
            model_state_prefix = 'model_state_'
            save_path_model_state = os.path.join(args.model_dir, model_state_prefix + str(epoch + 1))
            torch.save(net.state_dict(), save_path_model_state)

        toc = time.time()
        print('This epoch take time {:.2f}'.format(toc - tic))
print('Reach the maximal epochs! Finish training')
