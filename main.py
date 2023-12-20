import logging
from tqdm import tqdm
from time import time
from dataset import Flickr


# TODO: Rewrite the following function
def train(model, epoch, train_loader, optimizer, loss_func, scheduler, global_step, device):
    max_epoch = 100
    model.train()
    timer = dict(start_time=time(), curr_time=time(), end_time=time(), dataloader=0.001, model=0.001, statistics=0.001)
    num_top1, num_sample = 0, 0
    train_iter = train_loader  # if self.no_progress_bar else tqdm(self.train_loader, dynamic_ncols=True)
    for num, (x, y, _) in enumerate(train_iter):
        optimizer.zero_grad()

        # Using GPU
        x = x.float().to(device)
        y = y.long().to(device)
        timer['dataloader'] += time() - timer['curr_time']
        timer['curr_time'] = time()

        # Calculating Output
        out, _ = model(x)

        # Updating Weights
        loss = loss_func(out, y)
        loss.backward()
        optimizer.step()
        scheduler.step()
        global_step += 1

        timer['model'] += time() - timer['curr_time']
        timer['curr_time'] = time()

        # Calculating Recognition Accuracies
        num_sample += x.size(0)
        reco_top1 = out.max(1)[1]
        num_top1 += reco_top1.eq(y).sum().item()

        # Showing Progress
        lr = optimizer.param_groups[0]['lr']
        # if self.scalar_writer:
        #     self.scalar_writer.add_scalar('learning_rate', lr, self.global_step)
        #     self.scalar_writer.add_scalar('train_loss', loss.item(), self.global_step)
        # if self.no_progress_bar:
        logging.info('Epoch: {}/{}, Batch: {}/{}, Loss: {:.4f}, LR: {:.4f}'.format(
            epoch + 1, max_epoch, num + 1, len(train_loader), loss.item(), lr
        ))
        # else:
        #     train_iter.set_description('Loss: {:.4f}, LR: {:.4f}'.format(loss.item(), lr))
        timer['statistics'] += time() - timer['curr_time']
        timer['curr_time'] = time()

    timer['total'] = time() - timer['start_time']
    # Showing Train Results
    train_acc = num_top1 / num_sample
    # if self.scalar_writer:
    #     self.scalar_writer.add_scalar('train_acc', train_acc, global_step)
    logging.info('Epoch: {}/{}, Training accuracy: {:d}/{:d}({:.2%}), Training time: {:.2f}s'.format(
        epoch + 1, max_epoch, num_top1, num_sample, train_acc, timer['total']
    ))
    logging.info('Dataloader: {:.2f}s({:.1f}%), Network: {:.2f}s({:.1f}%), Statistics: {:.2f}s({:.1f}%)'.format(
        timer['dataloader'], timer['dataloader'] / timer['total'] * 100, timer['model'], timer['model'] / timer['total'] * 100,
        timer['statistics'], timer['statistics'] / timer['total'] * 100
    ))
    logging.info('')


def main():
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG) #, filename='example.log')
    train_dataset = Flickr()


if __name__ == '__main__':
    main()
