import logging, torch, pickle, numpy as np

from sklearn.metrics import jaccard_score
from tqdm import tqdm
from time import time

from . import utils as U
from .initializer import Initializer


class Processor(Initializer):

    def train(self, epoch):
        self.model.train()
        timer = dict(start_time=time(), curr_time=time(), end_time=time(), dataloader=0.001, model=0.001, statistics=0.001)
        buffer_perclass = [(torch.ones(self.train_batch_size) * i).to(self.device) for i in range(self.num_class)]
        num_top1, num_sample, num_top1_perclass, num_sample_perclass = 0, 0, torch.zeros(self.num_class), torch.zeros(self.num_class)
        batch_jaccard21, class_counts21 = 0, torch.zeros(21*21, dtype=int)
        batch_jaccard6, class_counts6 = 0, torch.zeros(6*6, dtype=int)
        train_iter = self.train_loader if self.no_progress_bar else tqdm(self.train_loader, dynamic_ncols=True)
        all_preds6, all_labels6 = [], []
        for num, (x, y) in enumerate(train_iter):
            self.optimizer.zero_grad()

            # Using GPU
            x = x.float().to(self.device)
            if self.args.dataset_args['subset'] == 'binary':
                y = y.long().to(self.device)
            elif self.args.dataset_args['subset'] == 'signature':
                y21 = y[0].long().to(self.device)
                y6 = y[1].long().to(self.device)
                all_labels6 += list(y6.detach().cpu())
            timer['dataloader'] += time() - timer['curr_time']
            timer['curr_time'] = time()

            # Calculating Output
            out, _ = self.model(x)

            # Updating Weights
            if self.args.dataset_args['subset'] == 'binary':
                loss = self.loss_func(out, y.float())
            elif self.args.dataset_args['subset'] == 'signature':
                out21, out6 = out
                all_preds6 += list(out6.detach().cpu())
                loss1 = self.loss_func(out21, y21.float())
                loss2 = self.loss_func(out6, y6.float())
                loss = loss1 + loss2
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.global_step += 1

            timer['model'] += time() - timer['curr_time']
            timer['curr_time'] = time()

            num_sample += x.size(0)
            if self.args.dataset_args['subset'] == 'binary':
                # Calculating Recognition Accuracies
                reco_top1 = out.max(1)[1]
                num_top1 += reco_top1.eq(y).sum().item()

                # Calculating Balanced Accuracy
                for i in range(self.num_class):
                    num_top1_perclass[i] += (reco_top1.eq(buffer_perclass[i][:len(reco_top1)]) & (y.eq(buffer_perclass[i][:len(reco_top1)]))).sum().item()
                    num_sample_perclass[i] += y.eq(buffer_perclass[i][:len(y)]).sum().item()
            elif self.args.dataset_args['subset'] == 'signature':
                # Calculating Jaccard Index
                preds = torch.sigmoid(out21.detach().cpu()) > self.multilabel_thresh
                batch_jaccard21 += jaccard_score(y21.cpu(), preds, average='macro') * x.size(0)  # multiplying with batch size
                class_counts21 += y21.sum(axis=0).detach().cpu()
                logging.debug(f"Batch jaccard 21 regions: {batch_jaccard21}")

                preds = torch.sigmoid(out6.detach().cpu()) > self.multilabel_thresh
                batch_jaccard6 += jaccard_score(y6.cpu(), preds, average='macro') * x.size(0)  # multiplying with batch size
                class_counts6 += y6.sum(axis=0).detach().cpu()
                logging.debug(f"Batch jaccard 6 regions: {batch_jaccard6}")

            # Showing Progress
            lr = self.optimizer.param_groups[0]['lr']
            if self.scalar_writer:
                self.scalar_writer.add_scalar('learning_rate', lr, self.global_step)
                self.scalar_writer.add_scalar('train_loss', loss.item(), self.global_step)
            if self.no_progress_bar:
                logging.info('Epoch: {}/{}, Batch: {}/{}, Loss: {:.4f}, LR: {:.4f}'.format(
                    epoch + 1, self.max_epoch, num + 1, len(self.train_loader), loss.item(), lr
                ))
            else:
                train_iter.set_description('Loss: {:.4f}, LR: {:.4f}'.format(loss.item(), lr))
            timer['statistics'] += time() - timer['curr_time']
            timer['curr_time'] = time()

        timer['total'] = time() - timer['start_time']
        if self.args.dataset_args['subset'] == 'binary':
            # Showing Train Results
            train_acc = num_top1 / num_sample
            train_bacc = float(np.average(num_top1_perclass / num_sample_perclass))
            if self.scalar_writer:
                self.scalar_writer.add_scalar('train_acc', train_acc, self.global_step)
                self.scalar_writer.add_scalar('train_bacc', train_bacc, self.global_step)
            logging.info('Epoch: {}/{}, Training accuracy: {:d}/{:d}({:.2%}), Training balanced accuracy: {:.2%}, Training time: {:.2f}s'.format(
                epoch + 1, self.max_epoch, num_top1, num_sample, train_acc, train_bacc, timer['total']
            ))
        elif self.args.dataset_args['subset'] == 'signature':
            # Showing Train Results
            train_jaccard21 = batch_jaccard21 / num_sample
            train_jaccard6 = batch_jaccard6 / num_sample
            if self.scalar_writer:
                self.scalar_writer.add_scalar('train_jaccard21', train_jaccard21, self.global_step)
                self.scalar_writer.add_scalar('train_jaccard6', train_jaccard6, self.global_step)
            logging.info('Epoch: {}/{}, Training jaccard index: 21 regions - {:.2%}, 6 regions - {:.2%} Training time: {:.2f}s'.format(
                epoch + 1, self.max_epoch, train_jaccard21, train_jaccard6, timer['total']
            ))
        logging.info('Dataloader: {:.2f}s({:.1f}%), Network: {:.2f}s({:.1f}%), Statistics: {:.2f}s({:.1f}%)'.format(
            timer['dataloader'], timer['dataloader'] / timer['total'] * 100, timer['model'], timer['model'] / timer['total'] * 100,
            timer['statistics'], timer['statistics'] / timer['total'] * 100
        ))
        logging.info('')

    def eval(self, save_score=False):
        self.model.eval()
        start_eval_time = time()
        score = {}
        with torch.no_grad():
            num_top1 = 0
            num_sample, eval_loss = 0, []
            buffer_perclass = [(torch.ones(self.eval_batch_size) * i).to(self.device) for i in range(self.num_class)]
            num_top1_perclass, num_sample_perclass = torch.zeros(self.num_class), torch.zeros(self.num_class)
            cm = np.zeros((self.num_class, self.num_class))
            batch_jaccard21, class_counts21 = 0, torch.zeros(21 * 21, dtype=torch.int64)
            batch_jaccard6, class_counts6 = 0, torch.zeros(6 * 6, dtype=torch.int64)
            eval_iter = self.eval_loader if self.no_progress_bar else tqdm(self.eval_loader, dynamic_ncols=True)
            for num, (x, y) in enumerate(eval_iter):

                # Using GPU
                x = x.float().to(self.device)
                if self.args.dataset_args['subset'] == 'binary':
                    y = y.long().to(self.device)
                elif self.args.dataset_args['subset'] == 'signature':
                    y21 = y[0].long().to(self.device)
                    y6 = y[1].long().to(self.device)

                # Calculating Output
                out, _ = self.model(x)

                # Getting Loss
                if self.args.dataset_args['subset'] == 'binary':
                    loss = self.loss_func(out, y.float())
                elif self.args.dataset_args['subset'] == 'signature':
                    out21, out6 = out
                    loss1 = self.loss_func(out21, y21.float())
                    loss2 = self.loss_func(out6, y6.float())
                    loss = loss1 + loss2
                eval_loss.append(loss.item())

                if save_score:
                    for n, c in zip("", out.detach().cpu().numpy()):
                        score[n] = c

                num_sample += x.size(0)
                if self.args.dataset_args['subset'] == 'binary':
                    # Calculating Recognition Accuracies
                    reco_top1 = out.max(1)[1]
                    num_top1 += reco_top1.eq(y).sum().item()

                    # Calculating Balanced Accuracy
                    for i in range(self.num_class):
                        num_top1_perclass[i] += (reco_top1.eq(buffer_perclass[i][:len(reco_top1)]) & (y.eq(buffer_perclass[i][:len(reco_top1)]))).sum().item()
                        num_sample_perclass[i] += y.eq(buffer_perclass[i][:len(y)]).sum().item()

                    # Calculating Confusion Matrix
                    for i in range(x.size(0)):
                        cm[y[i], reco_top1[i]] += 1
                elif self.args.dataset_args['subset'] == 'signature':
                    # Calculating Jaccard Index
                    preds = torch.sigmoid(out21.detach().cpu()) > self.multilabel_thresh
                    batch_jaccard21 += jaccard_score(y21.cpu(), preds, average='macro') * x.size(0)  # multiplying with batch size
                    class_counts21 += y21.sum(axis=0).detach().cpu()
                    logging.debug(f"Batch jaccard 21 regions: {batch_jaccard21}")

                    preds = torch.sigmoid(out6.detach().cpu()) > self.multilabel_thresh
                    batch_jaccard6 += jaccard_score(y6.cpu(), preds, average='macro') * x.size(0)  # multiplying with batch size
                    class_counts6 += y6.sum(axis=0).detach().cpu()
                    logging.debug(f"Batch jaccard 6 regions: {batch_jaccard6}")

                # Showing Progress
                if self.no_progress_bar and self.args.evaluate:
                    logging.info('Batch: {}/{}'.format(num + 1, len(self.eval_loader)))

        # Showing Evaluating Results
        eval_loss = sum(eval_loss) / len(eval_loss)
        eval_time = time() - start_eval_time
        eval_speed = len(self.eval_loader) * self.eval_batch_size / eval_time / len(self.args.gpus)

        if self.args.dataset_args['subset'] == 'binary':
            acc_top1 = num_top1 / num_sample
            bacc_top1 = float(np.average(num_top1_perclass / num_sample_perclass))
            logging.info('Top-1 accuracy: {:d}/{:d}({:.2f}), Balanced accuracy: {:.2f},Mean loss:{:.4f}'.format(
                num_top1, num_sample, acc_top1, bacc_top1, eval_loss
            ))
            if self.scalar_writer:
                self.scalar_writer.add_scalar('eval_acc', acc_top1, self.global_step)
                self.scalar_writer.add_scalar('eval_bacc', bacc_top1, self.global_step)
        elif self.args.dataset_args['subset'] == 'signature':
            test_jaccard21 = batch_jaccard21 / num_sample
            test_jaccard6 = batch_jaccard6 / num_sample
            logging.info('Test Jaccard index: 21 regions - {:.2f}, 6 regions - {:.2f}, Mean loss:{:.4f}'.format(
                test_jaccard21, test_jaccard6, eval_loss
            ))
            if self.scalar_writer:
                self.scalar_writer.add_scalar('eval_jaccard21', test_jaccard21, self.global_step)
                self.scalar_writer.add_scalar('eval_jaccard6', test_jaccard6, self.global_step)
        logging.info('Evaluating time: {:.2f}s, Speed: {:.2f} sequnces/(second*GPU)'.format(
            eval_time, eval_speed
        ))
        logging.info('')
        if self.scalar_writer:
            self.scalar_writer.add_scalar('eval_loss', eval_loss, self.global_step)

        torch.cuda.empty_cache()

        if self.args.dataset_args['subset'] == 'binary':
            if save_score:
                return bacc_top1, acc_top1, score
            else:
                return bacc_top1, acc_top1, cm
        elif self.args.dataset_args['subset'] == 'signature':
            if save_score:
                return test_jaccard21, test_jaccard6, score
            else:
                return test_jaccard21, test_jaccard6, cm

    def start(self):
        start_time = time()
        if self.args.evaluate:
            if self.args.debug:
                logging.warning('Warning: Using debug setting now!')
                logging.info('')

            # Loading Evaluating Model
            logging.info('Loading evaluating model ...')
            checkpoint = U.load_checkpoint(self.args.work_dir, self.model_name)
            if checkpoint:
                self.model.module.load_state_dict(checkpoint['model'])
            logging.info('Successful!')
            logging.info('')

            # Evaluating
            logging.info('Starting evaluating ...')
            self.eval()
            logging.info('Finish evaluating!')

        else:
            # Resuming
            start_epoch = 0
            best_state = {'acc_top1': 0, 'bacc_top1': 0, 'cm': 0, 'jaccard21': 0, 'jaccard6': 0, 'best_epoch': 0}
            if self.args.resume:
                logging.info('Loading checkpoint ...')
                checkpoint = U.load_checkpoint(self.args.work_dir)
                self.model.module.load_state_dict(checkpoint['model'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.scheduler.load_state_dict(checkpoint['scheduler'])
                start_epoch = checkpoint['epoch']
                best_state.update(checkpoint['best_state'])
                self.global_step = start_epoch * len(self.train_loader)
                logging.info('Start epoch: {}'.format(start_epoch + 1))
                if self.args.dataset_args['subset'] == 'binary':
                    logging.info('Best balanced accuracy: {:.2%}'.format(best_state['bacc_top1']))
                    logging.info('accuracy: {:.2%}'.format(best_state['acc_top1']))
                elif self.args.dataset_args['subset'] == 'signature':
                    logging.info('Best jaccard index: {:.2%}'.format(best_state['jaccard6']))
                logging.info('Successful!')
                logging.info('')

            # Training
            logging.info('Starting training ...')
            for epoch in range(start_epoch, self.max_epoch):

                # Training
                self.train(epoch)

                # Evaluating
                is_best = False
                if (epoch + 1) % self.eval_interval(epoch) == 0:
                    logging.info('Evaluating for epoch {}/{} ...'.format(epoch + 1, self.max_epoch))

                    if self.args.dataset_args['subset'] == 'binary':
                        bacc_top1, acc_top1, cm = self.eval()
                        if bacc_top1 > best_state['bacc_top1']:
                            is_best = True
                            best_state.update({'bacc_top1': bacc_top1, 'acc_top1': acc_top1, 'cm': cm, 'best_epoch': epoch + 1})
                    elif self.args.dataset_args['subset'] == 'signature':
                        jaccard21, jaccard6, cm = self.eval()
                        if jaccard6 > best_state['jaccard6']:
                            is_best = True
                            best_state.update({'jaccard21': jaccard21, 'jaccard6': jaccard6, 'cm': cm, 'best_epoch': epoch + 1})

                # Saving Model
                logging.info('Saving model for epoch {}/{} ...'.format(epoch + 1, self.max_epoch))
                U.save_checkpoint(
                    self.model.module.state_dict(), self.optimizer.state_dict(), self.scheduler.state_dict(),
                    epoch + 1, best_state, is_best, self.args.work_dir, self.save_dir, self.model_name
                )
                if self.args.dataset_args['subset'] == 'binary':
                    logging.info('Best balanced accuracy (accuracy): {:.2%} ({:.2%})@{}th epoch, Total time: {}'.format(
                        best_state['bacc_top1'], best_state['acc_top1'], best_state['best_epoch'], U.get_time(time() - start_time)
                    ))
                elif self.args.dataset_args['subset'] == 'signature':
                    logging.info('Best jaccard index: 6 regions {:.2%} @{}th epoch, Total time: {}'.format(
                        best_state['jaccard6'], best_state['best_epoch'], U.get_time(time() - start_time)
                    ))
                logging.info('')

            if self.args.dataset_args['subset'] == 'binary':
                np.savetxt('{}/cm.csv'.format(self.save_dir), cm, fmt="%s", delimiter=",")
            logging.info('Finish training!')
            logging.info('')

    def extract(self):
        logging.info('Starting extracting ...')
        if self.args.debug:
            logging.warning('Warning: Using debug setting now!')
            logging.info('')

        # Loading Model
        logging.info('Loading evaluating model ...')
        checkpoint = U.load_model(self.args.work_dir, self.model_name)
        if checkpoint:
            self.cm = checkpoint['best_state']['cm']
            self.model.module.load_state_dict(checkpoint['model'])
        logging.info('Successful!')
        logging.info('')

        # Loading Data
        # x, y, names = iter(self.eval_loader).next()
        # location = self.location_loader.load(names) if self.location_loader else []

        # Calculating Output
        bacc_top1, acc_top1, score = self.eval(save_score=True)
        # self.model.eval()
        # out, feature = self.model(x.float().to(self.device))

        # Processing Data
        # data, label = x.numpy(), y.numpy()
        # out = torch.nn.functional.softmax(out, dim=1).detach().cpu().numpy()
        # weight = self.model.module.classifier.fc.weight.squeeze().detach().cpu().numpy()
        # feature = feature.detach().cpu().numpy()

        # Saving Data
        if not self.args.debug:
            U.create_folder('./visualization')
            save_path = self.args.work_dir + '/' + 'score.npy'
            # np.savez('./visualization/extraction_{}.npz'.format(self.args.config),
            #     data=data, label=label, name=names, out=out, cm=self.cm,
            #     feature=feature, weight=weight, location=location
            # )
            with open(save_path, 'wb') as f:
                pickle.dump(score, f)
        logging.info('Finish extracting!')
        logging.info('')
