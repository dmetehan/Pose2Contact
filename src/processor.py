import logging
import torch
import pickle
import numpy as np
import os
from torch.nn import functional
from sklearn.metrics import jaccard_score
from tqdm import tqdm
from time import time
import json

from . import utils as U
from .initializer import Initializer
from .visualization import visualize
from .model.baseline import get_baseline_predictors, predict_with_predictors


class Processor(Initializer):

    def train(self, epoch):
        self.model.train()
        timer = dict(start_time=time(), curr_time=time(), end_time=time(), dataloader=0.001, model=0.001, statistics=0.001)
        buffer_perclass = [(torch.ones(self.train_batch_size) * i).to(self.device) for i in range(self.num_class)]
        num_top1, num_sample, num_top1_perclass, num_sample_perclass = 0, 0, torch.zeros(self.num_class), torch.zeros(self.num_class)
        pred_scores42, pred_scores12, pred_scores21x21, pred_scores6x6 = [], [], [], []
        all_labels42, all_labels12, all_labels21x21, all_labels6x6 = [], [], [], []
        batch_jaccard42, batch_jaccard12 = 0, 0
        batch_jaccard21x21, batch_jaccard6x6 = 0, 0
        train_iter = self.train_loader if self.no_progress_bar else tqdm(self.train_loader, dynamic_ncols=True)
        for num, (x, y, meta) in enumerate(train_iter):
            self.optimizer.zero_grad()

            # Using GPU
            x = x.float().to(self.device)
            if self.args.dataset_args['subset'] == 'binary':
                y = y.long().to(self.device)
            elif self.args.dataset_args['subset'] == 'signature':
                y42 = y[0].long().to(self.device)
                y12 = y[1].long().to(self.device)
                y21x21 = y[2].long().to(self.device)
                y6x6 = y[3].long().to(self.device)
                all_labels42 += y42.detach().cpu().tolist()
                all_labels12 += y12.detach().cpu().tolist()
                all_labels21x21 += y21x21.detach().cpu().tolist()
                all_labels6x6 += y6x6.detach().cpu().tolist()
            timer['dataloader'] += time() - timer['curr_time']
            timer['curr_time'] = time()

            # Calculating Output
            out, _ = self.model(x)

            # Updating Weights
            if self.args.dataset_args['subset'] == 'binary':
                loss = self.loss_func(out[-1], functional.one_hot(y, 2).float())
            elif self.args.dataset_args['subset'] == 'signature':
                out42, out12, out21x21, out6x6 = out
                loss1 = self.loss_weights['42'] * self.loss_func(out42, y42.float())
                loss2 = self.loss_weights['12'] * self.loss_func(out12, y12.float())
                loss3 = self.loss_weights['21x21'] * self.loss_func(out21x21, y21x21.float())
                loss4 = self.loss_weights['6x6'] * self.loss_func(out6x6, y6x6.float())
                loss = loss1 + loss2 + loss3 + loss4
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.global_step += 1

            timer['model'] += time() - timer['curr_time']
            timer['curr_time'] = time()

            num_sample += x.size(0)
            if self.args.dataset_args['subset'] == 'binary':
                # Calculating Recognition Accuracies
                reco_top1 = out[-1].max(1)[1]
                num_top1 += reco_top1.eq(y).sum().item()

                # Calculating Balanced Accuracy
                for i in range(self.num_class):
                    num_top1_perclass[i] += (reco_top1.eq(buffer_perclass[i][:len(reco_top1)]) & (y.eq(buffer_perclass[i][:len(reco_top1)]))).sum().item()
                    num_sample_perclass[i] += y.eq(buffer_perclass[i][:len(y)]).sum().item()
            elif self.args.dataset_args['subset'] == 'signature':
                pred_scores42 += torch.sigmoid(out42.detach().cpu()).tolist()
                pred_scores12 += torch.sigmoid(out12.detach().cpu()).tolist()
                pred_scores21x21 += torch.sigmoid(out21x21.detach().cpu()).tolist()
                pred_scores6x6 += torch.sigmoid(out6x6.detach().cpu()).tolist()

                # Calculating Jaccard Index
                preds = torch.sigmoid(out42.detach().cpu()) > self.multilabel_thresh['42']
                batch_jaccard42 += jaccard_score(y42.cpu(), preds, average='micro') * x.size(0)  # multiplying with batch size
                logging.debug(f"Batch jaccard 42: {batch_jaccard42}")

                preds = torch.sigmoid(out12.detach().cpu()) > self.multilabel_thresh['12']
                batch_jaccard12 += jaccard_score(y12.cpu(), preds, average='micro') * x.size(0)  # multiplying with batch size
                logging.debug(f"Batch jaccard 12: {batch_jaccard12}")

                preds = torch.sigmoid(out21x21.detach().cpu()) > self.multilabel_thresh['21x21']
                batch_jaccard21x21 += jaccard_score(y21x21.cpu(), preds, average='micro') * x.size(0)  # multiplying with batch size
                logging.debug(f"Batch jaccard 21x21: {batch_jaccard21x21}")

                preds = torch.sigmoid(out6x6.detach().cpu()) > self.multilabel_thresh['6x6']
                batch_jaccard6x6 += jaccard_score(y6x6.cpu(), preds, average='micro') * x.size(0)  # multiplying with batch size
                logging.debug(f"Batch jaccard 6x6: {batch_jaccard6x6}")

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
            # Visualize thresholding graphs
            all_labels = {'42': all_labels42, '12': all_labels12, '21x21': all_labels21x21, '6x6': all_labels6x6}
            pred_scores = {'42': pred_scores42, '12': pred_scores12, '21x21': pred_scores21x21, '6x6': pred_scores6x6}
            kwargs = {'epoch': epoch, 'save_dir': os.path.join(self.save_dir, 'thresholding', 'train'), 'average': 'micro'}
            visualize.vis_threshold_eval(all_labels, pred_scores, jaccard_score, **kwargs)
            
            # Showing Train Results
            train_jaccard42 = batch_jaccard42 / num_sample
            train_jaccard12 = batch_jaccard12 / num_sample
            train_jaccard21x21 = batch_jaccard21x21 / num_sample
            train_jaccard6 = batch_jaccard6x6 / num_sample
            if self.scalar_writer:
                self.scalar_writer.add_scalar('train_jaccard42', train_jaccard42, self.global_step)
                self.scalar_writer.add_scalar('train_jaccard12', train_jaccard12, self.global_step)
                self.scalar_writer.add_scalar('train_jaccard21x21', train_jaccard21x21, self.global_step)
                self.scalar_writer.add_scalar('train_jaccard6x6', train_jaccard6, self.global_step)
            logging.info('Epoch: {}/{}, Training jaccard: 42: {:.2%}, 12: {:.2%}, 21x21: {:.2%}, 6x6: {:.2%} Training time: {:.2f}s'.format(
                epoch + 1, self.max_epoch, train_jaccard42, train_jaccard12, train_jaccard21x21, train_jaccard6, timer['total']
            ))
        logging.info('Dataloader: {:.2f}s({:.1f}%), Network: {:.2f}s({:.1f}%), Statistics: {:.2f}s({:.1f}%)'.format(
            timer['dataloader'], timer['dataloader'] / timer['total'] * 100, timer['model'], timer['model'] / timer['total'] * 100,
            timer['statistics'], timer['statistics'] / timer['total'] * 100
        ))
        logging.info('')

    def get_all_train_labels(self):
        all_labels42, all_labels12, all_labels21x21, all_labels6x6 = [], [], [], []
        for num, (x, y, _) in enumerate(self.train_loader):
                y42 = y[0].long().to(self.device)
                y12 = y[1].long().to(self.device)
                y21x21 = y[2].long().to(self.device)
                y6x6 = y[3].long().to(self.device)
                all_labels42 += y42.detach().cpu().tolist()
                all_labels12 += y12.detach().cpu().tolist()
                all_labels21x21 += y21x21.detach().cpu().tolist()
                all_labels6x6 += y6x6.detach().cpu().tolist()
        return {'42': all_labels42, '12': all_labels12, '21x21': all_labels21x21, '6x6': all_labels6x6}

    def eval(self, epoch=None, save_score=False):
        self.model.eval()
        start_eval_time = time()
        score = {}
        with torch.no_grad():
            num_top1 = 0
            num_sample, eval_loss = 0, []
            buffer_perclass = [(torch.ones(self.eval_batch_size) * i).to(self.device) for i in range(self.num_class)]
            num_top1_perclass, num_sample_perclass = torch.zeros(self.num_class), torch.zeros(self.num_class)
            cm = np.zeros((self.num_class, self.num_class))
            pred_scores42, pred_scores12, pred_scores21x21, pred_scores6x6 = [], [], [], []
            all_labels42, all_labels12, all_labels21x21, all_labels6x6 = [], [], [], []
            all_subjects = []
            all_meta = []
            eval_iter = self.eval_loader if self.no_progress_bar else tqdm(self.eval_loader, dynamic_ncols=True)
            for num, (x, y, meta) in enumerate(eval_iter):
                all_subjects += meta[0]
                all_meta.append(meta)
                # Using GPU
                x = x.float().to(self.device)
                if self.args.dataset_args['subset'] == 'binary':
                    y = y.long().to(self.device)
                elif self.args.dataset_args['subset'] == 'signature':
                    y42 = y[0].long().to(self.device)
                    y12 = y[1].long().to(self.device)
                    y21x21 = y[2].long().to(self.device)
                    y6x6 = y[3].long().to(self.device)
                    all_labels42 += y42.detach().cpu().tolist()
                    all_labels12 += y12.detach().cpu().tolist()
                    all_labels21x21 += y21x21.detach().cpu().tolist()
                    all_labels6x6 += y6x6.detach().cpu().tolist()

                # Calculating Output
                out, _ = self.model(x)

                # Getting Loss
                if self.args.dataset_args['subset'] == 'binary':
                    loss = self.loss_func(out[-1], functional.one_hot(y, 2).float())
                elif self.args.dataset_args['subset'] == 'signature':
                    out42, out12, out21x21, out6x6 = out
                    loss1 = self.loss_weights['42'] * self.loss_func(out42, y42.float())
                    loss2 = self.loss_weights['12'] * self.loss_func(out12, y12.float())
                    loss3 = self.loss_weights['21x21'] * self.loss_func(out21x21, y21x21.float())
                    loss4 = self.loss_weights['6x6'] * self.loss_func(out6x6, y6x6.float())
                    loss = loss1 + loss2 + loss3 + loss4
                eval_loss.append(loss.item())

                if save_score:
                    for n, c in zip("", out.detach().cpu().numpy()):
                        score[n] = c

                num_sample += x.size(0)
                if self.args.dataset_args['subset'] == 'binary':
                    # Calculating Recognition Accuracies
                    reco_top1 = out[-1].max(1)[1]
                    num_top1 += reco_top1.eq(y).sum().item()

                    # Calculating Balanced Accuracy
                    for i in range(self.num_class):
                        num_top1_perclass[i] += (reco_top1.eq(buffer_perclass[i][:len(reco_top1)]) & (y.eq(buffer_perclass[i][:len(reco_top1)]))).sum().item()
                        num_sample_perclass[i] += y.eq(buffer_perclass[i][:len(y)]).sum().item()

                    # Calculating Confusion Matrix
                    for i in range(x.size(0)):
                        cm[y[i], reco_top1[i]] += 1
                elif self.args.dataset_args['subset'] == 'signature':
                    pred_scores42 += torch.sigmoid(out42.detach().cpu()).tolist()
                    pred_scores12 += torch.sigmoid(out12.detach().cpu()).tolist()
                    pred_scores21x21 += torch.sigmoid(out21x21.detach().cpu()).tolist()
                    pred_scores6x6 += torch.sigmoid(out6x6.detach().cpu()).tolist()

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
            all_eval_labels = {'42': all_labels42, '12': all_labels12, '21x21': all_labels21x21, '6x6': all_labels6x6}
            if epoch is not None:
                # Visualize thresholding graphs
                pred_scores = {'42': pred_scores42, '12': pred_scores12, '21x21': pred_scores21x21, '6x6': pred_scores6x6}
                kwargs = {'epoch': epoch, 'save_dir': os.path.join(self.save_dir, 'thresholding', 'eval'), 'average': 'micro'}
                visualize.vis_threshold_eval(all_eval_labels, pred_scores, jaccard_score, **kwargs)
            else:
                all_train_labels = self.get_all_train_labels()
                all_baseline_clfs = get_baseline_predictors(all_train_labels)
                kwargs = {'average': 'micro'}
                all_baseline_results, best_baseline_results = predict_with_predictors(all_eval_labels, all_baseline_clfs, jaccard_score, **kwargs)

                logging.info('Baseline Jaccard: 42: {:.2%} ({}), 12: {:.2%} ({}), 21x21: {:.2%} ({}), 6x6: {:.2%} ({})'.format(
                    best_baseline_results['42'][0], best_baseline_results['42'][1],
                    best_baseline_results['12'][0], best_baseline_results['12'][1],
                    best_baseline_results['21x21'][0], best_baseline_results['21x21'][1],
                    best_baseline_results['6x6'][0], best_baseline_results['6x6'][1]
                ))
                # Visualize prediction errors as heatmaps for 21 region segmentation predictions
                preds42 = torch.Tensor(pred_scores42) > self.multilabel_thresh['42']
                visualize.vis_pred_errors_heatmap(all_labels42, preds42, os.path.join(self.save_dir, 'prediction_errors'))

            jaccard_avg = 'micro'
            preds42 = torch.Tensor(pred_scores42) > self.multilabel_thresh['42']
            test_jaccard42 = jaccard_score(all_labels42, preds42, average=jaccard_avg)
            preds12 = torch.Tensor(pred_scores12) > self.multilabel_thresh['12']
            test_jaccard12 = jaccard_score(all_labels12, preds12, average=jaccard_avg)
            preds21x21 = torch.Tensor(pred_scores21x21) > self.multilabel_thresh['21x21']
            test_jaccard21x21 = jaccard_score(all_labels21x21, preds21x21, average=jaccard_avg)
            preds6x6 = torch.Tensor(pred_scores6x6) > self.multilabel_thresh['6x6']
            test_jaccard6x6 = jaccard_score(all_labels6x6, preds6x6, average=jaccard_avg)

            all_preds = {'42': preds42.long().tolist(), '12': preds12.long().tolist(), '21x21': preds21x21.long().tolist(), '6x6': preds6x6.long().tolist()}
            kwargs = {'average': 'samples'}
            # visualize.vis_touch_region_counts(all_eval_labels, all_preds, all_subjects, jaccard_score, self.save_dir, **kwargs)
            # visualize.vis_per_sample_score(all_eval_labels, all_preds, all_subjects, jaccard_score, self.save_dir, **kwargs)
            # visualize.vis_per_setting_score(all_eval_labels, all_preds, all_meta, jaccard_score, self.save_dir, **kwargs)

            # visualize.vis_box_and_whiskers_per_setting_score(all_eval_labels, all_preds, all_meta, jaccard_score, self.save_dir, **kwargs)

            save_preds = {'preds': all_preds,
                          'labels': all_eval_labels,
                          'scores': {'42': pred_scores42, '12': pred_scores12, '21x21': pred_scores21x21, '6x6': pred_scores6x6},
                          'metadata': all_meta}
            json.dump(save_preds, open(os.path.join(self.save_dir, "save_preds.json"), 'w'))

            logging.info('Test Jaccard: 42: {:.2%}, 12: {:.2%}, 21x21: {:.2%}, 6x6: {:.2%}, Mean loss:{:.4f}'.format(
                test_jaccard42, test_jaccard12, test_jaccard21x21, test_jaccard6x6, eval_loss
            ))
            if self.scalar_writer:
                self.scalar_writer.add_scalar('eval_jaccard42', test_jaccard42, self.global_step)
                self.scalar_writer.add_scalar('eval_jaccard12', test_jaccard12, self.global_step)
                self.scalar_writer.add_scalar('eval_jaccard21x21', test_jaccard21x21, self.global_step)
                self.scalar_writer.add_scalar('eval_jaccard6x6', test_jaccard6x6, self.global_step)
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
                return bacc_top1, acc_top1, cm.tolist()
        elif self.args.dataset_args['subset'] == 'signature':
            if save_score:
                return test_jaccard42, test_jaccard12, test_jaccard21x21, test_jaccard6x6, score
            else:
                return test_jaccard42, test_jaccard12, test_jaccard21x21, test_jaccard6x6

    def start(self):
        start_time = time()
        if self.args.evaluate:
            if self.args.debug:
                logging.warning('Warning: Using debug setting now!')
                logging.info('')

            # Loading Evaluating Model
            print(self.args.work_dir)
            print(self.model_name)
            logging.info('Loading evaluating model ...')
            checkpoint = U.load_checkpoint(self.args.work_dir, self.model_name, self.args.dataset_args['subset'])
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
            if self.args.dataset_args['subset'] == 'binary':
                best_state = {'bacc_top1': 0, 'acc_top1': 0, 'cm': 0, 'best_epoch': 0}
            else:
                best_state = {'jaccard42': 0, 'jaccard12': 0, 'jaccard21x21': 0, 'jaccard6x6': 0, 'best_epoch': 0}
            if self.args.resume:
                logging.info('Loading checkpoint ...')
                print(self.args.work_dir)
                print(self.model_name)
                checkpoint = U.load_checkpoint(self.args.work_dir, self.args.dataset_args['subset'])
                self.model.module.load_state_dict(checkpoint['model'])
                # self.optimizer.load_state_dict(checkpoint['optimizer'])
                # self.scheduler.load_state_dict(checkpoint['scheduler'])
                # start_epoch = checkpoint['epoch']
                best_state.update(checkpoint['best_state'])
                self.global_step = start_epoch * len(self.train_loader)
                logging.info('Start epoch: {}'.format(start_epoch + 1))
                if self.args.dataset_args['subset'] == 'binary':
                    logging.info('Best balanced accuracy: {:.2%}'.format(best_state['bacc_top1']))
                    logging.info('accuracy: {:.2%}'.format(best_state['acc_top1']))
                elif self.args.dataset_args['subset'] == 'signature':
                    logging.info('Best jaccard index: {:.2%}'.format(best_state['jaccard21x21']))
                logging.info('Successful!')
                logging.info('')

            logging.info(f'Training size: {len(self.train_loader.dataset)}, Test size: {len(self.eval_loader.dataset)}')
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
                        bacc_top1, acc_top1, cm = self.eval(epoch=epoch)
                        if bacc_top1 > best_state['bacc_top1']:
                            is_best = True
                            best_state.update({'bacc_top1': bacc_top1, 'acc_top1': acc_top1, 'cm': cm, 'best_epoch': epoch + 1})
                    elif self.args.dataset_args['subset'] == 'signature':
                        jaccard42, jaccard12, jaccard21x21, jaccard6x6 = self.eval(epoch=epoch)
                        if self.args.model_args['loss_weights']['21x21'] == 0:
                            if jaccard6x6 > best_state['jaccard6x6']:
                                print("best result in 6x6")
                                is_best = True
                                best_state.update({'jaccard42': jaccard42, 'jaccard12': jaccard12,
                                                   'jaccard21x21': jaccard21x21, 'jaccard6x6': jaccard6x6,
                                                   'best_epoch': epoch + 1})
                        else:
                            if jaccard21x21 > best_state['jaccard21x21']:
                                print("best result in 21x21")
                                is_best = True
                                best_state.update({'jaccard42': jaccard42, 'jaccard12': jaccard12,
                                                   'jaccard21x21': jaccard21x21, 'jaccard6x6': jaccard6x6,
                                                   'best_epoch': epoch + 1})

                # Saving Model
                logging.info('Saving model for epoch {}/{} ...'.format(epoch + 1, self.max_epoch))
                U.save_checkpoint(
                    self.model.module.state_dict(),
                    self.optimizer.state_dict(),
                    self.scheduler.state_dict(),
                    epoch + 1, best_state, is_best, self.args.work_dir, self.save_dir, self.model_name
                )
                if self.args.dataset_args['subset'] == 'binary':
                    logging.info('Best balanced accuracy (accuracy): {:.2%} ({:.2%})@{}th epoch, Total time: {}'.format(
                        best_state['bacc_top1'], best_state['acc_top1'], best_state['best_epoch'], U.get_time(time() - start_time)
                    ))
                elif self.args.dataset_args['subset'] == 'signature':
                    logging.info('Best jaccard index: 21x21 {:.2%} @{}th epoch, Total time: {}'.format(
                        best_state['jaccard21x21'], best_state['best_epoch'], U.get_time(time() - start_time)
                    ))
                    logging.info('Best Jaccard all results: 42: {:.2%}, 12: {:.2%}, 21x21: {:.2%}, 6x6: {:.2%}'.format(
                        best_state['jaccard42'], best_state['jaccard12'], best_state['jaccard21x21'],
                        best_state['jaccard6x6']
                    ))
                logging.info('')

            if self.args.dataset_args['subset'] == 'binary':
                np.savetxt('{}/cm.csv'.format(self.save_dir), cm, fmt="%s", delimiter=",")
            logging.info('Finish training!')
            logging.info('')
            return best_state

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
