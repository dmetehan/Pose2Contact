import os, sys, shutil, logging, json, torch
from time import time, strftime, localtime


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def get_time(total_time):
    s = int(total_time % 60)
    m = int(total_time / 60) % 60
    h = int(total_time / 60 / 60) % 24
    d = int(total_time / 60 / 60 / 24)
    return '{:0>2d}d-{:0>2d}h-{:0>2d}m-{:0>2d}s'.format(d, h, m, s)


def get_current_timestamp():
    ct = time()
    ms = int((ct - int(ct)) * 1000)
    return '[ {},{:0>3d} ] '.format(strftime('%Y-%m-%d %H:%M:%S', localtime(ct)), ms)


def load_checkpoint(work_dir, model_name='resume', subset='binary'):
    if model_name == 'resume':
        file_name = '{}/{}/checkpoint.pth.tar'.format(work_dir, subset)
    elif model_name == 'debug':
        file_name = '{}/{}/temp/debug.pth.tar'.format(work_dir, subset)
    else:
        dirs, accs = {}, {}
        # work_dir = '{}/{}'.format(work_dir, model_name)
        logging.info(f"work_dir: {work_dir}")
        if os.path.exists(work_dir):
            for i, dir_time in enumerate(os.listdir(work_dir)):
                if os.path.isdir('{}/{}'.format(work_dir, dir_time)):
                    state_file = '{}/{}/reco_results.json'.format(work_dir, dir_time)
                    if os.path.exists(state_file):
                        with open(state_file, 'r') as f:
                            best_state = json.load(f)
                        try:
                            accs[str(i + 1)] = best_state['acc_top1' if subset == 'binary' else 'jaccard42']
                            dirs[str(i + 1)] = dir_time
                        except KeyError:
                            print(f"Key error on {dir_time}")
                            continue
        if len(dirs) == 0:
            logging.warning('Warning: Do NOT exists any model in workdir!')
            logging.info('Evaluating initial or pretrained model.')
            return None
        logging.info('Please choose the evaluating model from the following models.')
        logging.info('Default is the initial or pretrained model.')
        for key in dirs.keys():
            logging.info('({}) accuracy: {:.2%} | training time: {}'.format(key, accs[key], dirs[key]))
        logging.info('Your choice (number of the model, q for quit): ')
        while True:
            idx = input(get_current_timestamp())
            if idx == '':
                logging.info('Evaluating initial or pretrained model.')
                return None
            elif idx in dirs.keys():
                break
            elif idx == 'q':
                logging.info('Quit!')
                sys.exit(1)
            else:
                logging.info('Wrong choice!')
        file_name = '{}/{}/{}.pth.tar'.format(work_dir, dirs[idx], model_name)
    try:
        checkpoint = torch.load(file_name, map_location=torch.device('cpu'))
    except:
        logging.info('')
        logging.error('Error: Wrong in loading this checkpoint: {}!'.format(file_name))
        raise ValueError()
    return checkpoint


def load_model(work_dir, model_name):
    if os.path.exists(work_dir):
        model_file = os.path.join(work_dir, model_name + '.pth.tar')
        if os.path.exists(model_file):
            try:
                checkpoint = torch.load(model_file, map_location=torch.device('cpu'))
            except:
                logging.info('')
                logging.error('Error: Wrong in loading this checkpoint: {}!'.format(model_file))
                raise ValueError()
        else:
            logging.info('')
            logging.error('Error: File {} does not exist!'.format(model_file))
            raise ValueError()
    else:
        logging.info('')
        logging.error('Error: {} does not exist!'.format(work_dir))
        raise ValueError()
    return checkpoint


def save_checkpoint(model, optimizer, scheduler, epoch, best_state, is_best, work_dir, save_dir, model_name):
    for key in model.keys():
        model[key] = model[key].cpu()
    checkpoint = {
        'model': model, 'optimizer': optimizer, 'scheduler': scheduler,
        'best_state': best_state, 'epoch': epoch,
    }
    cp_name = '{}/checkpoint.pth.tar'.format(work_dir)
    torch.save(checkpoint, cp_name)
    if is_best:
        shutil.copy(cp_name, '{}/{}.pth.tar'.format(save_dir, model_name))
        with open('{}/reco_results.json'.format(save_dir), 'w') as f:
            json.dump(best_state, f)


def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def set_logging(args):
    config = args.config.split('/')[1:]
    config[-1] = config[-1].replace('.yaml', '')
    config = '/'.join(config)
    if args.debug or args.evaluate or args.extract or args.generate_data:
        save_dir = '{}/temp'.format(args.work_dir)
    else:
        ct = strftime('%Y-%m-%d %H-%M-%S')
        save_dir = '{}/{}/{}'.format(args.work_dir, config, ct)
    create_folder(save_dir)
    log_format = '[ %(asctime)s ] %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG if args.debug else logging.INFO, format=log_format)
    handler = logging.FileHandler('{}/log.txt'.format(save_dir), mode='w', encoding='UTF-8')
    handler.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(handler)
    return save_dir


class CrossEntropyLabelSmooth(torch.nn.Module):
    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss
