import os
import sys
import time
import math
import random
import logging
import queue
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

DATASET_PATH = './sample_dataset'
DATASET_PATH = os.path.join(DATASET_PATH, 'train').replace('\\', '/')   # ./sample_dateset\train => ./sample_dataset/train

def train(model, total_batch_size, queue, criterion, optimizer, device, train_begin, train_loader_count,print_batch=5, teacher_forcing_ratio=1):
    total_loss = 0.
    total_num = 0
    total_dist = 0
    total_length = 0
    total_sent_num = 0
    batch = 0

    model.train()

    logger.info('train() start')

    begin = epoch_begin = time.time()

    while True:
        if queue.empty():
            logger.debug('queue is empty')

        feats, scripts, feat_lengths, script_lengths = queue.get()

        if feats.shape[0] == 0:
            # empty feats means closing one loader
            train_loader_count -= 1

            logger.debug('left train_loader: %d' % (train_loader_count))

            if train_loader_count == 0:
                break
            else:
                continue

        optimizer.zero_grad()

        feats = feats.to(device)
        scripts = scripts.to(device)

        src_len = scripts.size(1)
        target = scripts[:, 1:]

        batch_size = scripts.size(0)

        model.module.flatten_parameters()
        logit = model(feats, feat_lengths, scripts, teacher_forcing_ratio=teacher_forcing_ratio)
        # print(len(logit))

        logit = torch.stack(logit, dim=1).to(device)
        # print(logit.shape)

        # probs = nn.functional.log_softmax(logit, dim=2) ###### ctc

        y_hat = logit.max(-1)[1] # return max indices?
        # print('y_hat: ' + str(len(y_hat))) # y_hat: list[batchsize]
        # print('target: ' + str(len(target)))
        # for i in range(batch_size):
        #         y_hat_tensor = y_hat[i]
        #         target_tensor = target[i]
        #         if not torch.equal(y_hat_tensor, target_tensor):
        #             y_hat_char = ''
        #             target_char = ''
        #             for j in range(y_hat_tensor.size(0)):
        #                 y_hat_char += index2char[y_hat_tensor[j].item()]
        #                 target_char += index2char[target_tensor[j].item()]
        #             print('y_hat :' + y_hat_char)
        #             print('target_char :' + target_char)

        ########################################3 ctc
        # x_lst = list()
        # y_lst = list()

        # for i in range(batch_size):
        #     for j in range(y_hat.size(1)):
        #         if (y_hat[i][j] == 819):
        #             x_lst.append(j)
        #             break
        #         if (j == y_hat.size(1)-1):
        #             x_lst.append(j)

        # for i in range(batch_size):
        #     for j in range(scripts.size(1)):
        #         if (scripts[i][j] == 819):
        #             y_lst.append(j)
        #             break
        # xs = torch.tensor(x_lst, dtype=torch.int).to(device)
        # ys = torch.tensor(y_lst, dtype=torch.int).to(device)
        # batch_false_prob = logit.transpose(0,1).to(device)
        # batch_false_scripts = scripts.to(device)
        # loss = ctc(batch_false_prob, batch_false_scripts, xs, ys)
        ########################################3 ctc

        loss = criterion(logit.contiguous().view(-1, logit.size(-1)), target.contiguous().view(-1))
        total_loss += loss.item()
        total_num += sum(feat_lengths)

        display = random.randrange(0, 100) == 0
        dist, length = get_distance(target, y_hat, display=display)
        total_dist += dist
        total_length += length

        total_sent_num += target.size(0)

        loss.backward()
        optimizer.step()

        if batch % print_batch == 0:
            current = time.time()
            elapsed = current - begin
            epoch_elapsed = (current - epoch_begin) / 60.0
            train_elapsed = (current - train_begin) / 3600.0

            logger.info('batch: {:4d}/{:4d}, loss:{:.4f} cer: {:.2f}, elapsed: {:.2f}s {:.2f}m {:.2f}h'
                .format(batch,
                        #len(dataloader),
                        total_batch_size,
                        total_loss / total_num,
                        total_dist / total_length,
                        elapsed, epoch_elapsed, train_elapsed))
            begin = time.time()

            nsml.report(False,
                        step=train.cumulative_batch_count, train_step__loss=total_loss / total_num,
                        train_step__cer=total_dist/total_length)
        batch += 1
        train.cumulative_batch_count += 1

    logger.info('train() completed')
    return total_loss / total_num, total_dist / total_length


def evaluate(model, dataloader, queue, criterion, device):
    logger.info('evaluate() start')
    total_loss = 0.
    total_num = 0
    total_dist = 0
    total_length = 0
    total_sent_num = 0

    model.eval()

    with torch.no_grad():
        while True:
            feats, scripts, feat_lengths, script_lengths = queue.get()
            if feats.shape[0] == 0:
                break

            feats = feats.to(device)
            scripts = scripts.to(device)

            src_len = scripts.size(1)
            target = scripts[:, 1:]

            batch_size = scripts.size(0)

            model.module.flatten_parameters()
            logit = model(feats, feat_lengths, scripts, teacher_forcing_ratio=0.0)

            logit = torch.stack(logit, dim=1).to(device)

            probs = nn.functional.softmax(logit, dim=2).max(-1)[0]

            y_hat = logit.max(-1)[1]


            for i in range(batch_size):
                y_hat_tensor = y_hat[i]
                target_tensor = target[i]
                if not torch.equal(y_hat_tensor, target_tensor):
                    y_hat_char = ''
                    target_char = ''
                    probs_dict = dict()
                    for j in range(y_hat_tensor.size(0)):
                        y_hat_char += index2char[y_hat_tensor[j].item()]
                        target_char += index2char[target_tensor[j].item()]
                        probs_dict[(index2char[y_hat_tensor[j].item()], index2char[target_tensor[j].item()])] = probs[i][j].item()
                        # for k in threshold_lst:
                        #     if (probs[i][j] < k):
                        #         matrix[k][0] += 1
                        #         if(index2char[y_hat_tensor[j].item()] == index2char[target_tensor[j].item()]):
                        #             matrix[k][1] += 1
                            
                    print('y_hat :' + y_hat_char)
                    print('target_char :' + target_char)
                    print(probs_dict)

            loss = criterion(logit.contiguous().view(-1, logit.size(-1)), target.contiguous().view(-1))
            total_loss += loss.item()
            total_num += sum(feat_lengths)

            display = random.randrange(0, 100) == 0
            dist, length = get_distance(target, y_hat, display=display)
            total_dist += dist
            total_length += length
            total_sent_num += target.size(0)

    logger.info('evaluate() completed')
    return total_loss / total_num, total_dist / total_length

def split_dataset(config, wav_paths, script_paths, wav_path_len, valid_ratio=0.05):
    train_loader_count = config.workers
    records_num = len(wav_paths)
    batch_num = math.ceil(records_num / config.batch_size)

    valid_batch_num = math.ceil(batch_num * valid_ratio)
    train_batch_num = batch_num - valid_batch_num

    batch_num_per_train_loader = math.ceil(train_batch_num / config.workers)

    train_begin = 0
    train_end_raw_id = 0
    train_dataset_list = list()

    for i in range(config.workers):

        train_end = min(train_begin + batch_num_per_train_loader, train_batch_num)

        train_begin_raw_id = train_begin * config.batch_size
        train_end_raw_id = train_end * config.batch_size

        train_begin = train_end 

    valid_dataset = BaseDataset(wav_paths[train_end_raw_id:], script_paths[train_end_raw_id:], SOS_token, EOS_token)

    wav_path_len = wav_path_len[0:train_end_raw_id]
    wav_temp = sorted(wav_path_len, key = lambda wav_path_len: wav_path_len[1] )
    wav_paths_train = list()
    script_paths_train = list()
    for i in wav_temp:
        wav_paths_train.append(i[0])
    for i in wav_paths_train:
        script_paths_train.append( i.split('.')[0] + '.label')

    train_begin = 0
    train_end_raw_id = 0

    for i in range(config.workers):

        train_end = min(train_begin + batch_num_per_train_loader, train_batch_num)

        train_begin_raw_id = train_begin * config.batch_size
        train_end_raw_id = train_end * config.batch_size

        train_dataset_list.append(BaseDataset(
                                        wav_paths_train[train_begin_raw_id:train_end_raw_id],
                                        script_paths_train[train_begin_raw_id:train_end_raw_id],
                                        SOS_token, EOS_token))
        train_begin = train_end 

    # valid_dataset = BaseDataset(wav_paths[train_end_raw_id:], script_paths[train_end_raw_id:], SOS_token, EOS_token)

    return train_batch_num, train_dataset_list, valid_dataset

def get_distance():
    pass

def load(filename, **kwargs):
    state = torch.load(os.path.join(filename, 'model.pt'))
    model.load_state_dict(state['model'])
    if 'optimizer' in state and optimizer:
        optimizer.load_state_dict(state['optimizer'])
    print('Model loaded')

def save(filename, **kwargs):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, os.path.join(filename, 'model.pt'))

def infer(wav_path):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input = get_spectrogram_feature(wav_path)
    seq_len = input.size(0)
    max_seq_len = seq_len + ( 16 - (seq_len % 16))
    seq = torch.zeros(1, max_seq_len, 257)
    seq[0].narrow(0, 0, seq_len).copy_(input)
    input = seq.to(device)

    logit = model(input_variable=input, input_lengths=None, teacher_forcing_ratio=0)
    logit = torch.stack(logit, dim=1).to(device)

    y_hat = logit.max(-1)[1]
    hyp = label_to_string(y_hat)

    return hyp[0]

def main():
    ##
    seed = 0
    cuda = torch.cuda.is_available()
    ##
    logger = logging.getLogger('nagi')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    stream_handler = logging.StreamHandler()
    logger.addHandler(stream_handler)
    # stream_handler.setFormatter(formatter)
    file_handler = logging.StreamHandler('somename.log')
    logger.addHandler(file_handler)
    # file_handler.setFormatter(formatter)
    ##

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device('cuda' if cuda else 'cpu')

    model = CNN() ####
    model.flatten_parameters()
    model = model.to(device)

    optimizer = optim.Adam(model.module.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=PAD_token).to(device)

    data_list = os.path.join(DATASET_PATH, 'train_data', 'data_list.csv').replace('\\', '/')
    read_data_list()

    best_loss = 1e10
    begin_epoch = 0

    target_path = os.path.join(DATASET_PATH, 'train_label').replace('\\', '/')
    load_targets(target_path)

    train_batch_num, train_dataset_list, valid_dataset = split_dataset(wav_paths, script_paths, wav_path_len, valid_ratio=0.05)

    logger.info('start')
    train_begin = time.time()

    for epoch in range(begin_epoch, max_epochs):

        train_queue = queue.Queue(workers * 2)

        train_loader = MultiLoader(train_dataset_list, train_queue, batch_size, workers)
        train_loader.start()

        train_loss, train_cer = train(model, train_batch_num, train_queue, criterion, optimizer, device, train_begin, workers, 10, teacher_forcing)
        logger.info('Epoch %d (Training) Loss %0.4f CER %0.4f' % (epoch, train_loss, train_cer))

        train_loader.join()

        valid_queue = queue.Queue(workers * 2)
        valid_loader = BaseDataLoader(valid_dataset, valid_queue, batch_size, 0)
        valid_loader.start()

        eval_loss, eval_cer = evaluate(model, valid_loader, valid_queue, criterion, device)
        logger.info('Epoch %d (Evaluate) Loss %0.4f CER %0.4f' % (epoch, eval_loss, eval_cer))

        valid_loader.join()

    