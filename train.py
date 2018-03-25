import sys
import argparse
import copy
import os
import logging
import torch
import numpy as np
from torch import nn, optim
from torch.autograd import Variable
from time import gmtime, strftime

from visdom import Visdom
from model.BIMPM import BIMPM
from model.utils import SNLI, Quora, Searchqa, Quasart
from test import test


logger = logging.getLogger()
viz = Visdom(port=8093)

def train(args, data):
    model = BIMPM(args, data)
    viz.line(X=np.array([0]),Y=np.array([0]),win=args.loss_curve, name='train-%s' % args.line_suffix, opts={'title': args.title})
    viz.line(X=np.array([0]),Y=np.array([0]),win=args.loss_curve, name='dev-%s' % args.line_suffix, update='append')
    viz.line(X=np.array([0]),Y=np.array([0]),win=args.loss_curve, name='test-%s' % args.line_suffix, update='append')
    viz.line(X=np.array([0]),Y=np.array([0]),win=args.acc_curve, name='test-%s' % args.line_suffix, opts={'title': args.title})
    viz.line(X=np.array([0]),Y=np.array([0]),win=args.acc_curve, name='dev-%s' % args.line_suffix, update='append')
    viz.line(X=np.array([0]),Y=np.array([0]),win=args.auc_curve, name='auc-test-%s' % args.line_suffix, opts={'title': args.title})
    viz.line(X=np.array([0]),Y=np.array([0]),win=args.auc_curve, name='auc-dev-%s' % args.line_suffix, update='append')

    if args.gpu > -1:
        model.cuda(args.gpu)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(parameters, lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()


    model.train()
    loss, last_epoch = 0, -1
    max_dev_auc, max_test_auc = 0, 0

    iterator = data.train_iter
    for i, batch in enumerate(iterator):
        present_epoch = int(iterator.epoch)
        if present_epoch == args.epoch:
            break
        if present_epoch > last_epoch:
            logger.info('epoch: %s' % ( present_epoch + 1 ))
        last_epoch = present_epoch

        if args.data_type == 'SNLI':
            s1, s2 = 'premise', 'hypothesis'
        else:
            s1, s2 = 'q1', 'q2'

        s1, s2 = getattr(batch, s1), getattr(batch, s2)

        # limit the lengths of input sentences up to max_sent_len
        if args.max_sent_len >= 0:
            if s1.size()[1] > args.max_sent_len:
                s1 = s1[:, :args.max_sent_len]
            if s2.size()[1] > args.max_sent_len:
                s2 = s2[:, :args.max_sent_len]

        kwargs = {'p': s1, 'h': s2}

        if args.use_char_emb:
            char_p = Variable(torch.LongTensor(data.characterize(s1)))
            char_h = Variable(torch.LongTensor(data.characterize(s2)))

            if args.gpu > -1 :
                char_p = char_p.cuda(args.gpu)
                char_h = char_h.cuda(args.gpu)

            kwargs['char_p'] = char_p
            kwargs['char_h'] = char_h
        pred = model(**kwargs)
        optimizer.zero_grad()
        batch_loss = criterion(pred, batch.label)
        loss += batch_loss.data[0]
        batch_loss.backward()
        optimizer.step()
        if (i + 1) % args.print_freq == 0:

            dev_loss, dev_acc, dev_auc_pr = test(model, args, data, mode='dev')
            test_loss, test_acc, test_auc_pr = test(model, args, data)
            c = (i + 1) // args.print_freq

            viz.line(X=np.array([c]),Y=np.array([loss]), win=args.loss_curve, name='train-%s' % args.line_suffix, update='append')
            viz.line(X=np.array([c]),Y=np.array([dev_loss]), win=args.loss_curve, name='dev-%s' % args.line_suffix, update='append')
            viz.line(X=np.array([c]),Y=np.array([test_loss]), win=args.loss_curve, name='test-%s' % args.line_suffix, update='append')
            viz.line(X=np.array([c]), Y=np.array([dev_acc]), win=args.acc_curve, name='dev-%s' % args.line_suffix , update='append')
            viz.line(X=np.array([c]), Y=np.array([test_acc]), win=args.acc_curve, name='test-%s' % args.line_suffix , update='append')
            viz.line(X=np.array([c]), Y=np.array([dev_auc_pr]), win=args.auc_curve, name='auc-dev-%s' % args.line_suffix , update='append')
            viz.line(X=np.array([c]), Y=np.array([test_auc_pr]), win=args.auc_curve, name='auc-test-%s' % args.line_suffix , update='append')
            logger.info('train loss: %.3f / dev loss: %.3f / test loss: %.3f' % (loss, dev_loss, test_loss))
            logger.info('dev acc: %.3f / test acc: %.3f' % (dev_acc, test_acc) )
            logger.info('dev auc of pr : %.3f , test auc pr : %.3f' % (dev_auc_pr, test_auc_pr))
            if  dev_auc_pr > max_dev_auc:
                max_dev_auc = dev_auc_pr
                max_test_auc = test_auc_pr
                best_model = copy.deepcopy(model)
                torch.save(best_model.state_dict(), 'saved_models/BIBPM_%s_%s.pt' % (args.data_type, args.model_time))

            loss = 0
            model.train()

    logger.info('max dev acc: %.3f / max test acc: %.3f' % (max_dev_auc, max_test_auc))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--char-dim', default=20, type=int)
    parser.add_argument('--char-hidden-size', default=50, type=int)
    parser.add_argument('--data-type', default='SNLI', help='available: SNLI or Quora')
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--epoch', default=10, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--hidden-size', default=100, type=int)
    parser.add_argument('--learning-rate', default=0.001, type=float)
    parser.add_argument('--max-sent-len', default=200, type=int,
                        help='max length of input sentences model can accept, if -1, it accepts any length')
    parser.add_argument('--num-perspective', default=20, type=int)
    parser.add_argument('--print-freq', default=500, type=int)
    parser.add_argument('--use-char-emb', default=False, action='store_true')
    parser.add_argument('--word-dim', default=300, type=int)
    parser.add_argument('--loss-curve', default='default_loss', type=str)
    parser.add_argument('--title', default='default', type=str)
    parser.add_argument('--acc-curve', default='default_acc', type=str)
    parser.add_argument('--auc-curve', default='default_auc', type=str)
    parser.add_argument('--line-suffix', default='tmp', type=str)
    parser.add_argument('--log-file', default='output.log', type=str, help='log file path')
    args = parser.parse_args()

    # Set logging
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    if args.log_file:
        logfile = logging.FileHandler(args.log_file, 'w')
        logfile.setFormatter(fmt)
        logger.addHandler(logfile)
    logger.info('COMMAND: %s' % ' '.join(sys.argv))
    if args.data_type == 'SNLI':
        print('loading SNLI data...')
        data = SNLI(args)
    elif args.data_type == 'Quora':
        print('loading Quora data...')
        data = Quora(args)
    elif args.data_type == 'Searchqa':
        print('loading Searchqa data...')
        data = Searchqa(args)
    elif args.data_type == 'Quasart':
        print('loading Quasart data...')
        data = Quasart(args)
    else:
        raise NotImplementedError('only SNLI or Quora data is possible')

    setattr(args, 'char_vocab_size', len(data.char_vocab))
    setattr(args, 'word_vocab_size', len(data.TEXT.vocab))
    setattr(args, 'class_size', len(data.LABEL.vocab))
    setattr(args, 'max_word_len', data.max_word_len)
    setattr(args, 'model_time', strftime('%H:%M:%S', gmtime()))

    logger.info('training start!')
    train(args, data)
    logger.info('training finished!')


if __name__ == '__main__':
    main()
