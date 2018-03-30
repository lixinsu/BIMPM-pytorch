import argparse
import json
from collections import defaultdict
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn import metrics
from prettytable import PrettyTable
from model.BIMPM import BIMPM
from model.utils import SNLI, Quora, Quasart


def auc_para(pred_scores, gts):
    """auc paragraph"""
    pred_scores = [y for x in pred_scores for y in x]
    gts = [y for x in gts for y in x]
    print("PARA: pos %s neg %s" % (sum(gts), len(gts)- sum(gts)))
    fpr, tpr, thresholds = metrics.roc_curve(gts, pred_scores, pos_label=1)
    precision, recall, thresholds = metrics.precision_recall_curve(gts, pred_scores)
    return metrics.auc(fpr, tpr), metrics.auc(recall, precision)


def auc_question(pred_scores, gts):
    """question level auc"""
    pred_scores =[max(x) for x in pred_scores]
    gts = [max(x) for x in gts]
    print("QUESTION: pos %s neg %s" % (sum(gts), len(gts)- sum(gts)))
    fpr, tpr, thresholds = metrics.roc_curve(gts, pred_scores, pos_label=1)
    precision, recall, thresholds = metrics.precision_recall_curve(gts, pred_scores)
    return metrics.auc(fpr, tpr), metrics.auc(recall, precision)


def test(model, args, data,  mode='test'):
    if mode == 'dev':
        data_file = '.data/%s/sample_val.json' % args.task.lower()
        iterator = iter(data.dev_iter)
    else:
        data_file = '.data/%s/sample_test.json' % args.task.lower()
        iterator = iter(data.test_iter)

    criterion = nn.CrossEntropyLoss()
    model.eval()
    acc, loss, size = 0, 0, 0
    preds = []
    gts = []
    pred_scores = []
    qids = []
    debug = 10
    for batch in iterator:
        if args.data_type == 'SNLI':
            s1, s2 = 'premise', 'hypothesis'
        else:
            s1, s2 = 'q1', 'q2'

        s1, s2 = getattr(batch, s1), getattr(batch, s2)
        kwargs = {'p': s1, 'h': s2}

        if args.use_char_emb:
            char_p = Variable(torch.LongTensor(data.characterize(s1)))
            char_h = Variable(torch.LongTensor(data.characterize(s2)))

            if args.gpu > -1:
                char_p = char_p.cuda(args.gpu)
                char_h = char_h.cuda(args.gpu)

            kwargs['char_p'] = char_p
            kwargs['char_h'] = char_h

        pred = model(**kwargs)

        batch_loss = criterion(pred, batch.label)
        loss += batch_loss.data[0]

        _, pred_label = pred.max(dim=1)
        acc += (pred_label == batch.label).sum().float()
        size += len(pred)
        pred = F.softmax(pred, dim=1)
        pred = pred.data.tolist()
        gt = batch.label.data.tolist()
        gts.extend(gt)
        pred_scores.extend([x[1] for x in pred])
        qids.extend(batch.id)
#        if debug ==0:
#            break
#        debug -= 1

    #--------------------------------------------------------------------------------
    # merge result by question
    qid2list = {}
    for iter_qid, iter_score, iter_gt  in zip(qids, pred_scores, gts):
        qid2list[iter_qid] = [iter_score, iter_gt]
    real_qids = [json.loads(line)['query_id'] for line in open(data_file)]
    qid2res, qid2gt = defaultdict(list), defaultdict(list)
    for iter_qid in real_qids:
        for i in range(10):
            qid2res[iter_qid].append( qid2list[ '%s-%s' % (iter_qid, i) ][0] )
            qid2gt[iter_qid].append( qid2list['%s-%s' % (iter_qid, i)][1] )
    pred_scores, gts = [], []
    for iter_qid in real_qids:
        pred_scores.append(qid2res[iter_qid])
        gts.append(qid2gt[iter_qid])
    #--------------------------------------------------------------------------------
    # print metrics
    p_roc, p_pr = auc_para(pred_scores, gts)
    q_roc, q_pr = auc_question(pred_scores, gts)
    table = PrettyTable(["p_roc", "p_pr", "q_roc", "q_pr"])
    table.add_row([p_roc, p_pr, q_roc, q_pr ])
    print(table)
    #--------------------------------------------------------------------------------
    # save predictions
    if args.data_type == 'Quasart':
        result_file = 'quasart.result'
    elif args.data_type == 'Searchqa':
        result_file = 'searchqa.result'
    else:
        result_file = 'tmp.result'
    outf = open(result_file, 'w')
    for qid in real_qids:
        outf.write(json.dumps({'query_id':qid, 'predictions':qid2res[qid], 'ground_truth':qid2gt[qid]}) + '\n')


def load_model(args, data):
    model = BIMPM(args, data)
    model.load_state_dict(torch.load(args.model_path))
    if args.gpu > -1:
        model.cuda(args.gpu)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--char-dim', default=20, type=int)
    parser.add_argument('--char-hidden-size', default=50, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--data-type', default='SNLI', help='available: SNLI or Quora')
    parser.add_argument('--epoch', default=10, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--hidden-size', default=100, type=int)
    parser.add_argument('--learning-rate', default=0.001, type=float)
    parser.add_argument('--num-perspective', default=20, type=int)
    parser.add_argument('--use-char-emb', default=False, action='store_true')
    parser.add_argument('--word-dim', default=300, type=int)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--task', type=str, default='Quasart')
    args = parser.parse_args()

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

    print('loading model...')
    model = load_model(args, data)

    test(model, args, data)

