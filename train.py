import os 
import sys
import math
import time
import shutil
import pickle
from lockfile import LockFile

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import numpy as np

import data
from vocab import Vocabulary
from model import PVSE
from loss import PVSELoss
from eval import i2t, t2i, encode_data
from logger import AverageMeter
from option import parser, verify_input_args

import logging
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO) 


def lock_and_write_to_file(filename, text):
  with LockFile(filename) as lock:
    with open(filename, 'a') as fid:
      fid.write('{}\n'.format(text))


def copy_input_args_from_ckpt(args, ckpt_args):
  args_to_copy = ['word_dim','crop_size','cnn_type','embed_size', 'num_embeds',
                  'img_attention','txt_attention','max_video_length']
  for arg in args_to_copy:
    val1, val2 = getattr(args, arg), getattr(ckpt_args, arg)
    if val1 != val2:
      logging.warning('Updating argument from checkpoint [{}]: [{}] --> [{}]'.format(arg, val1, val2))
      setattr(args, arg, val2)
  return args

def save_ckpt(state, is_best, filename='ckpt.pth.tar', prefix=''):
  torch.save(state, prefix + filename)
  if is_best:
    shutil.copyfile(prefix + filename, prefix + 'model_best.pth.tar')
    logging.info('Updating the best model checkpoint: {}'.format(prefix + 'model_best.pth.tar'))


def get_description(args, epoch=-1):
  return ('[{}][epoch:{}] {}'.format(args.logger_name.split('/')[-1], epoch, args))


def train(epoch, data_loader, model, criterion, optimizer, args):
  # switch to train mode
  model.train()

  # average meters to record the training statistics
  losses = AverageMeter()
  losses_dict = dict()
  losses_dict['ranking_loss'] = AverageMeter()
  if args.div_weight > 0:
    losses_dict['div_loss'] = AverageMeter()
  if args.mmd_weight > 0:
    losses_dict['mmd_loss'] = AverageMeter()

  for itr, data in enumerate(data_loader):
    img, txt, txt_len, _ = data
    if torch.cuda.is_available():
      img, txt, txt_len = img.cuda(), txt.cuda(), txt_len.cuda()

    # Forward pass and compute loss; _a: attention map, _r: residuals
    img_emb, txt_emb, img_a, txt_a, img_r, txt_r = model.forward(img, txt, txt_len)

    # Compute loss and update statstics
    loss, loss_dict = criterion(img_emb, txt_emb, img_r, txt_r)
    losses.update(loss.item())
    for key, val in loss_dict.items():
      losses_dict[key].update(val.item())

    # Backprop
    optimizer.zero_grad()
    loss.backward()
    if args.grad_clip > 0:
      nn.utils.clip_grad.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

    # Print log info
    if itr > 0 and (itr % args.log_step == 0 or itr + 1 == len(data_loader)):
      log_msg = 'loss: %.4f (%.4f)' %(losses.val, losses.avg)
      for key, val in losses_dict.items():
        log_msg += ', %s: %.4f, (%.4f)' %(key.replace('_loss',''), val.val, val.avg)
      n = int(math.ceil(math.log(len(data_loader) + 1, 10)))
      logging.info('[%d][%*d/%d] %s' %(epoch, n, itr, len(data_loader), log_msg))

  log_msg = 'loss: %.4f' %(losses.avg)
  for key, val in losses_dict.items():
    log_msg += ', %s: %.4f' %(key.replace('_loss',''), val.avg)
  exp_name = args.logger_name.split('/')[-1]
  lock_and_write_to_file(args.log_file, '[%s][%d] %s' %(exp_name, epoch, log_msg))

  del img_emb, txt_emb, img_a, txt_a, img_r, txt_r, loss
  return losses.avg
    

def validate(data_loader, model, args, epoch=-1, best_score=None):
  # switch to eval mode
  model.eval()

  nreps = 5 if 'coco' in args.data_name else 1
  order = args.order if hasattr(args, 'order') and args.order else False

  img_embs, txt_embs = encode_data(model, data_loader, args.eval_on_gpu)

  (r1, r5, r10, medr, meanr), (ranks, top1) = i2t(img_embs, txt_embs, 
      nreps=nreps, return_ranks=True, order=order, use_gpu=args.eval_on_gpu)

  (r1i, r5i, r10i, medri, meanri), (ranksi, top1i) = t2i(img_embs, txt_embs, 
      nreps=nreps, return_ranks=True, order=order, use_gpu=args.eval_on_gpu)

  # sum of recalls to be used for early stopping
  rsum = r1 + r5 + r10 + r1i + r5i + r10i
  med_rsum, mean_rsum = medr + medri, meanr + meanri

  # log
  exp_name = args.logger_name.split('/')[-1]
  vname = 'Video' if args.max_video_length>1 else 'Image'

  log_str1 = "[%s][%d] %s to text: %.2f, %.2f, %.2f, %.2f, %.2f" \
              %(exp_name, epoch, vname, r1, r5, r10, medr, meanr)
  log_str2 = "[%s][%d] Text to %s: %.2f, %.2f, %.2f, %.2f, %.2f" \
              %(exp_name, epoch, vname, r1i, r5i, r10i, medri, meanri)
  log_str3 = '[%s][%d] rsum: %.2f, med_rsum: %.2f, mean_rsum: %.2f' \
              %(exp_name, epoch, rsum, med_rsum, mean_rsum)
  if best_score:
    log_str3 += ' (best %s: %.2f)' %(args.val_metric, best_score)

  logging.info(log_str1)
  logging.info(log_str2)
  logging.info(log_str3)

  dscr = get_description(args, epoch)
  log_msg = '{}\n{}\n{}'.format(log_str1, log_str2, log_str3)
  lock_and_write_to_file(args.log_file, log_msg)

  if args.val_metric == 'rsum':
    return rsum
  elif args.val_metric == 'med_rsum':
    return med_rsum
  else:
    return mean_rsum


def update_best_score(new_score, old_score, is_higher_better):
  if not old_score:
    score, updated = new_score, True
  else:
    if is_higher_better:
      score = max(new_score, old_score)
      updated = new_score > old_score
    else:
      score = min(new_score, old_score)
      updated = new_score < old_score
  return score, updated
  

def main():
  multi_gpu = torch.cuda.device_count() > 1

  args = verify_input_args(parser.parse_args())
  if args.ckpt:
    ckpt = torch.load(args.ckpt)
    args = copy_input_args_from_ckpt(args, ckpt['args'])
  print(args)

  # Load Vocabulary Wrapper
  vocab_path = os.path.join(args.vocab_path, '%s_vocab.pkl' % args.data_name)
  vocab = pickle.load(open(vocab_path, 'rb'))

  # Dataloaders
  trn_loader, val_loader = data.get_loaders(args, vocab)
  val_loader = data.get_test_loader(args, vocab)

  # Construct the model
  model = PVSE(vocab.word2idx, args)
  if torch.cuda.is_available():
    model = nn.DataParallel(model).cuda() if multi_gpu else model.cuda()
    cudnn.benchmark = True

  # optionally resume from a ckpt
  if args.ckpt:
    target_vocab_path = './vocab/%s_vocab.pkl' % args.data_name
    src_vocab_path = './vocab/%s_vocab.pkl' % ckpt['args'].data_name
    if target_vocab_path != src_vocab_path:
      print('Vocab mismatch!')
      sys.exit(-1)
    model.load_state_dict(ckpt['model'])
    #validate(val_loader, model, args)

  # Loss and optimizer
  criterion = PVSELoss(args)
  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, 
      weight_decay=args.weight_decay, amsgrad=True)
  lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
      optimizer, factor=0.5, min_lr=1e-10, verbose=True)

  # Train the Model
  if args.ckpt and 'best_score' in ckpt and ckpt['args'].val_metric == args.val_metric:
    best_score = ckpt['best_score']
  else:
    best_score = None

  for epoch in range(args.num_epochs):
    # train for one epoch
    loss = train(epoch, trn_loader, model, criterion, optimizer, args)

    # evaluate on validation set
    val_score = validate(val_loader, model, args, epoch, best_score)

    # adjust learning rate if rsum stagnates
    lr_scheduler.step(val_score)

    # remember best rsum and save ckpt
    best_score, updated = update_best_score(val_score, best_score, 
                                            args.val_metric=='rsum')
    save_ckpt({
      'args': args,
      'epoch': epoch,
      'best_score': best_score,
      'model': model.state_dict(),
    }, updated, prefix=args.logger_name + '/')


if __name__ == '__main__':
  main()
