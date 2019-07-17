import os
import sys
import math
import random
import glob

import scipy
import numpy as np
import json as jsonmod

import video_transforms
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import nltk
from PIL import Image
from pycocotools.coco import COCO
from gulpio import GulpDirectory
from operator import itemgetter

import io, threading
_lock = threading.Lock()

MAX_RAND_FRAME_SKIP = 3

def get_uid_tgif(url):
  return url.strip().split('/')[-1].replace('.gif','')


def get_paths(path, name='coco', use_restval=True):
  """
  Returns paths to images and annotations for the given datasets. For MSCOCO
  indices are also returned to control the data split being used.
  The indices are extracted from the Karpathy et al. splits using this snippet:

  >>> import json
  >>> dataset=json.load(open('dataset_coco.json','r'))
  >>> A=[]
  >>> for i in range(len(D['images'])):
  ...   if D['images'][i]['split'] == 'val':
  ...   A+=D['images'][i]['sentids'][:5]
  ...

  :param name: Dataset names
  :param use_restval: If True, the `restval` data is included in train for COCO dataset.
  """
  roots, ids = {}, {}
  if 'coco' == name:
    imgdir = os.path.join(path, 'images')
    capdir = os.path.join(path, 'annotations')
    roots['train'] = {
      'img': os.path.join(imgdir, 'train2014'),
      'cap': os.path.join(capdir, 'captions_train2014.json'),
    }
    roots['val'] = {
      'img': os.path.join(imgdir, 'val2014'),
      'cap': os.path.join(capdir, 'captions_val2014.json'),
    }
    roots['test'] = {
      'img': os.path.join(imgdir, 'val2014'),
      'cap': os.path.join(capdir, 'captions_val2014.json'),
    }
    roots['trainrestval'] = {
      'img': (roots['train']['img'], roots['val']['img']),
      'cap': (roots['train']['cap'], roots['val']['cap']),
    }
    ids['train'] = np.load(os.path.join(capdir, 'coco_train_ids.npy'))
    ids['val'] = np.load(os.path.join(capdir, 'coco_dev_ids.npy'))[:5000]
    ids['test'] = np.load(os.path.join(capdir, 'coco_test_ids.npy'))
    ids['trainrestval'] = (ids['train'],
        np.load(os.path.join(capdir, 'coco_restval_ids.npy')))
    if use_restval:
      roots['train'] = roots['trainrestval']
      ids['train'] = ids['trainrestval']

  elif 'mrw' == name:
    cap = os.path.join(path, 'mrw-v1.0.json')
    gulp_dir = os.path.join(path, 'gulp')
    train_ids = [l.strip() for l in open(os.path.join(path, 'split/train.tsv')).readlines()]
    val_ids = [l.strip() for l in open(os.path.join(path, 'split/valid.tsv')).readlines()]
    test_ids = [l.strip() for l in open(os.path.join(path, 'split/test.tsv')).readlines()]
    roots['train'] = {'img': os.path.join(gulp_dir, 'train'), 'cap': cap}
    roots['val'] = {'img': os.path.join(gulp_dir, 'valid'), 'cap': cap}
    roots['test'] = {'img': os.path.join(gulp_dir, 'test'), 'cap': cap}
    ids = {'train': train_ids, 'val': val_ids, 'test': test_ids}

  elif 'tgif' == name:
    cap = os.path.join(path, 'tgif-v1.0-gulp.tsv')
    gulp_dir = os.path.join(path, 'gulp')
    train_ids = [get_uid_tgif(l) for l in \
        open(os.path.join(path, 'split/train.txt')).readlines()]
    val_ids = [get_uid_tgif(l) for l in \
        open(os.path.join(path, 'split/valid.txt')).readlines()]
    test_ids = [get_uid_tgif(l) for l in \
        open(os.path.join(path, 'split/test.txt')).readlines()]
    roots['train'] = {'img': os.path.join(gulp_dir, 'train'), 'cap': cap}
    roots['val'] = {'img': os.path.join(gulp_dir, 'valid'), 'cap': cap}
    roots['test'] = {'img': os.path.join(gulp_dir, 'test'), 'cap': cap}
    ids = {'train': train_ids, 'val': val_ids, 'test': test_ids}

  return roots, ids


class CocoDataset(data.Dataset):

  def __init__(self, root, json, vocab, transform=None, ids=None):
    """
    Args:
      root: image directory.
      json: coco annotation file path.
      vocab: vocabulary wrapper.
      transform: transformer for image.
    """
    self.root = root
    # when using `restval`, two json files are needed
    if isinstance(json, tuple):
      self.coco = (COCO(json[0]), COCO(json[1]))
    else:
      self.coco = (COCO(json),)
      self.root = (root,)

    # if ids provided by get_paths, use split-specific ids
    self.ids = list(self.coco.anns.keys()) if ids is None else ids
    self.vocab = vocab
    self.transform = transform

    # if `restval` data is to be used, record the break point for ids
    if isinstance(self.ids, tuple):
      self.bp = len(self.ids[0])
      self.ids = list(self.ids[0]) + list(self.ids[1])
    else:
      self.bp = len(self.ids)


  def __len__(self):
    return len(self.ids)


  def __getitem__(self, index):
    vocab = self.vocab
    root, sentence, img_id, path, image = self.get_raw_item(index)
    if self.transform is not None:
      image = self.transform(image)

    # Convert sentence (string) to word ids.
    if sys.version_info.major > 2:
      tokens = nltk.tokenize.word_tokenize(str(sentence).lower())
    else:
      tokens = nltk.tokenize.word_tokenize(str(sentence).lower().decode('utf-8'))

    sentence = []
    sentence.append(vocab('<start>'))
    sentence.extend([vocab(token) for token in tokens])
    sentence.append(vocab('<end>'))
    target = torch.Tensor(sentence)
    return image, target, index, img_id


  def get_raw_item(self, index):
    if index < self.bp:
      coco, root = self.coco[0], self.root[0]
    else:
      coco, root = self.coco[1], self.root[1]
    ann_id = self.ids[index]
    sentence = coco.anns[ann_id]['caption']
    img_id = coco.anns[ann_id]['image_id']
    path = coco.loadImgs(img_id)[0]['file_name']
    image = Image.open(os.path.join(root, path)).convert('RGB')
    return root, sentence, img_id, path, image



class MRWDataset(data.Dataset):

  def __init__(self, root, json, vocab, ids, transform=None, random_crop=False,
                max_video_len=8, max_sentence_len=24):
    self.root = root
    self.vocab = vocab
    self.ids = ids
    self.transform = transform
    self.random_crop = random_crop
    self.max_video_len = max_video_len
    self.max_sentence_len = max_sentence_len
    
    dataset = jsonmod.load(open(json, 'r'))
    self.dataset = dataset
    self.sentences = dict([(str(d['id']), d['sentence']) for d in dataset])
    self.num_frames = dict([(str(d['id']), d['gulp_num_frames']) for d in dataset])
    self.gulp = GulpDirectory(root)
    

  def __len__(self):
    return len(self.ids)


  def __getitem__(self, index):
    n_frms = self.num_frames[self.ids[index]]

    if self.random_crop:
      # Random subsequence sampling
      skip = random.randint(1,MAX_RAND_FRAME_SKIP)
      start_idx = random.randint(0, max(0, n_frms - skip * self.max_video_len))
      end_idx = start_idx + skip * self.max_video_len
    else:
      # Equal interval sampling
      buf_frm = 0
      start_idx = buf_frm
      end_idx = n_frms - buf_frm
      skip = max(1, (end_idx - start_idx + 1) // self.max_video_len)
    frames, meta = self.gulp[self.ids[index], start_idx:end_idx:skip]

    # video
    if len(frames) > self.max_video_len:
      if self.random_crop:
        sample_idx = random.sample(range(len(frames)), self.max_video_len)
        sample_idx.sort()
        frames = itemgetter(*sample_idx)(frames)
      else:
        frames = frames[:self.max_video_len]

    # Convert to PIL object
    frames = [Image.fromarray(frm) for frm in frames]

    # Apply transformation
    if self.max_video_len == 1:
      video = self.transform(frames[0]).unsqueeze(0)
    else:
      video = self.transform(frames)

    # Pad
    if len(frames) < self.max_video_len: 
      video_tmp = torch.empty(self.max_video_len, 3, 224, 224)
      video_tmp[:len(frames)] = video
      diff = self.max_video_len - len(frames)
      for i in range(self.max_video_len - len(frames)):
        video_tmp[-(i+1)] = video[-1]
      video = video_tmp

    if self.max_video_len == 1:
      video = video.squeeze(0)

    # sentence
    sentence = self.sentences[self.ids[index]]
    if sys.version_info.major > 2:
      tokens = nltk.tokenize.word_tokenize(str(sentence).lower())
    else:
      tokens = nltk.tokenize.word_tokenize(str(sentence).lower().decode('utf-8'))

    sentence = []
    sentence.append(self.vocab('<start>'))
    sentence.extend([self.vocab(tok) for tok in tokens])
    sentence.append(self.vocab('<end>'))
    sentence = torch.Tensor(sentence)
    return video, sentence, index, (self.ids[index], range(start_idx, end_idx+1, skip))



class TGIFDataset(data.Dataset):

  def __init__(self, root, tsv, vocab, ids, transform=None, random_crop=False,
               max_video_len=8, max_sentence_len=24):
    self.root = root
    self.vocab = vocab
    self.ids = ids
    self.transform = transform
    self.random_crop = random_crop
    self.max_video_len = max_video_len
    self.max_sentence_len = max_sentence_len

    dataset = [l.strip().split('\t') for l in open(tsv, 'r').readlines()]
    self.sentences = dict([[get_uid_tgif(l[0]), l[1]] for l in dataset])
    self.num_frames = dict([[get_uid_tgif(l[0]), int(l[2])] for l in dataset])
    self.gulp = GulpDirectory(root)


  def __len__(self):
    return len(self.ids)


  def __getitem__(self, index):
    """This function returns a tuple that is passed to collate_fn"""
    n_frms = self.num_frames[self.ids[index]]

    if self.random_crop:
      # Random subsequence sampling
      skip = random.randint(1,MAX_RAND_FRAME_SKIP)
      start_idx = random.randint(0, max(0, n_frms - skip * self.max_video_len))
      end_idx = start_idx + skip * self.max_video_len
    else:
      # Equal interval sampling
      buf_frm = 6
      start_idx = buf_frm
      end_idx = n_frms - buf_frm
      skip = max(1, (end_idx - start_idx + 1) // self.max_video_len)
    frames, meta = self.gulp[self.ids[index], start_idx:end_idx:skip]

    # video
    if len(frames) > self.max_video_len:
      if self.random_crop:
        sample_idx = random.sample(range(len(frames)), self.max_video_len)
        sample_idx.sort()
        frames = itemgetter(*sample_idx)(frames)
      else:
        frames = frames[:self.max_video_len]

    # Convert to PIL object
    frames = [Image.fromarray(frm) for frm in frames]

    # Apply transformation
    if self.max_video_len == 1:
      video = self.transform(frames[0]).unsqueeze(0)
    else:
      video = self.transform(frames)

    # Pad
    if len(frames) < self.max_video_len: 
      video_tmp = torch.empty(self.max_video_len, 3, 224, 224)
      video_tmp[:len(frames)] = video
      diff = self.max_video_len - len(frames)
      for i in range(self.max_video_len - len(frames)):
        video_tmp[-(i+1)] = video[-1]
      video = video_tmp

    if self.max_video_len == 1:
      video = video.squeeze(0)

    # sentence
    sentence = self.sentences[self.ids[index]]
    if sys.version_info.major > 2:
      tokens = nltk.tokenize.word_tokenize(
        str(sentence).lower())
    else:
      tokens = nltk.tokenize.word_tokenize(
        str(sentence).lower().decode('utf-8'))

    sentence = []
    sentence.append(self.vocab('<start>'))
    sentence.extend([self.vocab(tok) for tok in tokens])
    sentence.append(self.vocab('<end>'))
    sentence = torch.Tensor(sentence)
    return video, sentence, index, (self.ids[index], range(start_idx, end_idx+1, skip))

      
def collate_fn(data):
  """Build mini-batch tensors from a list of (image, sentence) tuples.
  Args:
    data: list of (image, sentence) tuple.
      - image: torch tensor of shape (3, 256, 256) or (?, 3, 256, 256).
      - sentence: torch tensor of shape (?); variable length.

  Returns:
    images: torch tensor of shape (batch_size, 3, 256, 256) or 
            (batch_size, padded_length, 3, 256, 256).
    targets: torch tensor of shape (batch_size, padded_length).
    lengths: list; valid length for each padded sentence.
  """
  # Sort a data list by sentence length
  data.sort(key=lambda x: len(x[1]), reverse=True)
  images, sentences, ids, img_ids = zip(*data)

  # Merge images (convert tuple of 3D tensor to 4D tensor)
  images = torch.stack(images, 0)

  # Merge sentences (convert tuple of 1D tensor to 2D tensor)
  cap_lengths = torch.tensor([len(cap) for cap in sentences])
  targets = torch.zeros(len(sentences), max(cap_lengths)).long()
  for i, cap in enumerate(sentences):
    end = cap_lengths[i]
    targets[i, :end] = cap[:end]

  return images, targets, cap_lengths, ids


def get_loader_single(data_name, split, root, json, vocab, transform,
                      batch_size=128, shuffle=True, num_workers=2, 
                      ids=None, collate_fn=collate_fn, opt=None):
  """Returns torch.utils.data.DataLoader for custom coco dataset."""
  if 'coco' in data_name:
    dataset = CocoDataset(root=root,
                          json=json,
                          vocab=vocab,
                          transform=transform, 
                          ids=ids)

  elif 'mrw' in data_name:
    dataset = MRWDataset(root=root,
                         json=json,
                         vocab=vocab,
                         ids=ids,
                         transform=transform,
                         random_crop=shuffle,
                         max_video_len=opt.max_video_length)

  elif 'tgif' in data_name:
    dataset = TGIFDataset(root=root, 
                          tsv=json, 
                          vocab=vocab, 
                          ids=ids, 
                          transform=transform,
                          random_crop=shuffle,
                          max_video_len=opt.max_video_length)

  # Data loader
  data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                            batch_size=batch_size,
                                            shuffle=shuffle,
                                            pin_memory=True,
                                            num_workers=num_workers,
                                            collate_fn=collate_fn)
  return data_loader


def get_transform(data_name, split_name, opt):
  if (data_name == 'mrw' or data_name == 'tgif') and opt.max_video_length > 1:
    return get_video_transform(data_name, split_name, opt)
  else:
    return get_image_transform(data_name, split_name, opt)


def get_video_transform(data_name, split_name, opt):
  normalizer = video_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])
  t_list = []
  if split_name == 'train':
    t_list = [video_transforms.RandomResizedCrop(opt.crop_size),
              video_transforms.RandomHorizontalFlip(),
              video_transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1)]
  else:
    t_list = [video_transforms.Resize(256),
              video_transforms.CenterCrop(opt.crop_size)]

  t_end = [video_transforms.ToTensor(), normalizer]
  transform = video_transforms.Compose(t_list + t_end)
  return transform


def get_image_transform(data_name, split_name, opt):
  normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
  t_list = []
  if split_name == 'train':
    t_list = [transforms.RandomResizedCrop(opt.crop_size)]
    if not (data_name == 'mrw' or data_name == 'tgif'):
      t_list += [transforms.RandomHorizontalFlip()]
  elif split_name == 'val':
    t_list = [transforms.Resize(256), transforms.CenterCrop(opt.crop_size)]
  elif split_name == 'test':
    t_list = [transforms.Resize(256), transforms.CenterCrop(opt.crop_size)]

  t_end = [transforms.ToTensor(), normalizer]
  transform = transforms.Compose(t_list + t_end)
  return transform


def get_loaders(opt, vocab):
  dpath = os.path.join(opt.data_path, opt.data_name)
  roots, ids = get_paths(dpath, opt.data_name)

  transform = get_transform(opt.data_name, 'train', opt)
  train_loader = get_loader_single(opt.data_name, 'train',
                   roots['train']['img'],
                   roots['train']['cap'],
                   vocab, transform, ids=ids['train'],
                   batch_size=opt.batch_size, shuffle=True,
                   num_workers=opt.workers,
                   collate_fn=collate_fn,
                   opt=opt)

  transform = get_transform(opt.data_name, 'val', opt)
  val_loader = get_loader_single(opt.data_name, 'val',
                   roots['val']['img'],
                   roots['val']['cap'],
                   vocab, transform, ids=ids['val'],
                   batch_size=opt.batch_size_eval, shuffle=False,
                   num_workers=opt.workers,
                   collate_fn=collate_fn,
                   opt=opt)

  return train_loader, val_loader


def get_test_loader(opt, vocab):
  dpath = os.path.join(opt.data_path, opt.data_name)
  roots, ids = get_paths(dpath, opt.data_name)
  transform = get_transform(opt.data_name, 'test', opt)
  return get_loader_single(opt.data_name, 'test',
                  roots['test']['img'],
                  roots['test']['cap'],
                  vocab, transform, ids=ids['test'],
                  batch_size=opt.batch_size_eval, shuffle=False,
                  num_workers=opt.workers,
                  collate_fn=collate_fn,
                  opt=opt)
