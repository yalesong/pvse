#!/usr/bin/env python
import os, sys, json, glob
import random
import numpy as np
import progressbar as pb

from gulpio import GulpDirectory
from gulpio.fileio import GulpIngestor
from gulpio.adapters import AbstractDatasetAdapter
from gulpio.utils import (resize_images,
                          burst_video_into_frames,
                          temp_dir_for_bursting,
                          remove_entries_with_duplicate_ids,
                          ImageNotFound)

def get_uid(val):
  return val.strip().split('/')[-1].replace('.gif','')

def retrieve_nfrms_from_gulp(gulp_dir):
  id2nfrms = dict()
  gulp = GulpDirectory(gulp_dir)
  pbar = pb.ProgressBar(widgets=[pb.Percentage(), pb.Bar()], maxval=gulp.num_chunks).start()
  i = 0
  for chunk in gulp:
    for frames, meta in chunk:
      id2nfrms[meta['id']] = len(frames)
    pbar.update(i + 1)
    i += 1
  return id2nfrms

class GulpTGIFAdapter(AbstractDatasetAdapter):
  """Adapter for TGIF dataset specified by JSON file and gif videos."""

  def __init__(self, data, split_tsv, videos_dir, output_dir, shuffle=False,
                frame_size=-1, frame_rate=8, shm_dir_path='/dev/shm',
                label_name='template', remove_duplicate_ids=False):

    self.label_name = label_name
    self.videos_dir = videos_dir
    self.output_dir = output_dir
    self.shuffle = bool(shuffle)
    self.frame_size = int(frame_size)
    self.frame_rate = int(frame_rate)
    self.shm_dir_path = shm_dir_path

    valid_ids = [get_uid(line) for line in open(split_tsv, 'r').readlines()]
    self.all_meta = [{'id': item[0], 'sentence': item[1]} \
                       for item in data if item[0] in valid_ids]

    for meta in self.all_meta:
      gif_path = os.path.join(self.videos_dir, str(meta['id'] + '.gif'))
      if not os.path.exists(gif_path):
        print('{} missing!'.format(gif_path))

    if remove_duplicate_ids:
      self.all_meta = remove_entries_with_duplicate_ids(
          self.output_dir, self.all_meta)

    if self.shuffle:
      random.shuffle(self.all_meta)

  def iter_data(self, slice_element=None):
    slice_element = slice_element or slice(0, len(self))
    for meta in self.all_meta[slice_element]:
      video_path = os.path.join(self.videos_dir, str(meta['id'] + '.gif'))
      with temp_dir_for_bursting(self.shm_dir_path) as temp_burst_dir:
        frame_paths = burst_video_into_frames(
            video_path, temp_burst_dir, frame_rate=self.frame_rate)
        frames = list(resize_images(frame_paths, self.frame_size))
      result = {'meta': meta, 'frames': frames, 'id': meta['id']}
      yield result

  def __len__(self):
    return len(self.all_meta)


if __name__ == '__main__':
    data_file = 'tgif-v1.0-gulp.tsv'
    splits_dir = './split'
    videos_dir = './gifs'
    output_dir = './gulp'
    shm_dir = './shm'
    label_name = ''
    num_workers = 12
    frame_rate = 8
    frame_size = 256
    videos_per_chunk = 512
    remove_duplicates = True
    shuffle = True

    splits = ['valid','test','train']

    # Load the dataset
    data = [line.strip().split('\t') for line in open(data_file).readlines()]
    data_gulp = [[get_uid(item[0]), item[1]] for item in data]

    # GULP the dataset
    for split in splits:
      print('Gulping from [{}] split'.format(split))
      split_tsv = os.path.join(splits_dir, split+'.txt')
      output_dir_split = output_dir + '/{}'.format(split)
      shuffle = False if split != 'train' else shuffle
     
      adapter = GulpTGIFAdapter(data_gulp, split_tsv, videos_dir, output_dir_split,
                               shuffle=shuffle,
                               frame_size=frame_size,
                               frame_rate=frame_rate,
                               shm_dir_path=shm_dir,
                               label_name=label_name,
                               remove_duplicate_ids=remove_duplicates)
      ingestor = GulpIngestor(adapter, output_dir_split, videos_per_chunk, num_workers)
      ingestor()

    # Update TSV with num_frames from gulped data
    uid2nfrms = dict()
    for split in splits:
      uid2nfrms_split = retrieve_nfrms_from_gulp(os.path.join(output_dir, split))
      uid2nfrms.update(uid2nfrms_split)

    with open(data_file, 'w') as outfile:
      for item in data:
        outfile.write('{}\t{}\t{}\n'.format(item[0], item[1], \
          uid2nfrms[get_uid(item[0])] if get_uid(item[0]) in uid2nfrms else -1))
