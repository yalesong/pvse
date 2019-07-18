#!/usr/bin/env python
import os, sys, json, glob
import random
import urllib.request
import numpy as np
import progressbar as pb

from subprocess import call
from multiprocessing import Pool

from gulpio import GulpDirectory
from gulpio.fileio import GulpIngestor
from gulpio.adapters import AbstractDatasetAdapter
from gulpio.utils import (resize_images,
                          burst_video_into_frames,
                          temp_dir_for_bursting,
                          remove_entries_with_duplicate_ids,
                          ImageNotFound)

def url_is_alive(url):
  request = urllib.request.Request(url)
  request.get_method = lambda: 'HEAD'
  try:
    urllib.request.urlopen(request)
    return True
  except urllib.request.HTTPError:
    return False

def _download_video(item):
  save_path = './mp4/{}.mp4'.format(item['id'])
  if not os.path.isfile(save_path):
    url = item['images']['mp4_url']
    if not url_is_alive(url):
      print('{} is no longer available!'.format(url))
    else:
      urllib.request.urlretrieve(url, save_path)

def download_videos(data, videos_dir, parallel=False, pool_sz=128):
  if parallel:
    pool = Pool(pool_sz)
    pool.map(_download_video, data)
    pool.close()
    pool.join()
  else:
    for item in data:
      _download_video(item)

def verify_downloaded_videos(data, videos_dir):
  missing = []
  for item in data:
    filepath = os.path.join(videos_dir, '{}.mp4'.format(item['id']))
    if not os.path.isfile(filepath):
      missing += item['id']
  if len(missing) > 0:
    raise Exception('Some of the video files are missing ({}). '
                    'Try downloading videos again or contact the author '
                    'at yalesong@csail.mit.edu')


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
      

class GulpMRWAdapter(AbstractDatasetAdapter):

  """Adapter for MRW dataset specified by JSON file and mp4 videos."""

  def __init__(self, json_file, split_tsv, videos_dir, output_dir, shuffle=False,
                frame_size=-1, frame_rate=8, shm_dir_path='/dev/shm',
                label_name='template', remove_duplicate_ids=False):

    self.label_name = label_name
    self.videos_dir = videos_dir
    self.output_dir = output_dir
    self.shuffle = bool(shuffle)
    self.frame_size = int(frame_size)
    self.frame_rate = int(frame_rate)
    self.shm_dir_path = shm_dir_path

    valid_ids = [line.strip() for line in open(split_tsv, 'r').readlines()]
    self.all_meta = [{'id': item['id'], 'sentence': item['sentence']} \
                      for item in json.load(open(json_file, 'r')) \
                        if item['id'] in valid_ids]
    if self.shuffle:
      random.shuffle(self.all_meta)

  def iter_data(self, slice_element=None):
    slice_element = slice_element or slice(0, len(self))
    for meta in self.all_meta[slice_element]:
      video_path = os.path.join(self.videos_dir, str(meta['id'] + '.mp4'))
      with temp_dir_for_bursting(self.shm_dir_path) as temp_burst_dir:
        frame_paths = burst_video_into_frames(
            video_path, temp_burst_dir, frame_rate=self.frame_rate)
        frames = list(resize_images(frame_paths, self.frame_size))
      result = {'meta': meta, 'frames': frames, 'id': meta['id']}
      yield result

  def __len__(self):
    return len(self.all_meta)


if __name__ == '__main__':
    dataset_path = 'mrw-v1.0.json'
    splits_dir = './split'
    videos_dir = './mp4'
    output_dir = './gulp'
    shm_dir = './shm'
    label_name = ''
    num_workers = 32
    frame_rate = 8
    frame_size = 256 
    videos_per_chunk = 512
    remove_duplicates = True
    shuffle = True

    splits = ['val','test','train']

    # Load the dataset 
    with open(dataset_path, 'r') as f:
      data = json.load(f)

    # Download videos if needed
    if not os.path.isdir(videos_dir):
      os.makedirs(videos_dir)
    download_videos(data, videos_dir, True, 128)
    verify_downloaded_videos(data, videos_dir)
      
    # GULP the dataset
    for split in splits:
      print('Gulping from [{}] split of [{}]'.format(split, dataset_path))
      split_tsv = os.path.join(splits_dir, split+'.tsv')
      output_dir_split = output_dir + '/{}'.format(split)
      shuffle = False if split != 'train' else shuffle
     
      adapter = GulpMRWAdapter(dataset_path, split_tsv, videos_dir, output_dir_split,
                               shuffle=shuffle,
                               frame_size=frame_size,
                               frame_rate=frame_rate,
                               shm_dir_path=shm_dir,
                               label_name=label_name,
                               remove_duplicate_ids=remove_duplicates)
      ingestor = GulpIngestor(adapter, output_dir_split, videos_per_chunk, num_workers)
      ingestor()

    # Update JSON file with num_frames from gulped data
    uids = [item['id'] for item in data]
    uid2nfrms = dict()
    for split in splits:
      uid2nfrms_split = retrieve_nfrms_from_gulp(os.path.join(output_dir, split))
      uid2nfrms.update(uid2nfrms_split)

    for key, val in uid2nfrms.items():
      data[uids.index(key)]['gulp_num_frames'] = val

    with open(dataset_path, 'w') as outfile:
      json.dump(data, outfile) 
