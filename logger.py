from __future__ import print_function

import sys
from collections import OrderedDict


class AverageMeter(object):
  """Computes and stores the average and current value"""

  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / (.0001 + self.count)

  def __str__(self):
    """String representation for logging"""
    # for values that should be recorded exactly e.g. iteration number
    if self.count == 0:
      return str(self.val)
    # for stats
    return '%.4f (%.4f)' % (self.val, self.avg)


class LogCollector(object):
  """A collection of logging objects that can change from train to val"""

  def __init__(self):
    # to keep the order of logged variables deterministic
    self.meters = OrderedDict()

  def update(self, k, v, n=1):
    # create a new meter if previously not recorded
    if k not in self.meters:
      self.meters[k] = AverageMeter()
    self.meters[k].update(v, n)

  def __str__(self):
    """Concatenate the meters in one log line"""
    s = ''
    if sys.version_info.major > 2:
      for i, (k, v) in enumerate(self.meters.items()):
        if i > 0:
          s += '  '
        s += k + ' ' + str(v)
    else:
      for i, (k, v) in enumerate(self.meters.iteritems()):
        if i > 0:
          s += '  '
        s += k + ' ' + str(v)
    return s
