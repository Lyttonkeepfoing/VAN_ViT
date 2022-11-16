#  Copyright Université de Rouen Normandie (1), INSA Rouen (2),
#  tutelles du laboratoire LITIS (1 et 2)
#  contributors :
#  - Denis Coquenet
#
#
#  This software is a computer program written in XXX whose purpose is XXX.
#
#  This software is governed by the CeCILL-C license under French law and
#  abiding by the rules of distribution of free software.  You can  use,
#  modify and/ or redistribute the software under the terms of the CeCILL-C
#  license as circulated by CEA, CNRS and INRIA at the following URL
#  "http://www.cecill.info".
#
#  As a counterpart to the access to the source code and  rights to copy,
#  modify and redistribute granted by the license, users are provided only
#  with a limited warranty  and the software's author,  the holder of the
#  economic rights,  and the successive licensors  have only  limited
#  liability.
#
#  In this respect, the user's attention is drawn to the risks associated
#  with loading,  using,  modifying and/or developing or reproducing the
#  software by the user in light of its specific status of free software,
#  that may mean  that it is complicated to manipulate,  and  that  also
#  therefore means  that it is reserved for developers  and  experienced
#  professionals having in-depth computer knowledge. Users are therefore
#  encouraged to load and test the software's suitability as regards their
#  requirements in conditions enabling the security of their systems and/or
#  data to be ensured and,  more generally, to use and operate it in the
#  same conditions as regards security.
#
#  The fact that you are presently reading this means that you have had
#  knowledge of the CeCILL-C license and that you accept its terms.

import editdistance
import re

# Charset / labels conversion

def LM_str_to_ind(labels, str):
    return [labels.index(c) for c in str]


def LM_ind_to_str(labels, ind, oov_symbol=None):
    if oov_symbol is not None:
        res = []
        for i in ind:
            if i < len(labels):
                res.append(labels[i])
            else:
                res.append(oov_symbol)
    else:
        res = [labels[i] for i in ind]
    return "".join(res)


# OCR METRICS


def edit_cer_from_list(truth, pred):
    edit = 0
    for t, p in zip(truth, pred):
        edit += editdistance.eval(t, p)
    return edit


def format_string_for_wer(str):
    str = re.sub('([\[\]{}/\\()\"\'&+*=<>?.;:,!\-—_€#%°])', r' \1 ', str)
    str = re.sub('([ \n])+', " ", str).strip()
    return str


def edit_wer_from_list(truth, pred):
    edit = 0
    for pred, gt in zip(pred, truth):
        gt = format_string_for_wer(gt)
        pred = format_string_for_wer(pred)
        gt = gt.split(" ")
        pred = pred.split(" ")
        edit += editdistance.eval(gt, pred)
    return edit


def nb_words_from_list(list_gt):
    len_ = 0
    for gt in list_gt:
        gt = format_string_for_wer(gt)
        gt = gt.split(" ")
        len_ += len(gt)
    return len_


def nb_chars_from_list(list_gt):
    return sum([len(t) for t in list_gt])


def cer_from_list_str(str_gt, str_pred):
        len_ = 0
        edit = 0
        for pred, gt in zip(str_pred, str_gt):
            edit += editdistance.eval(gt, pred)
            len_ += len(gt)
        cer = edit / len_
        return cer


def wer_from_list_str(str_gt, str_pred):
    len_ = 0
    edit = 0
    for pred, gt in zip(str_pred, str_gt):
        gt = format_string_for_wer(gt)
        pred = format_string_for_wer(pred)
        gt = gt.split(" ")
        pred = pred.split(" ")
        edit += editdistance.eval(gt, pred)
        len_ += len(gt)
    cer = edit / len_
    return cer


def cer_from_files(file_gt, file_pred):
        with open(file_pred, "r") as f_p:
            str_pred = f_p.readlines()
        with open(file_gt, "r") as f_gt:
            str_gt = f_gt.readlines()
        return cer_from_list_str(str_gt, str_pred)


def wer_from_files(file_gt, file_pred):
    with open(file_pred, "r") as f_p:
        str_pred = f_p.readlines()
    with open(file_gt, "r") as f_gt:
        str_gt = f_gt.readlines()
    return wer_from_list_str(str_gt, str_pred)


import os
import sys
import time
import math
import warnings
import torch
import logging


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


try:
	_, term_width = os.popen('stty size', 'r').read().split()
except:
	term_width = 80
term_width = int(term_width)


TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time

def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def get_logger(out_dir):
    logger = logging.getLogger('Exp')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    file_path = os.path.join(out_dir, "run.log")
    file_hdlr = logging.FileHandler(file_path)
    file_hdlr.setFormatter(formatter)

    strm_hdlr = logging.StreamHandler(sys.stdout)
    strm_hdlr.setFormatter(formatter)

    logger.addHandler(file_hdlr)
    logger.addHandler(strm_hdlr)
    return logger

# class GaussianBlur(object):
#     """
#     Apply Gaussian Blur to the PIL image.
#     """
#     def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
#         self.prob = p
#         self.radius_min = radius_min
#         self.radius_max = radius_max
#
#     def __call__(self, img):
#         do_it = random.random() <= self.prob
#         if not do_it:
#             return img
#
#         return img.filter(
#             ImageFilter.GaussianBlur(
#                 radius=random.uniform(self.radius_min, self.radius_max)
#             )
#         )
def clip_gradients(model, clip):
    norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            norms.append(param_norm.item())
            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)
    return norms

def one_hot(x, num_classes, on_value=1., off_value=0., device='cuda'):
    x = x.long().view(-1, 1)
    return torch.full((x.size()[0], num_classes), off_value, device=device).scatter_(1, x, on_value)

def mixup_target(target, num_classes, lam=1., smoothing=0.0, device='cuda'):
    off_value = smoothing / num_classes
    on_value = 1. - smoothing + off_value
    y1 = one_hot(target, num_classes, on_value=on_value, off_value=off_value, device=device)
    y2 = one_hot(target.flip(0), num_classes, on_value=on_value, off_value=off_value, device=device)
    return y1 * lam + y2 * (1. - lam)

def get_dict():
    from models.vit import vit_tiny
    net1 = vit_tiny()
    para1 = net1.state_dict()
    for k, v in para1.items():
        print(k)
    return k