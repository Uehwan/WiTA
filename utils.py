import re
import math
import collections
import editdistance
import matplotlib.pyplot as plt
from bisect import bisect_right

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.optim import Optimizer


ALPHABET = 'abcdefghijklmnopqrstuvwxyz'
HANGUL = 'ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎㅃㅉㄸㄲㅆㄳㄵㄶㄺㄻㄼㄽㄾㄿㅀㅀㅄㅏㅑㅓㅕㅗㅛㅜㅠㅡㅣㅐㅒㅔㅖㅘㅙㅚㅝㅞㅟㅢᴥ '
ALPHA_HAN = 'abcdefghijklmnopqrstuvwxyzㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎㅃㅉㄸㄲㅆㄳㄵㄶㄺㄻㄼㄽㄾㄿㅀㅀㅄㅏㅑㅓㅕㅗㅛㅜㅠㅡㅣㅐㅒㅔㅖㅘㅙㅚㅝㅞㅟㅢᴥ '


def calc_seq_len_2d_kor(len_in):
    len_out = len_in
    len_out = len_out + 2
    len_out = math.floor((len_out - 2) / 2) + 1
    return len_out


def calc_seq_len_2d_eng(len_in):
    len_out = len_in
    len_out = math.floor((len_out - 2) / 2) + 1
    return len_out


def calc_seq_len(len_in):       # 2+1(eng&kor) & r3d(eng)
    len_out = len_in
    len_out = math.floor((len_out - 3 + 2) / 2) + 1
    len_out = math.floor((len_out - 3 + 2) / 2) + 1
    return len_out


def calc_seq_len_mc3(len_in):   # eng
    len_out = len_in
    len_out = math.floor((len_out - 3 + 2) / 2) + 1
    len_out = math.floor((len_out - 2) / 2) + 1
    return len_out


def calc_seq_len_rmc3(len_in):  # eng
    len_out = len_in
    len_out = math.floor((len_out - 2) / 2) + 1
    len_out = math.floor((len_out - 3 + 2) / 2) + 1
    return len_out


def seq_len_r3d_kor(len_in):
    len_out = len_in
    len_out = len_out + 2
    len_out = math.floor((len_out - 3 + 2) / 2) + 1
    len_out = math.floor((len_out - 3 + 2) / 2) + 1
    return len_out


def seq_len_mc3_kor(len_in):
    len_out = len_in
    len_out = len_out + 2
    len_out = math.floor((len_out - 3 + 2) / 2) + 1
    len_out = math.floor((len_out - 2) / 2) + 1
    return len_out


def seq_len_rmc3_kor(len_in):
    len_out = len_in
    len_out = len_out + 2
    len_out = math.floor((len_out - 2) / 2) + 1
    len_out = math.floor((len_out - 3 + 2) / 2) + 1
    return len_out


# original: https://github.com/nianticlabs/monodepth2/blob/master/utils.py
def sec_to_hm_str(t):
    """Convert time in seconds to a nice string
    e.g. 10239 -> '02h50m39s'
    """
    h, m, s = sec_to_hm(t)
    return "{:02d}h{:02d}m{:02d}s".format(h, m, s)


def sec_to_hm(t):
    """Convert time in seconds to time in hours, minutes and seconds
    e.g. 10239 -> (2, 50, 39)
    """
    t = int(t)
    s = t % 60
    t //= 60
    m = t % 60
    t //= 60
    return t, m, s


# original: https://github.com/meijieru/crnn.pytorch/blob/master/utils.py
# modified by UHKIM
class StrLabelConverter(object):
    """Convert between str and label.

    NOTE:
        Insert `blank` to the alphabet for CTC.

    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet, ignore_case=True):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + '-'  # for `-1` index

        self.dict = {}
        for i, char in enumerate(self.alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1

    def encode(self, text):
        """Support batch or single str.

        Args:
            text (str or list of str): texts to convert.

        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """
        if isinstance(text, str):
            text_list = []
            prev = ''
            for char in text:
                if char == prev:
                    text_list.append(self.dict['-'])
                text_list.append(
                    self.dict[char.lower() if self._ignore_case else char])
                prev = char
            text = text_list
            '''
            text = [
                self.dict[char.lower() if self._ignore_case else char]
                for char in text
            ]
            '''
            length = [len(text)]
        elif isinstance(text, collections.Iterable):
            length = [len(s) for s in text]
            text = ''.join(text)
            text, _ = self.encode(text)
        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, t, length, raw=False):
        """Decode encoded texts back into strs.

        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.

        Raises:
            AssertionError: when the texts and its length does not match.

        Returns:
            text (str or list of str): texts to convert.
        """
        if length.numel() == 1:     # batch size = 1
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(
                t.numel(), length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(int(length)):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(
            ), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts


def remove_non_silence_noises(input_text):
    """
      Removes non_silence noises from a transcript
    """
    non_silence_noises = ["noise", "um", "ah", "er", "umm",
                          "uh", "mm", "mn", "mhm", "mnh", "<START>", "<END>"]
    re_non_silence_noises = re.compile(
        r"\b({})\b".format("|".join(non_silence_noises)))
    return re.sub(re_non_silence_noises, '', input_text)


def cer(ref, hyp, remove_nsns=False):
    """
      Calculate character error rate between two strings or time_aligned_text objects
      >>> cer("this cat", "this bad")
      25.0
    """
    ref = re.sub('^<START>|<EOS>$', '', ref)
    hyp = re.sub('^<START>|<EOS>$', '', hyp)

    if remove_nsns:
        ref = remove_non_silence_noises(ref)
        hyp = remove_non_silence_noises(hyp)

    # calculate per line CER
    return editdistance.eval(ref, hyp), len(ref)


def wer(ref, hyp, remove_nsns=False):
    """
      Calculate word error rate between two string or time_aligned_text objects
      >>> wer("this is a cat", "this is a dog")
      25.0
    """
    ref = re.sub('^<START>|<EOS>$', '', ref)
    hyp = re.sub('^<START>|<EOS>$', '', hyp)

    # optionally, remove non silence noises
    if remove_nsns:
        ref = remove_non_silence_noises(ref)
        hyp = remove_non_silence_noises(hyp)

    # calculate WER
    return editdistance.eval(ref.split(' '), hyp.split(' ')), len(ref.split(' '))


def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


# original: https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/solver/lr_scheduler.py
# modified by UHKIM
class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


# original: https://github.com/cybertronai/pytorch-lamb/blob/master/pytorch_lamb/lamb.py
class Lamb(Optimizer):
    r"""Implements Lamb algorithm.
    It has been proposed in `Large Batch Optimization for Deep Learning: Training BERT in 76 minutes`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        adam (bool, optional): always use trust ratio = 1, which turns this into
            Adam. Useful for comparison purposes.
    .. _Large Batch Optimization for Deep Learning: Training BERT in 76 minutes:
        https://arxiv.org/abs/1904.00962
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6,
                 weight_decay=0, adam=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        self.adam = adam
        super(Lamb, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'Lamb does not support sparse gradients, consider SparseAdam instad.')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                # m_t
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                # v_t
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                # Paper v3 does not use debiasing.
                # bias_correction1 = 1 - beta1 ** state['step']
                # bias_correction2 = 1 - beta2 ** state['step']
                # Apply bias to lr to avoid broadcast.
                # * math.sqrt(bias_correction2) / bias_correction1
                step_size = group['lr']

                weight_norm = p.data.pow(2).sum().sqrt().clamp(0, 10)

                adam_step = exp_avg / exp_avg_sq.sqrt().add(group['eps'])
                if group['weight_decay'] != 0:
                    adam_step.add_(group['weight_decay'], p.data)

                adam_norm = adam_step.pow(2).sum().sqrt()
                if weight_norm == 0 or adam_norm == 0:
                    trust_ratio = 1
                else:
                    trust_ratio = weight_norm / adam_norm
                state['weight_norm'] = weight_norm
                state['adam_norm'] = adam_norm
                state['trust_ratio'] = trust_ratio
                if self.adam:
                    trust_ratio = 1

                p.data.add_(-step_size * trust_ratio, adam_step)

        return loss

# original: https://github.com/ZhengkunTian/CTC-Visualization/blob/master/ctc-viz/viz.py
# modified by UHKIM
def draw(probs, index=None, vocab=None, blank=0, save_or_show='show', saved_name=None):
    """
    args:
        probs: the probs from softmax layer of CTC model. dim: [time_steps, vocab_size] or [time_steps, target_length].
        index(optional): list, target index of vocabulary. such as [2, 4, 8, 3]
        vocab(optional): ordered list or dict, the vocabulary in your model. such ['a', 'b'] or {'a': 0, 'b': 1}
        blank: the index of blank in your vocabulary, default: 0.
        save_or_show: 'show', 'save' or 'all'
        saved_name: you can set saved name to store the generated pic
    >>> probs = np.array([
            [0.91, 0.03, 0.03, 0.03], [0.91, 0.03, 0.03, 0.03],
            [0.91, 0.03, 0.03, 0.03], [0.03, 0.91, 0.03, 0.03],
            [0.91, 0.03, 0.03, 0.03], [0.03, 0.03, 0.91, 0.03],
            [0.91, 0.03, 0.03, 0.03], [0.03, 0.03, 0.03, 0.91],
            [0.91, 0.03, 0.03, 0.03], [0.91, 0.03, 0.03, 0.03]
        ])
    >>> index = [1, 2, 3]
    >>> vocab = ['blank', 'a', 'b', 'c']
    >>> draw(probs, index=index, vocab=vocab, blank=0)
    """
    assert len(
        probs.shape) == 2, 'Please set the dimension of probs as [time_steps, vocab_size] or [time_steps, target_length].'
    assert save_or_show in [
        'show', 'save', 'all'], "Please set value of save_or_show in ['show', 'save', 'all'] "

    if index is not None:
        if blank not in index:
            index.append(blank)

        index.sort()

        if vocab is not None:
            if type(vocab).__name__ == 'list':
                vocab2int = {vocab[i]: i for i in range(len(vocab))}
            else:
                vocab2int = vocab

            int2vocab = {i: v for (v, i) in vocab2int.items()}

        if probs.shape[1] > len(index):
            probs = probs[:, index]

    if index is not None:
        if vocab is not None:
            label = [int2vocab[i] for i in index]
        else:
            label = [str(i) for i in index]
    else:
        label = None

    plt.title('Result Analysis')

    x = list(range(probs.shape[0]))
    for i in range(probs.shape[1]):
        if label is not None:
            if index[i] == blank:
                plt.plot(x, probs[:, i], linestyle='--', label='blank')
            else:
                plt.plot(x, probs[:, i], label=label[i])
        else:
            plt.plot(x, probs[:, i])

    plt.legend()
    plt.xlabel('Frams')
    plt.ylabel('Probs')

    if save_or_show in ['save', 'all']:
        if saved_name is not None:
            plt.savefig(saved_name)
        else:
            plt.savefig('ctc-viz.jpg')

    if save_or_show in ['show', 'all']:
        plt.show()
