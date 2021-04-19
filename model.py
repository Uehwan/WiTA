import copy
import utils
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from resnet3d import r3d, mc3, rmc3, r2plus1d


class GestureTranslator(nn.Module):
    """Master class of TypingInTheAir

    Receives a sequence of images (video) and
    returns a sequence of characters hand-written.
    """

    def __init__(self, opts):
        super(GestureTranslator, self).__init__()
        self.opts = opts
        if self.opts.data_type == 'english':
            self.num_class = len(utils.ALPHABET) + 2 if self.opts.loss_type == 'ctc' else len(utils.ALPHABET) + 1
        elif self.opts.data_type == 'korean':
            self.num_class = len(utils.HANGUL) + 2 if self.opts.loss_type == 'ctc' else len(utils.HANGUL) + 1
        else:
            self.num_class = len(utils.ALPHA_HAN) + 2 if self.opts.loss_type == 'ctc' else len(utils.ALPHA_HAN) + 1

        if self.opts.model_type == "r3d":       # 3D Conv
            self.encoder_module = r3d()
        elif self.opts.model_type == "mc3":     # 3D Conv => 2D Conv
            self.encoder_module = mc3()
        elif self.opts.model_type == "rmc3":    # 2D Conv => 3D Conv
            self.encoder_module = rmc3()
        elif self.opts.model_type == "twoplusone":
            self.encoder_module = r2plus1d()        
        
        if self.opts.pretrained == "True":
            if self.opts.model_type == "r3d":  # 3D Conv
                model_ft = torchvision.models.video.r3d_18(pretrained=True)
            elif self.opts.model_type == "mc3":  # 3D Conv => 2D Conv
                model_ft = torchvision.models.video.mc3_18(pretrained=True)
            elif self.opts.model_type == "twoplusone":
                model_ft = torchvision.models.video.r2plus1d_18(pretrained=True)
            pre_dict = model_ft.state_dict()
            del pre_dict['fc.weight']
            del pre_dict['fc.bias']

            if self.opts.data_type == "korean":
                pre_dict['stem.0.weight'] = pre_dict['stem.0.weight'][:, :, 0] + pre_dict['stem.0.weight'][:, :, 1] + pre_dict['stem.0.weight'][:, :, 2]
                pre_dict['stem.0.weight'] /= 3

            encoder_dict = self.encoder_module.state_dict()
            encoder_dict.update(pre_dict)
            self.encoder_module.load_state_dict(encoder_dict)

        self.recurrent_module = None
        if self.opts.recurrent_type.lower() == 'lstm':
            self.recurrent_module = nn.LSTM(
                opts.input_size,
                opts.hidden_size,
                opts.num_layers,
                batch_first=True,
                bidirectional=True)
        elif self.opts.recurrent_type.lower() == 'gru':
            self.recurrent_module = nn.GRU(
                opts.input_size,
                opts.hidden_size,
                opts.num_layers,
                batch_first=True,
                bidirectional=True)
        
        if self.recurrent_module:
            self.fc1 = nn.Linear(2 * opts.hidden_size, opts.hidden_size_fc)
            self.fc2 = nn.Linear(opts.hidden_size_fc, self.num_class)
        else:
            self.fc1 = None
            self.fc2 = nn.Linear(opts.input_size, self.num_class)

    def forward(self, seq_img, seq_lens):
        embeddings = self.encoder_module(seq_img.permute(0, 2, 1, 3, 4))  # (batch_size, seq_length, feature_size)
        if self.recurrent_module:
            x_packed = pack_padded_sequence(embeddings, seq_lens, batch_first=True, enforce_sorted=False)
            outputs_packed, _ = self.recurrent_module(x_packed)           # (batch_size, seq_length, bi * hidden_size)
            outputs_padded, _ = pad_packed_sequence(outputs_packed, batch_first=True)
            outputs = F.relu(self.fc1(outputs_padded))                    # (batch_size, seq_length, hidden_size_fc)
            outputs = self.fc2(outputs)                                   # (batch_size, seq_length, num_class)
            return outputs
        outputs = self.fc2(embeddings)
        return outputs


if __name__ == "__main__":
    import torch
    from options import AirTypingOptions

    options = AirTypingOptions()
    opts = options.parse()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    gt = GestureTranslator(opts).to(device)
    
    input = torch.rand(1, 92, 3, 224, 224).to(device)
    out = gt(input, len(input))
    print("input shape: {}".format(input.shape))
    print("output shape: {}".format(out.shape))
