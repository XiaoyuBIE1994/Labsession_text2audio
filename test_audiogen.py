import torch
import torchaudio
import IPython
import julius
import numpy as np
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

## Load config
cfg_filepath = 'encodec16k.yaml'
cfg = OmegaConf.load(cfg_filepath)

## Prepare data
audio_dir = Path(cfg.toy_data)
audio_manifest = 'mani_audio.csv'
ext = 'wav'

# audio_len = 0
# with open(audio_manifest, 'w') as f:
#     f.write('id,filepath,sr,length\n') # libri-light too large, no silence trim
#     for audio_filepath in tqdm(sorted(list(audio_dir.glob(f'**/*.{ext}'))), desc=f'prepare..'):
#         audio_id = audio_filepath.stem
#         x, sr = torchaudio.load(audio_filepath)
#         length = x.shape[-1]
#         utt_len = length / sr
#         audio_len += utt_len
#         line = '{},{},{},{}\n'.format(audio_id, audio_filepath, sr, length)
#         f.write(line)
#     print('Total audio len: {:.2f}h'.format(audio_len/3600))


from audiocraft.solvers.builders import (
    get_optimizer,
    get_adversarial_losses,
    get_loss,
    get_balancer
)
from audiocraft.models.builders import get_compression_model
from audiodata import DatasetAudioTrain
# get dataloader
dataset = DatasetAudioTrain(csv_file=audio_manifest,
                            sample_rate=cfg.sample_rate,
                            n_examples=cfg.dataset.n_examples,
                            chunk_size=cfg.dataset.segment_duration,
                            trim_silence=cfg.dataset.trim_silence,
                            normalize=cfg.dataset.normalize,
                            lufs_norm_db=cfg.dataset.lufs_norm_db,
                            lufs_var=cfg.dataset.lufs_var)

dataloader = DataLoader(dataset=dataset, 
                        batch_size=cfg.dataset.batch_size, num_workers=cfg.dataset.num_workers,
                        shuffle=cfg.dataset.shuffle, drop_last=cfg.dataset.drop_last)


# get model and optimizer
model = get_compression_model(cfg.model)
optimizer = get_optimizer(model.parameters(), cfg.optim)


# get loss function
adv_losses = get_adversarial_losses(cfg)
aux_losses = torch.nn.ModuleDict()
info_losses = torch.nn.ModuleDict()
loss_weights = dict()
for loss_name, weight in cfg.losses.items():
    if loss_name in ['adv', 'feat']:
        for adv_name, _ in adv_losses.items():
            loss_weights[f'{loss_name}_{adv_name}'] = weight
    elif weight > 0:
        aux_losses[loss_name] = get_loss(loss_name, cfg)
        loss_weights[loss_name] = weight
    else:
        info_losses[loss_name] = get_loss(loss_name, cfg)
balancer = get_balancer(loss_weights, cfg.balancer)
print("Total # of params: {:.2f} M".format(sum(p.numel() for p in model.parameters())/1e6))


# train
ckpt_path = 'last_ckpt.pth'
total_epoch = cfg.optim.epochs
model = model.to(cfg.device)
model.train()
for epo in range(total_epoch):
    for audio_data in tqdm(dataloader, total=len(dataloader)):
        # prepare data
        x = audio_data.to(cfg.device)
        y = x.clone()
        metrics = {}

        # forward
        qres = model(x)
        y_pred = qres.x

        # discrimilator loss
        d_losses: dict = {}
        for adv_name, adversary in adv_losses.items():
            disc_loss = adversary.train_adv(y_pred, y)
            d_losses[f'd_{adv_name}'] = disc_loss
        metrics['d_loss'] = torch.sum(torch.stack(list(d_losses.values())))
        
        balanced_losses: dict = {}
        other_losses: dict = {}

        # penalty from quantization
        if qres.penalty is not None and qres.penalty.requires_grad:
            other_losses['penalty'] = qres.penalty  # penalty term from the quantizer

        # adversarial losses
        for adv_name, adversary in adv_losses.items():
            adv_loss, feat_loss = adversary(y_pred, y)
            balanced_losses[f'adv_{adv_name}'] = adv_loss
            balanced_losses[f'feat_{adv_name}'] = feat_loss

        # auxiliary losses
        for loss_name, criterion in aux_losses.items():
            loss = criterion(y_pred, y)
            balanced_losses[loss_name] = loss

        # backprop losses that are not handled by balancer
        other_loss = torch.tensor(0., device=cfg.device)
        if 'penalty' in other_losses:
            other_loss += other_losses['penalty']
        if other_loss.requires_grad:
            other_loss.backward(retain_graph=True)
        
        # balancer losses backward
        metrics['g_loss'] = balancer.backward(balanced_losses, y_pred)

        # optimize
        optimizer.step()
        optimizer.zero_grad()

    # save model every epoch
    print('====> Epoch: {}, d_loss: {:.3f}, g_loss: {:.3f}'.format(epo, metrics['d_loss'], metrics['g_loss']))
    torch.save({
            'epoch': epo,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            }, ckpt_path)


# Load the model
ckpt_path = 'last_ckpt.pth'
checkpoint = torch.load(ckpt_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.cpu().eval()

## Load an audio, re-sample to 32kHz
# x, fs = torchaudio.load('example.wav')
x, fs = torchaudio.load('/home/xbie/Data/toy_dataset/LJ001-0001.wav')
x = julius.resample_frac(x, old_sr=fs, new_sr=32000)
print('Audio length: {:.1f}s'.format(x.shape[-1]/32000))

## Encoder
codes, scale = model.encode(x[None,])
y = model.decode(codes, scale)
torchaudio.save('example_recon.wav', y[0], sample_rate=32000)