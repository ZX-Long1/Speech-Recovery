import os, sys, json
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

sys.path.insert(0, '/media/sda1/zxlong/tmc_re')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import ManifestDataset
from cremad_experiment.cremad_models import ResNet18, BasicBlock
from cremad_experiment.cremad_models import Conv_sp1_shallow, Conv_sp3_shallow
from speech_command_Model import Conv_sp1 as SC_Conv_sp1
from speech_command_Model import Conv_sp3 as SC_Conv_sp3

SC_BASE = '/media/sda1/zxlong/L_minghao/InverCRS/Speech_Command'
CREMAD = '/media/sda1/zxlong/tmc_re/CREMA-D'
REPRO = os.path.dirname(os.path.abspath(__file__))

SC_MIN = -80.00000381469727
SC_MAX = 3.814697265625e-06

CREMA_EMO = {'ANG': 0, 'DIS': 1, 'FEA': 2, 'HAP': 3, 'NEU': 4, 'SAD': 5}
CREMA_NORM = json.load(open(f'{CREMAD}/processed/norm_info.json'))


def TV(x):
    batch_size = x.size(0)
    h_x, w_x = x.size(2), x.size(3)
    count_h = x[:, :, 1:, :].numel()
    count_w = x[:, :, :, 1:].numel()
    h_tv = torch.pow(x[:, :, 1:, :] - x[:, :, :h_x - 1, :], 2).sum()
    w_tv = torch.pow(x[:, :, :, 1:] - x[:, :, :, :w_x - 1], 2).sum()
    return (h_tv / count_h + w_tv / count_w) / batch_size


def TV1d(x):
    batch_size = x.size(0)
    diff_f = x[:, 1:, :] - x[:, :-1, :]
    diff_t = x[:, :, 1:] - x[:, :, :-1]
    return (diff_f.pow(2).sum() / diff_f.numel() +
            diff_t.pow(2).sum() / diff_t.numel()) / batch_size


def per_sample_snr(orig, recon):
    b = orig.size(0)
    signal = orig.reshape(b, -1).pow(2).mean(dim=1)
    noise = (orig - recon).reshape(b, -1).pow(2).mean(dim=1)
    snr = 10 * torch.log10(signal / (noise + 1e-12))
    return snr.mean().item()


def per_pixel_mse(recon, data):
    return F.mse_loss(recon, data).item()


def sc_label(stem):
    return int(stem.split('_')[0])


def cremad_label(stem):
    return CREMA_EMO[stem.split('_')[2]]


def build_loaders(dataset, smoke):
    if dataset == 'sc':
        root = f'{SC_BASE}/audio/train_set_mel_spect_numpy'
        manifest = f'{REPRO}/manifests/sc_aux.txt'
        minv, maxv = SC_MIN, SC_MAX
        label_fn = sc_label
        n_classes = 35
        test_root = f'{SC_BASE}/audio/test_set_mel_spect_numpy'
        test_manifest = f'{REPRO}/manifests/sc_test.txt'
    else:
        root = f'{CREMAD}/processed'
        manifest = f'{REPRO}/manifests/cremad_aux.txt'
        minv, maxv = CREMA_NORM['mel_min'], CREMA_NORM['mel_max']
        label_fn = cremad_label
        n_classes = 6
        test_root = f'{CREMAD}/processed'
        test_manifest = f'{REPRO}/manifests/cremad_test.txt'
    tf = transforms.Compose([transforms.ToTensor()])
    limit = 4 if smoke else None
    aux_set = ManifestDataset(root, manifest, minv, maxv, label_fn, tf,
                              limit_per_class=limit)
    test_set = ManifestDataset(test_root, test_manifest, minv, maxv, label_fn, tf,
                               limit_per_class=limit)
    return aux_set, test_set, n_classes


class TestDirDataset:
    def __init__(self, root, min_val, max_val, label_fn, transform=None,
                 limit_per_class=None):
        import glob
        self.root = root
        self.min_val = min_val
        self.max_val = max_val
        self.label_fn = label_fn
        self.transform = transform
        files = sorted(glob.glob(f'{root}/*_mel.npy'))
        self.items = [os.path.basename(f) for f in files]
        if limit_per_class is not None:
            seen = {}
            picked = []
            for it in self.items:
                c = label_fn(it.replace('_mel.npy', ''))
                if seen.get(c, 0) < limit_per_class:
                    seen[c] = seen.get(c, 0) + 1
                    picked.append(it)
            self.items = picked

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        feat = np.load(os.path.join(self.root, self.items[idx]))
        norm = ((feat - self.min_val) / (self.max_val - self.min_val)) * 255
        img = np.uint8(norm)
        if self.transform is not None:
            img = Image.fromarray(img)
            img = self.transform(img)
        label = self.label_fn(self.items[idx].replace('_mel.npy', ''))
        return img, label


def build_classifier(dataset, device):
    nz = 35 if dataset == 'sc' else 6
    model = ResNet18(BasicBlock, nc=1, nz=nz).to(device)
    if dataset == 'sc':
        path = f'{SC_BASE}/Result/classifier/mel_spect_train_record/classifier.pth'
    else:
        path = f'{CREMAD}/classifier/mel_train_record/classifier.pth'
    ck = torch.load(path, map_location='cpu')
    model.load_state_dict(ck['model'])
    model.eval()
    return model


def build_decoder(dataset, sp, device):
    if dataset == 'sc':
        cls = {1: SC_Conv_sp1, 3: SC_Conv_sp3}[sp]
    else:
        cls = {1: Conv_sp1_shallow, 3: Conv_sp3_shallow}[sp]
    return cls().to(device)


def run_resnet(dataset='sc', mode='whitebox', split_point=1, smoke=False,
               epochs=100, batch_size=128, lr=2e-4, ntv=0.5, seed=42,
               gpu=0, out_dir=None):
    import random
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device(f'cuda:{gpu}')
    torch.cuda.set_device(gpu)

    if smoke:
        epochs = 1
    aux_set, test_set, n_classes = build_loaders(dataset, smoke)
    aux_loader = torch.utils.data.DataLoader(aux_set, batch_size=batch_size,
                                             shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                              shuffle=False, num_workers=4)
    print(f'[{dataset} {mode} sp{split_point}] aux={len(aux_set)} test={len(test_set)} '
          f'smoke={smoke}', flush=True)

    classifier = build_classifier(dataset, device)
    decoder = build_decoder(dataset, split_point, device)
    opt = torch.optim.Adam(decoder.parameters(), lr=lr, betas=(0.5, 0.999))

    step = 0.001
    n_dirs = 25
    best_aux_repr = float('inf')
    best_state = None
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        decoder.train()
        for data, _ in aux_loader:
            data = data.to(device)
            opt.zero_grad()
            with torch.no_grad():
                pred = classifier(data, split_point=split_point)
            recon = decoder(pred)
            if mode == 'whitebox':
                recon_pred = classifier(recon, split_point=split_point)
                loss = F.mse_loss(recon_pred, pred) + ntv * TV(recon)
                loss.backward()
            else:
                with torch.no_grad():
                    grad = torch.zeros_like(recon)
                    for _ in range(n_dirs):
                        d = torch.randn_like(recon)
                        t1 = classifier(recon + step * d, split_point=split_point)
                        t2 = classifier(recon - step * d, split_point=split_point)
                        l1 = F.mse_loss(t1, pred)
                        l2 = F.mse_loss(t2, pred)
                        grad += (l1 - l2) * d
                    grad = grad / (2 * n_dirs * step)
                loss_tv = ntv * TV(recon)
                loss_tv.backward(retain_graph=True)
                recon.backward(grad)
            opt.step()

        decoder.eval()
        aux_repr = 0.0
        n = 0
        with torch.no_grad():
            for data, _ in aux_loader:
                data = data.to(device)
                pred = classifier(data, split_point=split_point)
                recon = decoder(pred)
                aux_repr += F.mse_loss(classifier(recon, split_point=split_point),
                                       pred).item() * data.size(0)
                n += data.size(0)
        aux_repr /= n
        if aux_repr < best_aux_repr:
            best_aux_repr = aux_repr
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in decoder.state_dict().items()}
        print(f'  epoch {epoch}/{epochs}: aux_repr_mse={aux_repr:.6f} best@epoch{best_epoch}', flush=True)

    decoder.load_state_dict(best_state)
    decoder.eval()
    test_mse = 0.0
    test_snr = 0.0
    n = 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            pred = classifier(data, split_point=split_point)
            recon = decoder(pred)
            test_mse += F.mse_loss(recon, data).item() * data.size(0)
            test_snr += per_sample_snr(data, recon) * data.size(0)
            n += data.size(0)
    test_mse /= n
    test_snr /= n
    print(f'=== [{dataset} {mode} sp{split_point}] best_epoch={best_epoch} '
          f'best_aux_repr={best_aux_repr:.6f} | test_mse={test_mse:.6f} '
          f'test_snr={test_snr:.2f}dB', flush=True)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        torch.save({'epoch': best_epoch, 'model': decoder.state_dict(),
                    'best_aux_repr': best_aux_repr},
                   f'{out_dir}/inversion.pth')
        print(f'saved {out_dir}/inversion.pth', flush=True)
    return test_mse, test_snr


def run_whisper(mode='whitebox', split_point=1, smoke=False, epochs=100,
                batch_size=128, lr=2e-4, ntv=0.5, seed=42, gpu=0, out_dir=None):
    import random
    import whisper
    import librosa
    from torch.utils.data import Dataset, DataLoader
    from cremad_experiment.whisper_models import (WhisperEncoder,
                                                  Decoder_whitebox,
                                                  Decoder_blackbox)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device(f'cuda:{gpu}')
    torch.cuda.set_device(gpu)

    if smoke:
        epochs = 1
    stems = [os.path.basename(l.strip()).replace('_mel.npy', '')
             for l in open(f'{REPRO}/manifests/cremad_aux.txt') if l.strip()]
    if smoke:
        stems = stems[:12]
    test_stems = sorted(os.path.basename(f).replace('_mel.npy', '')
                        for f in os.listdir(f'{CREMAD}/processed/test')
                        if f.endswith('_mel.npy'))
    if smoke:
        test_stems = test_stems[:12]
    print(f'[whisper {mode} sp{split_point}] aux={len(stems)} test={len(test_stems)} '
          f'smoke={smoke}', flush=True)

    class AudioSet(Dataset):
        def __init__(self, stems):
            self.stems = stems
        def __len__(self):
            return len(self.stems)
        def __getitem__(self, i):
            a, _ = librosa.load(f'{CREMAD}/AudioWAV/{self.stems[i]}.wav',
                                sr=16000, mono=True)
            return torch.from_numpy(a).float(), self.stems[i]

    def collate(batch):
        max_len = max(a.shape[0] for a, _ in batch)
        padded = [F.pad(a, (0, max_len - a.shape[0])) for a, _ in batch]
        return torch.stack(padded), [s for _, s in batch]

    aux_loader = DataLoader(AudioSet(stems), batch_size=batch_size, shuffle=True,
                            collate_fn=collate, num_workers=0)
    test_loader = DataLoader(AudioSet(test_stems), batch_size=1, shuffle=False,
                             collate_fn=collate, num_workers=0)

    encoder = WhisperEncoder(split_point=split_point, freeze=True).to(device)
    encoder.eval()
    decoder = (Decoder_whitebox if mode == 'whitebox' else Decoder_blackbox)().to(device)
    opt = torch.optim.Adam(decoder.parameters(), lr=lr, betas=(0.5, 0.999))

    step = 0.001
    n_dirs = 25
    best_aux_repr = float('inf')
    best_state = None
    best_epoch = 0

    def mels_of(batch_audio):
        ms = [whisper.log_mel_spectrogram(a.numpy(), padding=0)
              for a in batch_audio]
        T = max(m.shape[1] for m in ms)
        mb = torch.zeros(len(ms), 80, T, device=device)
        for i, m in enumerate(ms):
            mb[i, :, :m.shape[1]] = m.to(device)
        return mb

    for epoch in range(1, epochs + 1):
        decoder.train()
        for audio, _ in aux_loader:
            mel_batch = mels_of(audio)
            opt.zero_grad()
            with torch.no_grad():
                feat = encoder(mel_batch)
            recon = decoder(feat)
            t = min(recon.shape[-1], mel_batch.shape[-1])
            recon = recon[:, :, :t]
            with torch.no_grad():
                feat_t = encoder(mel_batch[:, :, :t])
            recon2 = decoder(feat_t)
            t2 = min(recon2.shape[-1], t)
            recon2 = recon2[:, :, :t2]
            if mode == 'whitebox':
                with torch.no_grad():
                    feat_tt = encoder(mel_batch[:, :, :t2])
                feat_recon = encoder(recon2)
                loss = F.mse_loss(feat_recon, feat_tt) + ntv * TV1d(recon2)
                loss.backward()
            else:
                with torch.no_grad():
                    feat_obs = encoder(mel_batch[:, :, :t2])
                    grad = torch.zeros_like(recon2)
                    for _ in range(n_dirs):
                        d = torch.randn_like(recon2)
                        t1 = encoder(recon2 + step * d)
                        t2q = encoder(recon2 - step * d)
                        l1 = F.mse_loss(t1, feat_obs)
                        l2 = F.mse_loss(t2q, feat_obs)
                        grad += (l1 - l2) * d
                    grad = grad / (2 * n_dirs * step)
                loss_tv = ntv * TV1d(recon2)
                loss_tv.backward(retain_graph=True)
                recon2.backward(grad)
            opt.step()

        decoder.eval()
        aux_repr = 0.0
        n = 0
        with torch.no_grad():
            for audio, _ in aux_loader:
                mel_batch = mels_of(audio)
                feat = encoder(mel_batch)
                recon = decoder(feat)
                t = min(recon.shape[-1], mel_batch.shape[-1])
                recon = recon[:, :, :t]
                feat_target = encoder(mel_batch[:, :, :t])
                aux_repr += F.mse_loss(encoder(recon), feat_target).item() * audio.size(0)
                n += audio.size(0)
        aux_repr /= n
        if aux_repr < best_aux_repr:
            best_aux_repr = aux_repr
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in decoder.state_dict().items()}
        print(f'  epoch {epoch}/{epochs}: aux_repr_mse={aux_repr:.6f} best@epoch{best_epoch}', flush=True)

    decoder.load_state_dict(best_state)
    decoder.eval()
    test_mse = 0.0
    test_snr = 0.0
    n = 0
    with torch.no_grad():
        for audio, _ in test_loader:
            mel_batch = mels_of(audio)
            feat = encoder(mel_batch)
            recon = decoder(feat)
            t = min(recon.shape[-1], mel_batch.shape[-1])
            recon = recon[:, :, :t]
            mel_t = mel_batch[:, :, :t]
            test_mse += F.mse_loss(recon, mel_t).item() * audio.size(0)
            test_snr += per_sample_snr(mel_t, recon) * audio.size(0)
            n += audio.size(0)
    test_mse /= n
    test_snr /= n
    print(f'=== [whisper {mode} sp{split_point}] best_epoch={best_epoch} '
          f'best_aux_repr={best_aux_repr:.6f} | test_mse={test_mse:.6f} '
          f'test_snr={test_snr:.2f}dB', flush=True)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        torch.save(decoder.state_dict(), f'{out_dir}/decoder_best.pth')
        print(f'saved {out_dir}/decoder_best.pth', flush=True)
    return test_mse, test_snr
