import os, sys, time, torch
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cremad_experiment'))

from speech_command_Model import ResNet18 as SCResNet18, BasicBlock as SCBasicBlock
from cremad_models import ResNet18 as CRResNet18, BasicBlock as CRBasicBlock
from whisper_models import WhisperEncoder
from Model2 import Classifier as UrbanClassifier

SC_BASE = '/media/sda1/zxlong/L_minghao/InverCRS/Speech_Command'
US_BASE = '/media/sda1/zxlong/L_minghao/InverCRS/UrbanSound8K/UrbanSound8K'

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
NUM_SAMPLES = 1000
BATCH_SIZE = 1000  # try full batch, fall back if OOM
WARMUP = 5

CONFIGS = [
    {'name': 'CREMA-D-ResNet18', 'shape': (1, 64, 640), 'split_points': [1, 2, 3, 4],
     'load_fn': lambda: load_cremad_resnet()},
    {'name': 'CREMA-D-Whisper',  'shape': (80, 300),    'split_points': [1, 2, 4, 8],
     'load_fn': lambda: load_whisper()},
    {'name': 'SC-ResNet18',      'shape': (1, 64, 96),   'split_points': [1, 2, 3, 4],
     'load_fn': lambda: load_sc_resnet()},
    {'name': 'Urban-CNN',        'shape': (1, 64, 200),  'split_points': [1, 2, 3, 4],
     'load_fn': lambda: load_urban_cnn()},
]

def load_cremad_resnet():
    model = CRResNet18(CRBasicBlock, nc=1, nz=6).to(DEVICE)
    ck = torch.load('/media/sda1/zxlong/tmc_re/CREMA-D/classifier/mel_train_record/classifier.pth', map_location='cpu')
    model.load_state_dict(ck['model'])
    model.eval()
    return model

def load_whisper():
    model = WhisperEncoder(split_point=1, freeze=True).to(DEVICE)
    model.eval()
    return model

def load_sc_resnet():
    model = SCResNet18(SCBasicBlock, nc=1, nz=35).to(DEVICE)
    ck = torch.load(f'{SC_BASE}/Result/classifier/mel_spect_train_record/classifier.pth', map_location='cpu')
    model.load_state_dict(ck['model'])
    model.eval()
    return model

def load_urban_cnn():
    model = UrbanClassifier(nc=1, ndf=64, nz=10).to(DEVICE)
    ck = torch.load(f'{US_BASE}/classifier/mel_spect_train_record/classifier.pth', map_location='cpu')
    state = {k.replace('module.', ''): v for k, v in ck['model'].items()}
    model.load_state_dict(state)
    model.eval()
    return model

def measure(model, cfg):
    name = cfg['name']
    shape = cfg['shape']
    split_points = cfg['split_points']
    results = []

    for sp in split_points:
        batch = min(BATCH_SIZE, NUM_SAMPLES)
        while batch >= 1:
            try:
                x = torch.randn(batch, *shape, device=DEVICE)
                if name == 'CREMA-D-Whisper':
                    model.split_point = sp
                    _ = model(x)  # test forward
                else:
                    _ = model(x, split_point=sp)

                # warmup
                for _ in range(WARMUP):
                    if name == 'CREMA-D-Whisper':
                        model.split_point = sp
                        _ = model(x)
                    else:
                        _ = model(x, split_point=sp)

                # measure
                iters = NUM_SAMPLES // batch
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(iters):
                    if name == 'CREMA-D-Whisper':
                        model.split_point = sp
                        _ = model(x)
                    else:
                        _ = model(x, split_point=sp)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0
                total = iters * batch
                avg_ms = elapsed / total * 1000
                results.append((sp, batch, iters, total, elapsed, avg_ms))
                print(f'  {name} sp={sp}: batch={batch} × {iters} = {total} samples, '
                      f'{elapsed:.4f}s total, {avg_ms:.4f}ms/sample', flush=True)
                break
            except torch.cuda.OutOfMemoryError:
                batch //= 2
                torch.cuda.empty_cache()
            except RuntimeError as e:
                if 'out of memory' in str(e).lower() or 'CUDA' in str(e).upper():
                    batch //= 2
                    torch.cuda.empty_cache()
                else:
                    raise
        if batch < 1:
            print(f'  {name} sp={sp}: FAILED (OOM even at batch=1)', flush=True)
            results.append((sp, 0, 0, 0, float('nan'), float('nan')))

    return results

def main():
    print(f'Device: {DEVICE}', flush=True)
    all_results = {}

    for cfg in CONFIGS:
        name = cfg['name']
        print(f'\n=== {name} ===', flush=True)
        model = cfg['load_fn']()
        model.to(DEVICE)
        model.eval()
        all_results[name] = measure(model, cfg)
        del model
        torch.cuda.empty_cache()

    print('\n\n')
    sep = '-' * 90
    print(sep)
    print(f'{"Model":<22} {"Split Pt":<10} {"Batch":<7} {"Total Samp":<12} {"Total Time":<12} {"Avg(ms/samp)":<15}')
    print(sep)
    for name, results in all_results.items():
        for sp, batch, iters, total, elapsed, avg_ms in results:
            t_str = f'{elapsed:.4f}s' if not np.isnan(elapsed) else 'N/A'
            a_str = f'{avg_ms:.4f}' if not np.isnan(avg_ms) else 'N/A'
            print(f'{name:<22} {sp:<10} {batch:<7} {total:<12} {t_str:<12} {a_str:<15}')
    print(sep)

if __name__ == '__main__':
    main()