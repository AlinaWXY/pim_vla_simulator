"""Fresh H16 steady-state latency; baseline and trace use separate processes."""
import argparse
from collections import Counter
from contextlib import contextmanager, nullcontext
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from common import load_weights, stats

ROOT = Path('/var/tmp/shiyuhuang-vla-baseline')
CHECKPOINT = Path('/var/tmp/hf-cache/hub/models--lerobot--pi05_base/snapshots/b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba/model.safetensors')
PINNED = {
    'runtime/openpi/src/openpi/models_pytorch/pi0_pytorch.py': 'dcd7b6f508808ce4b7170a1720a4c62c6da04edae3652d16c94a469dca924e16',
    'runtime/overlay/transformers/models/gemma/modeling_gemma.py': '17eeb54e277939e58cd6f8a92719a275e36baf0c9f463b2754227d42f87aedde',
    'runtime/openpi/src/openpi/models_pytorch/gemma_pytorch.py': '225092e06509da0ae37bb9dca55c00d2804255bb1496f7c8700304b4c52a95eb',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Probe:
    def __init__(self, model, torch):
        self.model, self.torch = model, torch
        self.vit = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(3)]
        self.vlm = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        self.steps = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(10)]
        self.origin, self.end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for event in [self.origin, self.end] + [e for pair in self.vit + [self.vlm] + self.steps for e in pair]:
            event.record()
        torch.cuda.synchronize()
        self.kv = None
        self.reset()

    def reset(self):
        self.vit_count = self.vlm_count = self.step_count = 0

    @contextmanager
    def applied(self, step_events=False, nvtx=False, labels=False):
        restores = []
        def replace(obj, name, function):
            restores.append((obj, name, getattr(obj, name), name in obj.__dict__))
            setattr(obj, name, function)
        vision = self.model.paligemma_with_expert.paligemma.vision_tower
        language = self.model.paligemma_with_expert.paligemma.language_model
        native_vit, native_vlm, native_step = vision.forward, language.forward, self.model.denoise_step
        def vit(*args, **kwargs):
            i = self.vit_count
            self.vit_count += 1
            self.vit[i][0].record()
            value = native_vit(*args, **kwargs)
            self.vit[i][1].record()
            return value
        def vlm(*args, **kwargs):
            self.vlm_count += 1
            self.vlm[0].record()
            value = native_vlm(*args, **kwargs)
            self.vlm[1].record()
            return value
        def denoise(*args, **kwargs):
            i = self.step_count
            self.step_count += 1
            if self.kv is None:
                from attribution import kv_metadata
                self.kv = kv_metadata(args[2])
            with self.torch.cuda.nvtx.range('step/' + str(i)) if nvtx else nullcontext():
                if step_events:
                    self.steps[i][0].record()
                if labels and i == 4:
                    from attribution import scopes_for
                    with scopes_for(self.model, lambda: True, lambda role: self.torch.cuda.nvtx.range('op/' + role)):
                        with self.torch.autograd.profiler.emit_nvtx(record_shapes=True):
                            value = native_step(*args, **kwargs)
                else:
                    value = native_step(*args, **kwargs)
                if step_events:
                    self.steps[i][1].record()
                return value
        replace(vision, 'forward', vit)
        replace(language, 'forward', vlm)
        if step_events or nvtx or labels:
            replace(self.model, 'denoise_step', denoise)
        try:
            yield
        finally:
            for obj, name, original, had in reversed(restores):
                if had:
                    setattr(obj, name, original)
                else:
                    delattr(obj, name)

    def read(self, step_events):
        if (self.vit_count, self.vlm_count) != (3, 1):
            raise RuntimeError('Incorrect phase coverage')
        span = self.origin.elapsed_time(self.end)
        vit = sum(a.elapsed_time(b) for a, b in self.vit)
        vlm = self.vlm[0].elapsed_time(self.vlm[1])
        ae = self.vlm[1].elapsed_time(self.end)
        row = dict(vit_ms=vit, vlm_ms=vlm, ae_ms=ae, other_ms=span-vit-vlm-ae)
        if row['other_ms'] < -0.02:
            raise RuntimeError('Overlapping coarse phase intervals')
        if step_events:
            if self.step_count != 10:
                raise RuntimeError('Not ten denoising steps')
            row['steps_ms'] = [a.elapsed_time(b) for a, b in self.steps]
        return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('baseline', 'trace'), required=True)
    parser.add_argument('--tag', required=True)
    args = parser.parse_args()
    if not args.tag.replace('_', '').isalnum():
        raise ValueError('Unsafe output tag')
    out = ROOT / 'runs' / args.tag
    out.mkdir(exist_ok=False)
    report = dict(status='running', client_experiment_date='2026-11-05', tag=args.tag, mode=args.mode,
                  started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  host=os.uname().nodename, pid=os.getpid(), blocks=[], occupancy=[], snapshots=[])
    def save():
        (out / 'capture.json').write_text(json.dumps(report, indent=2) + '\n')
    def guard(label):
        raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader,nounits'], text=True)
        pids = [int(line.split(',')[0]) for line in raw.splitlines() if line.split(',')[0].strip().isdigit()]
        others = [pid for pid in pids if pid != os.getpid()]
        free = os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize
        report['occupancy'].append(dict(label=label, time_ns=time.monotonic_ns(), raw=raw, others=others, free_disk_bytes=free))
        if others or free < 512 * 1024**2:
            save()
            raise RuntimeError('Other GPU user or low disk reserve: ' + str((others, free)))
    def snapshot(label):
        raw = subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,temperature.gpu,power.draw,clocks.sm,clocks.mem,utilization.gpu', '--format=csv'], text=True)
        report['snapshots'].append(dict(label=label, raw=raw, loadavg=list(os.getloadavg())))
    def log(message):
        print(time.strftime('%H:%M:%S'), message, flush=True)
    tracing = False
    try:
        guard('before_import')
        snapshot('before_import')
        import torch
        import transformers
        from transformers.modeling_utils import no_init_weights
        from openpi.models.pi0_config import Pi0Config
        from openpi.models.model import Observation
        from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
        from adarms_cache import Pi05ConditioningCache
        from mha8 import convert
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.manual_seed(17)
        hashes = {key: sha(ROOT/key) for key in PINNED}
        if hashes != PINNED or torch.__version__ != '2.11.0+cu130' or transformers.__version__ != '4.53.2':
            raise RuntimeError('Runtime differs from accepted baseline')
        report['environment'] = dict(python=sys.executable, torch=torch.__version__, transformers=transformers.__version__,
            declared_openpi_commit='215abfb217dbac7d5f1273282331b9b1866c0479', source_sha256=hashes,
            code_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},
            cpu_threads=1, interop_threads=1, cpu_affinity=sorted(os.sched_getaffinity(0)),
            cuda_launch_blocking=os.environ.get('CUDA_LAUNCH_BLOCKING'),
            ld_preload=os.environ.get('LD_PRELOAD'), nsys_injection=os.environ.get('CUDA_INJECTION64_PATH'),
            precision='accepted selective BF16/FP32', no_compile=True, no_graph=True, no_cache_flush=True)
        if args.mode == 'baseline' and any('nsys' in str(report['environment'][key]).lower() for key in ('ld_preload','nsys_injection')):
            raise RuntimeError('Baseline must not run under Nsight')
        report['configuration'] = dict(model='OpenPI pi0.5', H=16, steps=10, batch=1, q_heads=8, kv_heads=8,
            head_dim=256, adarms_cache=True, cameras=3, image_hw=[224,224], seed=17, synthetic_input=True,
            synthetic_physical_mha=True, task_accuracy_claim=False, checkpoint=str(CHECKPOINT))
        config = Pi0Config(pi05=True, action_horizon=16, pytorch_compile_mode=None)
        log('Load existing local NVMe checkpoint once using sequential readinto')
        with no_init_weights():
            model = PI0Pytorch(config)
        model.paligemma_with_expert.paligemma.tie_weights()
        model.eval()
        report['load'] = load_weights(model, CHECKPOINT, log)
        guard('after_load')
        model.to('cuda')
        torch.cuda.synchronize()
        tokens = json.loads((ROOT/'code/pi_tokens.json').read_text())['pi05']
        gen = torch.Generator().manual_seed(17)
        images = {key:(torch.rand((1,3,224,224),generator=gen)*2-1).to('cuda')
                  for key in ('base_0_rgb','left_wrist_0_rgb','right_wrist_0_rgb')}
        obs = Observation(images=images, image_masks={key:torch.ones(1,dtype=torch.bool,device='cuda') for key in images},
            state=torch.zeros((1,32),device='cuda'), tokenized_prompt=torch.tensor([tokens['ids']],device='cuda'),
            tokenized_prompt_mask=torch.tensor([tokens['mask']],device='cuda',dtype=torch.bool))
        noise = torch.randn((1,16,32),generator=gen).to('cuda')
        def call():
            return model.sample_actions('cuda',obs,noise=noise,num_steps=10)
        with torch.inference_mode():
            report['mha_conversion'] = convert(model)
            report['dimensions'] = {name:dict(layers=len(tr.layers),D=tr.config.hidden_size,F=tr.config.intermediate_size,
                q_heads=tr.config.num_attention_heads,kv_heads=tr.config.num_key_value_heads,head_dim=tr.config.head_dim,
                attention_implementation=tr.config._attn_implementation)
                for name,tr in [('vlm',model.paligemma_with_expert.paligemma.language_model),('ae',model.paligemma_with_expert.gemma_expert.model)]}
            for d in report['dimensions'].values():
                if (d['layers'],d['q_heads'],d['kv_heads'],d['head_dim']) != (18,8,8,256):
                    raise RuntimeError('Geometry mismatch')
            report['parameter_dtype_elements'] = dict(Counter({str(dtype):sum(p.numel() for p in model.parameters() if p.dtype==dtype)
                for dtype in {p.dtype for p in model.parameters()}}))
            reference = call().clone()
            torch.cuda.synchronize()
            t0 = time.monotonic()
            cache = Pi05ConditioningCache(model,num_steps=10,batch_size=1)
            torch.cuda.synchronize()
            report['cache_build_ms_excluded'] = 1000*(time.monotonic()-t0)
            meter = Probe(model,torch)
            def block(name,count,marked=False,step_events=False,nvtx=False,labels=False):
                guard(name+'_before')
                snapshot(name+'_before')
                log(name+': '+str(count))
                rows = []
                with meter.applied(step_events,nvtx,labels) if marked else nullcontext():
                    for i in range(count):
                        guard(name+'_'+str(i))
                        meter.reset()
                        torch.cuda.synchronize()
                        wall0 = time.perf_counter_ns()
                        meter.origin.record()
                        with torch.cuda.nvtx.range('request/'+name+'/'+str(i)) if nvtx else nullcontext():
                            value = call()
                            meter.end.record()
                            meter.end.synchronize()
                        wall1 = time.perf_counter_ns()
                        row = dict(iteration=i,wall_ms=(wall1-wall0)/1e6,gpu_ms=meter.origin.elapsed_time(meter.end))
                        if marked:
                            row.update(meter.read(step_events))
                        if not torch.equal(value,reference) or cache.dense_hits != 370:
                            raise RuntimeError('Output or cache validation failed')
                        rows.append(row)
                        time.sleep(.005)
                result = dict(name=name,samples=rows,wall_ms=stats([r['wall_ms'] for r in rows]),gpu_ms=stats([r['gpu_ms'] for r in rows]))
                if marked:
                    result['phase_ms'] = {key:stats([r[key+'_ms'] for r in rows]) for key in ('vit','vlm','ae','other')}
                if step_events:
                    result['steps_ms'] = [stats([r['steps_ms'][i] for r in rows]) for i in range(10)]
                report['blocks'].append(result)
                save()
                log(json.dumps({name:result['wall_ms']}))
            with cache.applied():
                if not torch.equal(call(),reference):
                    raise RuntimeError('Cache changed output')
                report['output_bitwise_equal'] = True
                report['warmup_requests'] = 20
                log('20 full requests warmup, natural cache state')
                for i in range(20):
                    guard('warmup_'+str(i))
                    call()
                torch.cuda.synchronize()
                if args.mode == 'baseline':
                    block('clean_before',50)
                    block('phases_before',50,marked=True)
                    block('step_control',50,marked=True,step_events=True)
                    block('phases_after',50,marked=True)
                    block('clean_after',50)
                else:
                    block('trace_control_before',20,marked=True,step_events=True,nvtx=True)
                    guard('before_profiler_start')
                    torch.cuda.profiler.start()
                    tracing = True
                    block('trace_burnin',2,marked=True,step_events=True,nvtx=True)
                    block('trace',5,marked=True,step_events=True,nvtx=True)
                    block('attribution',1,marked=True,step_events=True,nvtx=True,labels=True)
                    torch.cuda.profiler.stop()
                    tracing = False
                    for _ in range(10):
                        call()
                    torch.cuda.synchronize()
                    block('trace_control_after',20,marked=True,step_events=True,nvtx=True)
                report['prefix_kv'] = meter.kv
                report['cache_audit'] = cache.metadata()
                report['reference_actions'] = reference.cpu().tolist()
        guard('finished')
        snapshot('finished')
        report.update(status='complete',finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        save()
        log('CAPTURE_COMPLETE')
    except BaseException:
        report.update(status='failed',error=traceback.format_exc())
        save()
        raise
    finally:
        if tracing:
            torch.cuda.profiler.stop()
        (out/'exit.json').write_text(json.dumps(dict(status=report['status']))+'\n')


if __name__ == '__main__':
    main()
