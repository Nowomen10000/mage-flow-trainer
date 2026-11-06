"""MOSS quantization presets and frozen INT8 adapter execution."""
from dataclasses import asdict
import os
from types import SimpleNamespace

import pytest
import toml
import torch

from trainer.training.config import load_config
from trainer.training.quant import QuantConfig, text_encoder_quant_config, quantize_module, quantized_layer_report


def config_data(preset='balanced'):
    return dict(dataset=dict(path='/images'), train=dict(model_family='moss_image', dtype='bfloat16'),
                adapter=dict(kind='lycoris_lora', dtype='bfloat16'), quant=dict(preset=preset))


@pytest.mark.parametrize('preset,dtype,qmm,rotation', [
    ('standard','uint8',False,False), ('balanced','int8',True,True), ('fast','int8',True,False)])
def test_preset_cli_roundtrip_and_encoder_isolation(tmp_path,preset,dtype,qmm,rotation):
    data=config_data(preset)
    data['quant'].update(mode='none', weights_dtype='int4', quantize_text_encoder=True,
                         text_encoder_weights_dtype='uint8')
    path=tmp_path/'preset.toml';path.write_text(toml.dumps(data))
    cfg=load_config(path)
    assert cfg.quant.mode=='frozen' and cfg.quant.weights_dtype==dtype
    assert cfg.quant.use_quantized_matmul is qmm and cfg.quant.use_hadamard is rotation
    te=text_encoder_quant_config(cfg.quant)
    assert te.preset=='custom' and not te.use_hadamard and not te.use_quantized_matmul
    assert te.weights_dtype=='uint8'
    assert asdict(QuantConfig(**asdict(cfg.quant)))==asdict(cfg.quant)


@pytest.mark.parametrize('section,key,value', [
    ('train','model_family','mage_flow'), ('train','dtype','float32'),
    ('adapter','kind','none'), ('adapter','kind','lora'), ('adapter','lycoris_algo','dora'),
    ('adapter','lycoris_bypass',False), ('adapter','dtype','float32')])
def test_accelerated_preset_rejects_unsupported_training(tmp_path,section,key,value):
    data=config_data();data[section][key]=value
    path=tmp_path/'bad.toml';path.write_text(toml.dumps(data))
    with pytest.raises(ValueError):load_config(path)


def test_legacy_settings_unchanged():
    cfg=QuantConfig(mode='training',weights_dtype='uint8')
    assert cfg.preset=='custom' and cfg.mode=='training' and cfg.use_stochastic_rounding
    assert not cfg.use_hadamard


GPU = pytest.mark.skipif(os.environ.get('BLUFLOW_GPU_TESTS')!='1', reason='GPU opt-in')


def test_fixed_ada_tile_respects_backend_and_user_overrides(monkeypatch):
    from sdnq import kernel_wrappers
    from sdnq.kernels import triton_scaled_mm as kernel
    from trainer.training import frozen_matmul as f
    # Simulate the old ABI irrespective of the installed version.
    monkeypatch.setattr(kernel, 'min_block_size', 256, raising=False)
    monkeypatch.setattr(f, '_is_ada', lambda device: True)
    monkeypatch.setattr(kernel_wrappers, 'use_triton_scaled_mm', True)
    for key in list(os.environ):
        if key.startswith('SDNQ_TRITON_MM_'):
            monkeypatch.delenv(key)
    assert f.uses_fixed_ada_tile(torch.device('cuda:0'))
    monkeypatch.setenv('SDNQ_TRITON_MM_BLOCK_SIZE_K_LIST', '64')
    assert not f.uses_fixed_ada_tile(torch.device('cuda:0'))
    monkeypatch.delenv('SDNQ_TRITON_MM_BLOCK_SIZE_K_LIST')
    monkeypatch.setattr(kernel_wrappers, 'use_triton_scaled_mm', False)
    assert not f.uses_fixed_ada_tile(torch.device('cuda:0'))
    monkeypatch.setattr(kernel_wrappers, 'use_triton_scaled_mm', True)
    monkeypatch.setattr(f, '_is_ada', lambda device: False)
    assert not f.uses_fixed_ada_tile(torch.device('cuda:0'))
    monkeypatch.setattr(f, '_is_ada', lambda device: True)
    monkeypatch.delattr(kernel, 'min_block_size')
    assert not f.uses_fixed_ada_tile(torch.device('cuda:0'))


@GPU
@pytest.mark.parametrize('m,k,n,transposed', [
    (97, 256, 512, True), (513, 3072, 3072, True),
    (769, 3072, 16384, True), (1025, 8192, 3072, True),
    (513, 16384, 3072, False)])
def test_fixed_ada_gemm_matches_sdnq_without_autotuning(monkeypatch, m, k, n, transposed):
    from sdnq.kernels.triton_scaled_mm import sdnq_scaled_mm, sdnq_scaled_mm_kernel
    from trainer.training import frozen_matmul as f
    if not f._is_ada(torch.device('cuda:0')):
        pytest.skip('Ada-specific launch configuration')
    a = torch.randint(-127, 128, (m, k), device='cuda', dtype=torch.int8)
    b = torch.randint(-127, 128, (n, k) if transposed else (k, n), device='cuda', dtype=torch.int8)
    if transposed:
        b = b.T
    sa = torch.rand(m, 1, device='cuda') * .01
    sb = torch.rand(1, n, device='cuda') * .01
    reference = sdnq_scaled_mm(a, b, sa, sb, out_dtype=torch.bfloat16)
    monkeypatch.setattr(sdnq_scaled_mm_kernel, 'run', lambda *a, **kw: pytest.fail('autotuner called'))
    result = f._ada_scaled_mm(a, b, sa, sb, torch.bfloat16)
    assert torch.equal(result, reference)


@GPU
@pytest.mark.parametrize('rotation',[False,True])
def test_native_parity_compile_moves_and_state(rotation):
    from sdnq import SDNQConfig
    from sdnq.training import apply_sdnq_training_to_module
    from trainer.training.frozen_matmul import install_frozen_matmul
    torch.set_num_threads(2);torch.manual_seed(42)
    model=torch.nn.Sequential(torch.nn.Linear(256,512,bias=False)).cuda().bfloat16().requires_grad_(False)
    cfg=SDNQConfig(weights_dtype='int8',group_size=-1,use_hadamard=rotation,hadamard_group_size=256,
        use_quantized_matmul=True,quantized_matmul_dtype='int8',is_training=True,
        use_stochastic_rounding=False,add_skip_keys=False,quantization_device='cuda',return_device='cuda')
    model,_=apply_sdnq_training_to_module(model,cfg,torch_dtype=torch.bfloat16)
    references=[]
    for length in [64,96]:
        x=torch.randn(1,length,256,device='cuda',dtype=torch.bfloat16,requires_grad=True)
        grad=torch.randn(1,length,512,device='cuda',dtype=torch.bfloat16)
        y=model(x);y.backward(grad)
        references.append((x.detach(),grad,y.detach(),x.grad.detach()))
    model=install_frozen_matmul(model)
    original_scale=model[0]._sdnq_scale.clone()
    model.cpu().bfloat16().cuda()
    assert torch.equal(model[0]._sdnq_scale,original_scale)
    assert model[0]._sdnq_scale.dtype==torch.float32
    saved={k:v.clone() for k,v in model.state_dict().items()}
    model[0]._sdnq_scale.zero_();model.load_state_dict(saved)
    compiled=torch.compile(model,fullgraph=True,dynamic=True)
    for fn in [model,compiled]:
        for x,grad,y,dx in references:
            z=x.clone().requires_grad_(True);actual=fn(z);actual.backward(grad)
            assert torch.equal(actual,y) and torch.equal(z.grad,dx)
            assert torch.isfinite(z.grad).all()
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())


@GPU
@pytest.mark.parametrize('preset',['balanced','fast'])
@pytest.mark.parametrize('calibrated', [False, True])
def test_moss_adapter_checkpointed_training(preset, calibrated, tmp_path, monkeypatch):
    from trainer.modeling.moss_image import MOSSImage,MOSSParams
    from trainer.training.params import AdapterConfig,apply_adapter
    from trainer.training.lycoris import lycoris_state_dict,load_lycoris_state_dict
    torch.set_num_threads(2);torch.manual_seed(42)
    if calibrated:
        from trainer.training import int8_calibration as c, frozen_matmul as f
        tile = dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=128, GROUP_SIZE_M=8, num_warps=4, num_stages=2)
        monkeypatch.setattr(c, 'candidate_tiles', lambda: [tile])
        monkeypatch.setattr(c, '_ACTIVE', {})
        c.prepare(torch.device('cuda:0'), [(256,256,False), (1024,256,False), (256,512,False),
                  (256,1024,False), (512,256,False)], tokens=(32,96), root=tmp_path)
        monkeypatch.setattr(f, 'int_scaled_mm_func', lambda *a, **kw: pytest.fail(f'unprofiled projection A={a[0].shape} B={a[1].shape} contiguous={a[1].is_contiguous()}'))
        monkeypatch.setattr(f, 'uses_fixed_ada_tile', lambda device: False)
    m=MOSSImage(MOSSParams(dim=256,ffn_dim=512,num_heads=4,num_layers=4,text_dim=128,latent_channels=8)).bfloat16()
    m=quantize_module(m,QuantConfig(preset=preset),torch.device('cuda'),torch.bfloat16,True)
    assert quantized_layer_report(m)[0]>0
    apply_adapter(m,AdapterConfig(kind='lycoris_lora',rank=4,alpha=4,dtype='bfloat16',moss_train_text_refiner=False))
    m.cuda().bfloat16()
    m.configure_execution(True,compile_mode='default')
    params=[p for p in m.parameters() if p.requires_grad]
    before=[p.detach().clone() for p in params]
    opt=torch.optim.AdamW(params,lr=1e-3)
    for length in [8,10]:
        x=torch.randn(1,8,1,length,8,device='cuda',dtype=torch.bfloat16)
        text=torch.randn(1,9,128,device='cuda',dtype=torch.bfloat16)
        mask=torch.ones(1,9,device='cuda',dtype=torch.bool)
        y=m(x,torch.tensor([.4],device='cuda'),(text,mask))[0]
        y.float().square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
        assert all(not p.requires_grad for n,p in m.named_parameters() if 'lycoris_adapter' not in n)
        opt.step();opt.zero_grad(set_to_none=True)
    assert any(not torch.equal(a,b) for a,b in zip(before,params))
    state=lycoris_state_dict(m);load_lycoris_state_dict(m,state)
    assert all('_sdnq' not in k for k in state)


@pytest.mark.skipif(os.environ.get('BLUFLOW_GPU_TESTS')!='1' or os.environ.get('WORLD_SIZE')!='2',
                    reason='requires two GPU ranks')
@pytest.mark.parametrize('calibrated', [False, True])
def test_frozen_matmul_ddp_buffers_and_updates(calibrated, tmp_path, monkeypatch, ddp_device):
    import torch.distributed as dist
    from trainer.training.distributed import configure_quantized_ddp, sync_quantized_model, model_storage

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model_family='moss_image'
            self.params=SimpleNamespace(num_layers=4)
            self.layers=torch.nn.ModuleList([torch.nn.Linear(256,256,bias=False) for _ in range(4)])
        def forward(self,x):
            return self.layers[1](self.adapter(x)).float().square().mean()

    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
    device=ddp_device;torch.set_num_threads(2)
    try:
        if calibrated:
            from trainer.training import int8_calibration as c, frozen_matmul as f
            tile = dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=128, GROUP_SIZE_M=8, num_warps=4, num_stages=2)
            monkeypatch.setattr(c, 'candidate_tiles', lambda: [tile])
            monkeypatch.setattr(c, '_ACTIVE', {})
            monkeypatch.setattr(c, 'moss_shapes', lambda path: [(256,256,False), (256,256,True)])
            prepare = c.prepare
            monkeypatch.setattr(c, 'prepare', lambda device, shapes, **kw:
                                prepare(device, shapes, tokens=(32,96), root=tmp_path, **kw))
            cfg = SimpleNamespace(quant=QuantConfig(preset='fast'), train=SimpleNamespace(model_path='unused'))
            c.prepare_training(cfg, SimpleNamespace(num_processes=2, process_index=rank, device=device))
            monkeypatch.setattr(f, 'int_scaled_mm_func', lambda *a, **kw: pytest.fail('autotuner fallback'))
            monkeypatch.setattr(f, 'uses_fixed_ada_tile', lambda device: False)
        for preset in ['balanced','fast']:
            torch.manual_seed(42+rank)
            m=quantize_module(Model().bfloat16(),QuantConfig(preset=preset),device,torch.bfloat16)
            m.adapter=torch.nn.Linear(256,256,bias=False,device=device,dtype=torch.bfloat16)
            m.to(device).bfloat16()
            sync_quantized_model(m)
            for _,value in model_storage(m):
                ref=value.detach().contiguous().clone();dist.broadcast(ref,0);assert torch.equal(ref,value)
            kwargs=configure_quantized_ddp().to_kwargs()
            wrapped=torch.nn.parallel.DistributedDataParallel(m,device_ids=[rank],**kwargs)
            optimizer=torch.optim.AdamW(m.adapter.parameters(),lr=1e-3)
            before=m.adapter.weight.detach().clone()
            for _ in range(2):
                x=torch.randn(1,64,256,device=device,dtype=torch.bfloat16)
                wrapped(x).backward()
                assert torch.isfinite(m.adapter.weight.grad).all()
                optimizer.step();optimizer.zero_grad(set_to_none=True)
            assert not torch.equal(before,m.adapter.weight)
            ref=m.adapter.weight.detach().clone();dist.broadcast(ref,0)
            assert torch.equal(ref,m.adapter.weight)
            assert m.layers[1]._sdnq_scale.dtype==torch.float32
            del wrapped,m,optimizer
            torch.cuda.empty_cache()
    finally:
        torch._dynamo.reset()


def test_encoder_cache_ignores_new_transformer_only_fields():
    from trainer.data.caption_variations import _encoder_fingerprint_payload
    from trainer.training.config import Config
    from trainer.data.dataset import DatasetConfig
    cfg=Config(dataset=DatasetConfig(path="/images"))
    cfg.quant=QuantConfig(mode='frozen',quantize_text_encoder=True,weights_dtype='uint8')
    payload=_encoder_fingerprint_payload(cfg)['quant']
    legacy=asdict(cfg.quant)
    for key in ['preset','use_hadamard','backend','text_encoder_weights_dtype']:
        legacy.pop(key)
    legacy.update(extra_skip=['lm_head'],use_quantized_matmul=False)
    assert payload==legacy
