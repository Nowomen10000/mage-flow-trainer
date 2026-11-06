"""Narrow compatibility fixes for the tested SDNQ release."""
from importlib.metadata import version


def migrate_optimizer_state(optimizer, state_dict):
    """Restore pre-0.2.8 quantized buffers without changing their stored values."""
    import inspect

    from sdnq.dequantizer import SDNQDequantizer
    from sdnq.training import SDNQTensor
    from sdnq.utils import get_sdnq_params

    constructor_keys = set(inspect.signature(SDNQDequantizer).parameters)

    def migrate(value):
        if isinstance(value, SDNQTensor):
            old = value.sdnq_dequantizer
            if not constructor_keys.issubset(vars(old)):
                # Constructor defaults describe the old single-quantization
                # layout. Recompute derived flags; do not requantize buffers.
                metadata = SDNQDequantizer(**{k: v for k, v in vars(old).items()
                                             if k in constructor_keys})
                return SDNQTensor(metadata, **get_sdnq_params(value))
        elif isinstance(value, dict):
            return {k: migrate(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            return type(value)(migrate(v) for v in value)
        return value

    # Old saved groups lack the new scale/codebook options. Preserve every
    # saved setting (including LR); fill only fields that did not exist then.
    groups = [dict(current, **saved) for current, saved in
              zip(optimizer.param_groups, state_dict['param_groups'])]
    if len(groups) != len(state_dict['param_groups']):
        return state_dict  # Let upstream report the group-count mismatch.
    if any(group.get('use_kahan', False) for group in groups):
        # Saved groups can enable Kahan even if the new constructor did not.
        fix_kahan_weight_decay()
    return dict(state_dict, state=migrate(state_dict['state']), param_groups=groups)


def fix_kahan_weight_decay():
    """Include decay in the compensated update instead of cancelling it next step.

    SDNQ 0.2.9 computes the residual against the original parameter after applying
    decay to its FP32 copy. That incorrectly treats decay as a rounding error.
    Preserve normalization/cautious ordering, then include decay in the update
    passed to the upstream Kahan implementation. Non-Kahan steps are untouched.
    """
    if version('sdnq') != '0.2.9':
        return False
    import torch
    from sdnq.common import compile_func
    from sdnq.optim import optimizer, utils
    if getattr(optimizer.update_param_, '_bluflow_kahan_decay_fix', False):
        return True
    original = utils.update_param_

    @torch.no_grad()
    def update_param(param, param_fp32, grad, update, kahan_buffer,
                     learning_rate, weight_decay, clips, final_norm_mode='clip',
                     use_cautious=False, use_stochastic_rounding=True,
                     use_stochastic_buffers=True):
        if kahan_buffer is not None and weight_decay != 0:
            update = utils.apply_norm_to_update_(update, param_fp32, final_norm_mode, clips)
            if use_cautious:
                mask = (update * grad > 0).float()
                mask.div_(mask.mean().clamp_(min=clips[-1]))
                update = update * mask
            update = torch.add(update, param_fp32, alpha=weight_decay)
            weight_decay, final_norm_mode, use_cautious = 0, 'none', False
        return original(param=param, param_fp32=param_fp32, grad=grad, update=update,
                        kahan_buffer=kahan_buffer, learning_rate=learning_rate,
                        weight_decay=weight_decay, clips=clips, final_norm_mode=final_norm_mode,
                        use_cautious=use_cautious, use_stochastic_rounding=use_stochastic_rounding,
                        use_stochastic_buffers=use_stochastic_buffers)

    update_param._bluflow_kahan_decay_fix = True
    optimizer.update_param_ = update_param
    optimizer.update_param_compiled_ = compile_func(update_param)
    import logging
    logging.getLogger(__name__).warning(
        'SDNQ 0.2.9: enabled Kahan weight-decay correction to preserve decoupled decay.'
    )
    return True
