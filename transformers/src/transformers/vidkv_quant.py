import triton
import triton.language as tl
import random
import numpy as np
import torch

def unpack_and_dequant_vcache(code: torch.Tensor, 
                              scale: torch.Tensor, 
                              mn: torch.Tensor,
                              group_size: int, 
                              bits: float,
							  type: torch.dtype = torch.float16
							  ):
    if abs(bits - 1.58) < 1e-3:
        data = unpack_tensor(code, 2, pack_dim=3)
        shape = data.shape
        num_groups = shape[-1] // group_size
        data = data.view(shape[:-1] + (num_groups, group_size,))
        data = data.to(torch.float16)
        data = data - 1.0
        return data.view(shape)
    else:
        assert bits in [1, 2, 4, 8]
        assert len(code.shape) == 4
        data = unpack_tensor(code, bits, pack_dim=3)
        shape = data.shape
        num_groups = shape[-1] // group_size
        data = data.view(shape[:-1] + (num_groups, group_size,))
        data = data.to(type)
        data = data * scale + mn 
        return data.view(shape)
	
def pack_tensor(data, bits, pack_dim):
	shape = data.shape
	feat_per_int = 32 // bits
	assert bits in [2,4,8], "Only 2, 4, 8 bits are supported"
	assert shape[pack_dim] % feat_per_int == 0, "Dimension length must be divisible by number of features per int"
	code = torch.zeros(shape[:pack_dim] + (shape[pack_dim] // feat_per_int,)+shape[pack_dim+1:], 
					dtype=torch.int32, 
					device=data.device)
	i = 0
	row = 0
	unpacked_indices = [slice(None)] * len(data.shape)
	packed_indices = [slice(None)] * len(data.shape)
	while row < code.shape[pack_dim]:
		packed_indices[pack_dim] = row
		for j in range(i, i + (32 // bits)):
			unpacked_indices[pack_dim] = j
			code[packed_indices] |= data[unpacked_indices] << (bits * (j - i))
		i += 32 // bits
		row += 1
	return code


def unpack_tensor(v_code: torch.FloatTensor, 
				  bits: int, 
				  pack_dim: int):
	assert bits in [1,2,4,8]
	shape = v_code.shape
	feat_per_int = 32 // bits
	new_shape = shape[:pack_dim] + (shape[pack_dim] * feat_per_int,) + shape[pack_dim+1:]
	unpacked_v_code = torch.zeros(new_shape, dtype=torch.int8, device=v_code.device)
	i = torch.arange(new_shape[pack_dim], device=v_code.device) // feat_per_int
	j = torch.arange(new_shape[pack_dim], device=v_code.device) % feat_per_int
	num = 0xFF >> (8 - bits)
	packed_indices = [slice(None)] * len(new_shape)
	packed_indices[pack_dim] = i
	if pack_dim == 2:
		unpacked_v_code = ((v_code[packed_indices] >> (j * bits)[None, None, :, None]).to(torch.int16)) & num
	elif pack_dim == 3:
		unpacked_v_code = ((v_code[packed_indices] >> (j * bits)).to(torch.int16)) & num
	else:
		raise NotImplementedError
	return unpacked_v_code


@triton.jit
def _pack_along_last_dim(
	bits: tl.constexpr,
	intensor_ptr,
	code_ptr,
	N,
	num_feats: tl.constexpr,
	feat_per_int: tl.constexpr,
	BLOCK_SIZE_N: tl.constexpr
):
	num_int_per_y_dim = num_feats // feat_per_int
	bid = tl.program_id(axis=0)
	yid = tl.program_id(axis=1)
	offs_N = bid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
	block_start = intensor_ptr + offs_N * num_feats + yid * feat_per_int
	packed = tl.zeros((BLOCK_SIZE_N,), dtype=tl.int32)
	for i in range(feat_per_int):
		ptr = block_start + i
		element = tl.load(ptr, mask=offs_N<N, other=0.)
		element = element << (i * bits)
		packed = packed | element
	tl.store(code_ptr + offs_N * num_int_per_y_dim + yid, packed, mask=offs_N < N)


@triton.jit
def _minmax_along_last_dim(
	x_ptr,
	mn_ptr, mx_ptr,
	total_elements: tl.constexpr, 
	N: tl.constexpr,
	num_groups: tl.constexpr, 
	group_size: tl.constexpr,
	BLOCK_SIZE_N: tl.constexpr
):
	bid = tl.program_id(axis=0)
	offsets_b = bid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
	offsets = offsets_b[:, None] * group_size + tl.arange(0, group_size)[None, :]
	mask = offsets < total_elements
	x = tl.load(x_ptr + offsets, mask=mask)
	mx_val = tl.max(x, axis=1)
	mn_val = tl.min(x, axis=1)
	tl.store(mn_ptr+offsets_b, mn_val, mask=offsets_b<N*num_groups)
	tl.store(mx_ptr+offsets_b, mx_val, mask=offsets_b<N*num_groups)


def triton_quantize_and_pack_along_last_dim(data: torch.Tensor, group_size: int, bit: float):
    assert len(data.shape) == 4
    B, nh, D, T = data.shape
    assert T % group_size == 0
    num_groups = T // group_size
    scale_mn_shape = (B, nh, D, num_groups)
    orig_shape = data.shape

    if bit == 1:
        data = data.reshape(B * nh * D, num_groups, group_size)
        quantized = torch.where(data >= 0, 1, -1)
        data_q = ((quantized + 1) // 2).to(torch.int32)
        T_total = T
        data_q = data_q.reshape(-1, T_total)
        feat_per_int = 32
        packshape = (int(np.prod(orig_shape[:-1])), T_total // feat_per_int)
        code = torch.zeros(packshape, device=data.device, dtype=torch.int32)
        BLOCK_SIZE_N = 128
        grid = lambda meta: (triton.cdiv(data_q.shape[0], BLOCK_SIZE_N), data_q.shape[1] // feat_per_int)
        with torch.cuda.device(data.device):
            _pack_along_last_dim[grid](1, data_q, code, data_q.shape[0],
                                        data_q.shape[1], feat_per_int,
                                        BLOCK_SIZE_N=BLOCK_SIZE_N, num_warps=8)
        scale_groups = data.abs().mean(dim=-1)
        scale = scale_groups.reshape(scale_mn_shape)
        mn_ret = torch.zeros(scale_mn_shape, device=data.device, dtype=data.dtype)
        return code.view(B, nh, D, -1), scale, mn_ret
    elif bit == 1.58:
        orig_shape = data.shape
        data = data.reshape(B * nh * D, num_groups, group_size)
        scale_groups = data.abs().mean(dim=-1)
        scale = scale_groups.reshape(scale_mn_shape)
        abs_data = data.abs()
        threshold = 0.5 * abs_data.mean(dim=-1, keepdim=True)
        quantized = torch.where(data > threshold, 1, torch.where(data < -threshold, -1, 0))
        data_q = (quantized + 1).to(torch.int32)

        T_total = T
        data_q = data_q.reshape(-1, T_total)

        feat_per_int = 32 // 2
        packshape = (int(np.prod(orig_shape[:-1])), T_total // feat_per_int)
        code = torch.zeros(packshape, device=data.device, dtype=torch.int32)

        BLOCK_SIZE_N = 128
        grid = lambda meta: (triton.cdiv(data_q.shape[0], BLOCK_SIZE_N), data_q.shape[1] // feat_per_int)
        with torch.cuda.device(data.device):
            _pack_along_last_dim[grid](2, data_q, code, data_q.shape[0],
                                       data_q.shape[1], feat_per_int,
                                       BLOCK_SIZE_N=BLOCK_SIZE_N, num_warps=8)
        
        mn_ret = -torch.ones(scale_mn_shape, device=data.device, dtype=data.dtype)
        return code.view(B, nh, D, -1), scale, mn_ret

    else:
        data = data.reshape(B * nh * D, num_groups, group_size)
        mx = torch.empty((B * nh * D, num_groups), device=data.device, dtype=data.dtype)
        mn = torch.empty((B * nh * D, num_groups), device=data.device, dtype=data.dtype)
        BLOCK_SIZE_N = 128
        grid = lambda meta: (triton.cdiv(data.shape[0]*data.shape[1], BLOCK_SIZE_N),)
        with torch.cuda.device(data.device):
            _minmax_along_last_dim[grid](data, mn, mx,
                                         data.numel(), data.shape[0], num_groups, group_size,
                                         BLOCK_SIZE_N=BLOCK_SIZE_N, num_warps=8)
        scale = (mx - mn) / (2 ** bit - 1)
        data = data - mn.unsqueeze(-1)
        data.div_(scale.unsqueeze(-1))
        data = data.clamp_(0, 2 ** bit - 1).round_().to(torch.int32)
        data = data.view(-1, T)
        feat_per_int = 32 // int(bit)
        packshape = (np.prod(orig_shape[:-1]), orig_shape[-1] // feat_per_int)
        code = torch.zeros(*packshape, device=data.device, dtype=torch.int32)
        grid = lambda meta: (triton.cdiv(data.shape[0], BLOCK_SIZE_N), data.shape[1] // feat_per_int)
        with torch.cuda.device(data.device):
            _pack_along_last_dim[grid](bit, data, code, data.shape[0],
                                       data.shape[1], feat_per_int,
                                       BLOCK_SIZE_N=BLOCK_SIZE_N, num_warps=8)
        return code.view(B, nh, D, -1), scale.reshape(scale_mn_shape), mn.reshape(scale_mn_shape)

if __name__ == '__main__':
	torch.manual_seed(0)
	np.random.seed(0)
	random.seed(0)
	B, nh, T, hd = 555, 32, 128, 128
	v = torch.randn((B, nh, T, hd), device='cuda', dtype=torch.float16)
	group_size = 32
	for bits in [1, 1.58, 2, 4, 8]:
		code, scale, mn = triton_quantize_and_pack_along_last_dim(v.transpose(2, 3), group_size, bits)
		memory_bytes = code.numel() * code.element_size()
		print(f"Packed tensor memory usage: {memory_bytes} bytes")
		dequant_v = unpack_and_dequant_vcache(code, scale.unsqueeze(-1), mn.unsqueeze(-1), group_size, bits)
		assert not dequant_v.isnan().any()
		gap = (dequant_v - v) / v
		gap = torch.nan_to_num(gap)
		print(f'bit {bits}, mean v rel arr: {torch.mean(torch.abs(gap))}')