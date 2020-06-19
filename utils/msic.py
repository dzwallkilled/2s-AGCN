import os
import torch
import subprocess


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def to_item(data):
    if isinstance(data, int):
        return data
    elif isinstance(data, torch.Tensor):
        return data.cpu().item()


def get_available_device(num_gpus=1, memory_used=100, memory_available=-1, gpus=[]):
    """
    Retrives the resources and return the available devices according to the requirement
    :param num_gpus: int, num of gpus to be used
    :param memory_used: int Mb, gpus with memory used less than this value, negative value ignores this option
    :param memory_available: int Mb, gpus with memory available larger than this value, negative value ignores this option
    :param gpus: list(int), the gpu range constrained, e.g. gpus=[0, 1, 2], allocating memory only to these gpus
    :return: torch.device('cuda') or torch.device('cpu')
    """
    print(f'Requirement: {num_gpus} GPUs with >{memory_available}M available and <{memory_used}M used')
    if memory_used < 0:
        memory_used = 1e10
    if memory_available < 0:
        memory_available = 1e10
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    gpu_memory_used = [int(x) for x in result.strip().split('\n')]

    result2 = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.free',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    gpu_memory_free = [int(x) for x in result2.strip().split('\n')]

    free_gpus = []
    for gpu, (mem_used, mem_free) in enumerate(zip(gpu_memory_used, gpu_memory_free)):
        if mem_used < memory_used or mem_free > memory_available:
            free_gpus.append(gpu)

    if gpus:
        free_gpus = [gpu for gpu in free_gpus if gpu in gpus]

    if num_gpus == 0:
        print(f'Allocating memory into CPU.')
        return torch.device('cpu')
    elif len(free_gpus) < num_gpus:
        print(f"Not enough GPUs available. {num_gpus} required but {len(free_gpus)} available.")
        print(f"Allocating memory into CPU.")
        return torch.device('cpu')
    else:
        gpus = ','.join(str(gpu) for gpu in free_gpus[:num_gpus])
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(gpu) for gpu in free_gpus[:num_gpus])
        print(f'Allocating memory into GPU: {gpus}')
        return torch.device('cuda')