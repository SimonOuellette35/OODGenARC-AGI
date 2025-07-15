import numpy as np
import random
from functools import partial

MAX_GRID_SIZE = 30


def apply_data_augmentation(task, hflip, n_rot90, color_map=None):
    augmented_task = _apply_augmentation_to_task(task, partial(geometric_augmentation, hflip=hflip, n_rot90=n_rot90))
    if color_map is not None:
        augmented_task = swap_task_colors(augmented_task, color_map)
    augmented_task = permute_train_samples(augmented_task)
    return augmented_task


def revert_data_augmentation(grid, hflip, n_rot90, color_map=None):
    grid = revert_geometric_augmentation(grid, hflip, n_rot90)
    if color_map is not None:
        grid = revert_color_swap(grid, color_map)
    return grid


def apply_data_augmentation_star(task, hflip, n_rot90, color_map=None):
    augmented_task = permute_train_samples(task)
    return augmented_task


def random_augment_task_star(task, swap_train_and_test=True):
    augmented_task = apply_data_augmentation_star(
        task, color_map=get_random_color_map(), **get_random_geometric_augmentation_params())
    if swap_train_and_test:
        augmented_task = random_swap_train_and_test(augmented_task)
    return augmented_task

def random_augment_task(task, swap_train_and_test=True):
    augmented_task = apply_data_augmentation(
        task, color_map=get_random_color_map(), **get_random_geometric_augmentation_params())
    if swap_train_and_test:
        augmented_task = random_swap_train_and_test(augmented_task)
    return augmented_task


def random_compose_new_task_by_adding_additional_transformation(task, weights=None, verbose=False, is_wrong_prediction=False):
    """
    Creates a new task by randomly applying transformations to the inputs or the outputs

    Parameters
    ----------
    task : dict
        The task to be transformed
    weights: list
        The weights for the different transformations. The order is ['geometric', 'padding', 'upscale', 'mirror']
    """
    augmentation_targets = [random.choice(['input', 'output'])]
    if is_wrong_prediction and augmentation_targets[0] == 'output':
        augmentation_targets.append('wrong_prediction')

    augmentation_map = {
        'geometric': (geometric_augmentation, get_random_geometric_augmentation_params),
        'padding': (add_padding, get_random_padding_params),
        'upscale': (upscale, get_random_upscale_params),
        'mirror': (mirror, get_random_mirror_params),
        # TODO: does it have sense to add also color swap? I believe it might make the tasks too hard
    }
    if weights is not None and len(weights) != len(augmentation_map):
        raise ValueError(f"weights must have the same length as the number of augmentations {len(augmentation_map)} but has {len(weights)}")

    try:
        augmentation = random.choices(list(augmentation_map.keys()), weights=weights)[0]
        if verbose: print(f"Applying {augmentation} augmentation to {augmentation_targets}")

        aug_func, param_func = augmentation_map[augmentation]
        if augmentation == 'geometric':
            kwargs = param_func()
        else:
            max_grid_shape = get_max_grid_shape(task, augmentation_targets)
            kwargs = param_func(max_grid_shape)
        new_task = _apply_augmentation_to_task(
            task,
            partial(aug_func, **kwargs),
            augmentation_targets=augmentation_targets
        )
    except GridTooBigToAugmentError as e:
        if verbose: print(e)
        return task

    return new_task


def _apply_augmentation_to_task(task, augmentation, augmentation_targets=None):
    augmented_task = dict()
    for partition, samples in task.items():
        augmented_task[partition] = [_augment_sample(sample, augmentation, augmentation_targets) for sample in samples]
    return augmented_task


def _augment_sample(sample, augmentation, augmentation_targets=None):
    if augmentation_targets is None:
        return {name:augmentation(grid) for name, grid in sample.items()}
    else:
        if all(augmentation_target not in sample for augmentation_target in augmentation_targets):
            raise ValueError(f"augmentation_target {augmentation_targets} not found in sample")
        return {name:augmentation(grid) if name in augmentation_targets else grid for name, grid in sample.items()}


def get_random_geometric_augmentation_params():
    return dict(hflip=random.choice([True, False]), n_rot90=random.choice([0, 1, 2, 3]))


def geometric_augmentation(grid, hflip, n_rot90):
    grid = np.array(grid)
    if hflip:
        grid = np.flip(grid, axis=1)
    grid = np.rot90(grid, k=n_rot90)
    return grid.tolist()


def revert_geometric_augmentation(grid, hflip, n_rot90):
    grid = np.array(grid)
    grid = np.rot90(grid, k=-n_rot90)
    if hflip:
        grid = np.flip(grid, axis=1)
    return grid.tolist()


def swap_task_colors(task, color_map=None, change_background_probability=0.1):
    if color_map is None:
        color_map = get_random_color_map(change_background_probability)
    vectorized_mapping = np.vectorize(color_map.get)

    new_task = dict()
    for key in task.keys():
        new_task[key] = [{name:vectorized_mapping(grid).tolist() for name, grid in sample.items()} for sample in task[key]]
    return new_task


def revert_color_swap(grid, color_map):
    reverse_color_map = {v: k for k, v in color_map.items()}
    vectorized_mapping = np.vectorize(reverse_color_map.get)
    return vectorized_mapping(grid).tolist()


def get_random_color_map(change_background_probability=0.1):
    colors = list(range(10))
    if random.random() < change_background_probability:
        new_colors = list(range(10))
        random.shuffle(new_colors)
    else:
        new_colors = list(range(1, 10))
        random.shuffle(new_colors)
        new_colors = [0] + new_colors

    color_map = {x: y for x, y in zip(colors, new_colors)}
    return color_map


def permute_train_samples(task):
    train_order = np.arange(len(task['train']))
    np.random.shuffle(train_order)
    augmented_task = dict()
    augmented_task['train'] = [task['train'][idx] for idx in train_order]
    augmented_task['test'] = task['test']
    return augmented_task


def random_swap_train_and_test(task):
    augmented_task = task.copy()
    all_samples = task['train'] + task['test']
    random.shuffle(all_samples)
    train_size = len(task['train'])
    augmented_task['train'] = all_samples[:train_size]
    augmented_task['test'] = all_samples[train_size:]
    return augmented_task


def set_random_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)


class GridTooBigToAugmentError(Exception):
    pass


def get_max_grid_shape(task, augmentation_targets):
    max_shape = (0, 0)
    for _, samples in task.items():
        for sample in samples:
            for augmentation_target in augmentation_targets:
                if augmentation_target in sample:
                    grid = sample[augmentation_target]
                    max_shape = (max(max_shape[0], len(grid)), max(max_shape[1], len(grid[0])))
    return max_shape


def add_padding(grid, color, size):
    cols = len(grid[0])
    padded_grid = [[color]*(cols + size[1]*2) for _ in range(size[0])]
    for row in grid:
        padded_grid.append([color]*size[1] + row + [color]*size[1])
    padded_grid += [[color]*(cols + size[1]*2) for _ in range(size[0])]
    return padded_grid


def get_random_padding_params(max_grid_shape, same_size_probability=0.5, max_padding=5, n_tries=10):
    safe_max_padding = (min(MAX_GRID_SIZE - max_grid_shape[0], max_padding),
                        min(MAX_GRID_SIZE - max_grid_shape[1], max_padding))
    if random.random() < same_size_probability:
        safe_max_padding = min(safe_max_padding)
        if safe_max_padding < 1:
            raise GridTooBigToAugmentError(f"Grid is too big to pad: {max_grid_shape}")
        size = random.randint(1, safe_max_padding)
        size = (size, size)
    else:
        if min(safe_max_padding) < 1:
            raise GridTooBigToAugmentError(f"Grid is too big to pad: {max_grid_shape}")
        for _ in range(n_tries):
            size = (random.randint(1, safe_max_padding[0]), random.randint(1, safe_max_padding[1]))
            if size[0] != size[1]:
                break
    color = random.randint(0, 9)
    return dict(color=color, size=size)


def upscale(grid, scale):
    grid = np.array(grid, dtype=int)
    for axis, scale in enumerate(scale):
        grid = np.repeat(grid, scale, axis=axis)
    return grid.tolist()


def get_random_upscale_params(max_grid_shape, min_upscale=2, max_upscale=4,
                              same_upscale_probability=0.5, n_tries=10):
    safe_max_upscale = (min(MAX_GRID_SIZE // max_grid_shape[0], max_upscale),
                        min(MAX_GRID_SIZE // max_grid_shape[1], max_upscale))
    if random.random() < same_upscale_probability:
        safe_max_upscale = min(safe_max_upscale)
        if safe_max_upscale < 2:
            raise GridTooBigToAugmentError(f"Grid is too big to upscale: {max_grid_shape}")
        scale = random.randint(min_upscale, safe_max_upscale)
        return dict(scale=(scale, scale))
    else:
        if max(safe_max_upscale) < 2 or min(safe_max_upscale) < 1:
            raise GridTooBigToAugmentError(f"Grid is too big to upscale: {max_grid_shape}")
        min_upscale = 1
        for _ in range(n_tries):
            scale = (random.randint(min_upscale, safe_max_upscale[0]),
                     random.randint(min_upscale, safe_max_upscale[1]))
            if scale[0] != scale[1]:
                break
        return dict(scale=scale)


def mirror(grid, axis, position):
    if axis == 0:
        if position == 0:
            return grid[::-1] + grid
        else:
            return grid + grid[::-1]
    elif axis == 1:
        new_grid = []
        for row in grid:
            if position == 0:
                new_grid.append(row[::-1] + row)
            else:
                new_grid.append(row + row[::-1])
        return new_grid


def get_random_mirror_params(max_grid_shape):
    if MAX_GRID_SIZE // max_grid_shape[0] < 2:
        if MAX_GRID_SIZE // max_grid_shape[1] < 2:
            raise GridTooBigToAugmentError(f"Grid is too big to mirror: {max_grid_shape}")
        else:
            axis = 1
    elif MAX_GRID_SIZE // max_grid_shape[1] < 2:
        axis = 0
    else:
        axis = random.randint(0, 1)
    return dict(axis=axis, position=random.randint(0, 1))



if __name__ == '__main__':
    sample_grid = np.eye(3, dtype=int).tolist()
    for flip in [True, False]:
        for n_rot90 in range(4):
            assert sample_grid == revert_geometric_augmentation(geometric_augmentation(sample_grid, flip, n_rot90), flip, n_rot90)
