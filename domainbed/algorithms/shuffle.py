import torch
import random
import matplotlib.pyplot as plt
import torchvision.transforms as T
import numpy as np
import torchvision.transforms.functional as TF
def random_swap_blocks_corners(x_batch, patch_size=4, k=20): # 单张图打乱（前k个位置顺序交换）
    batch_size, channels, height, width = x_batch.size()
    patch_height = height // patch_size
    patch_width = width // patch_size
    num_patches = patch_height * patch_width
    
    # Define indices for each corner
    top_left = [(i, j) for i in range(patch_height // 2) for j in range(patch_width // 2)]
    top_right = [(i, j) for i in range(patch_height // 2) for j in range(patch_width // 2, patch_width)]
    bottom_left = [(i, j) for i in range(patch_height // 2, patch_height) for j in range(patch_width // 2)]
    bottom_right = [(i, j) for i in range(patch_height // 2, patch_height) for j in range(patch_width // 2, patch_width)]

    # Combine all corner indices
    all_corners = top_left + top_right + bottom_left + bottom_right

    # Randomly select k unique patches from all corners
    if k > len(all_corners):
        k = len(all_corners)
    shuffled_indices = random.sample(all_corners, k)
    
    # Create a new_x_batch tensor and initialize with the original x_batch
    new_x_batch = x_batch.clone()
    
    # Rearrange only the first k patches according to shuffled_indices
    for idx in range(batch_size):
        patches = []
        for (i, j) in shuffled_indices:
            patches.append(x_batch[idx, :, i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size])
        random.shuffle(patches)
        for patch_idx, (i, j) in enumerate(shuffled_indices):
            new_x_batch[idx, :, i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size] = patches[patch_idx]
    
    return new_x_batch




def random_swap_blocks_across_images(x_batch, patch_size=4, k=20): # 不同图之间打乱
    batch_size, channels, height, width = x_batch.size()
    patch_height = height // patch_size
    patch_width = width // patch_size
    
    # Define indices for each corner
    top_left = [(i, j) for i in range(patch_height // 2) for j in range(patch_width // 2)]
    top_right = [(i, j) for i in range(patch_height // 2) for j in range(patch_width // 2, patch_width)]
    bottom_left = [(i, j) for i in range(patch_height // 2, patch_height) for j in range(patch_width // 2)]
    bottom_right = [(i, j) for i in range(patch_height // 2, patch_height) for j in range(patch_width // 2, patch_width)]
    
    # Combine all corner indices
    all_corners = top_left + top_right + bottom_left + bottom_right
    
    # Randomly select k unique patches from all corners
    if k > len(all_corners):
        k = len(all_corners)
    shuffled_indices = random.sample(all_corners, k)
    
    # Create a new_x_batch tensor and initialize with the original x_batch
    new_x_batch = x_batch.clone()
    
    # Rearrange the patches across different images
    for (i, j) in shuffled_indices:
        # Randomly select two images to swap patches
        img_indices = random.sample(range(batch_size), 2)
        img_idx1, img_idx2 = img_indices[0], img_indices[1]
        
        # Extract patches from the two images
        patch1 = x_batch[img_idx1, :, i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size]
        patch2 = x_batch[img_idx2, :, i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size]
        
        # Swap the patches
        new_x_batch[img_idx1, :, i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size] = patch2
        new_x_batch[img_idx2, :, i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size] = patch1
    
    return new_x_batch

# x_batch = torch.randn(4, 3, 224, 224) # Example batch of 4 images

def random_blur_blocks(x_batch, patch_size=4, k=20):  # 某些块发生模糊
    batch_size, channels, height, width = x_batch.size()
    patch_height = height // patch_size
    patch_width = width // patch_size
    num_patches = patch_height * patch_width
    
    # Define indices for each corner
    top_left = [(i, j) for i in range(patch_height // 2) for j in range(patch_width // 2)]
    top_right = [(i, j) for i in range(patch_height // 2) for j in range(patch_width // 2, patch_width)]
    bottom_left = [(i, j) for i in range(patch_height // 2, patch_height) for j in range(patch_width // 2)]
    bottom_right = [(i, j) for i in range(patch_height // 2, patch_height) for j in range(patch_width // 2, patch_width)]

    # Combine all corner indices
    all_corners = top_left + top_right + bottom_left + bottom_right

    # Randomly select k unique patches from all corners
    if k > len(all_corners):
        k = len(all_corners)
    selected_indices = random.sample(all_corners, k)
    
    # Create a new_x_batch tensor and initialize with the original x_batch
    new_x_batch = x_batch.clone()
    
    # Apply blur to the selected patches
    blur = T.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0))
    
    for idx in range(batch_size):
        for (i, j) in selected_indices:
            patch = x_batch[idx, :, i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size]
            blurred_patch = blur(patch)
            new_x_batch[idx, :, i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size] = blurred_patch
    
    return new_x_batch


def random_zero_blocks(x_batch, patch_size=4, k=20): # 屏蔽某些块
    batch_size, channels, height, width = x_batch.size()
    patch_height = height // patch_size
    patch_width = width // patch_size
    
    # Define indices for each corner
    top_left = [(i, j) for i in range(patch_height // 2) for j in range(patch_width // 2)]
    top_right = [(i, j) for i in range(patch_height // 2) for j in range(patch_width // 2, patch_width)]
    bottom_left = [(i, j) for i in range(patch_height // 2, patch_height) for j in range(patch_width // 2)]
    bottom_right = [(i, j) for i in range(patch_height // 2, patch_height) for j in range(patch_width // 2, patch_width)]
    
    # Combine all corner indices
    all_corners = top_left + top_right + bottom_left + bottom_right
    
    # Randomly select k unique patches from all corners
    if k > len(all_corners):
        k = len(all_corners)
    selected_indices = random.sample(all_corners, k)
    
    # Create a new_x_batch tensor and initialize with the original x_batch
    new_x_batch = x_batch.clone()
    
    # Set the selected patches to zero
    for idx in range(batch_size):
        for (i, j) in selected_indices:
            new_x_batch[idx, :, i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size] = 0
    
    return new_x_batch

# for test
def save_images(original, modified, original_path='original_image.png', modified_path='modified_image.png'): 
    # Clip the tensor values to be in the range [0, 1]
    original = original.clamp(0, 1)
    modified = modified.clamp(0, 1)
    # modified = modified - original
    # modified = modified.clamp(0, 1)

    
    # Original image
    plt.imsave(original_path, original.permute(1, 2, 0).cpu().numpy())
    
    # Modified image
    plt.imsave(modified_path, modified.permute(1, 2, 0).cpu().numpy())


def create_color_batch(batch_size, channels, height, width):
    # Define some colors (RGB format)
    colors = [
        (0, 0, 255),  # Blue
        (255, 0, 0),  # Red
        (0, 255, 0),  # Green
        (255, 255, 0),  # Yellow
        (255, 165, 0),  # Orange
        (0, 255, 255),  # Cyan
        (255, 0, 255),  # Magenta
        (128, 0, 128),  # Purple
        (165, 42, 42),  # Brown
        (255, 192, 203),  # Pink
    ]
    
    # Ensure we have enough colors for the batch size
    assert batch_size <= len(colors), "Not enough colors to create the batch"
    
    # Create the batch tensor
    x_batch = torch.zeros(batch_size, channels, height, width)
    
    for i in range(batch_size):
        color = colors[i]
        for c in range(channels):
            x_batch[i, c, :, :] = color[c]
    
    return x_batch




def create_gradient_batch(batch_size, channels, height, width):
    x_batch = torch.zeros(batch_size, channels, height, width, dtype=torch.float32)
    
    for i in range(batch_size):
        gradient = np.linspace(0, 1, width).astype(np.float32)  # Change range to [0, 1] for correct tensor scaling
        for c in range(channels):
            x_batch[i, c, :, :] = torch.tensor(gradient).unsqueeze(0).repeat(height, 1)
    
    return x_batch



def random_rotate_blocks(x_batch, patch_size=4, k=20, rotate_angles=[0, 90, 180, 270]):
    batch_size, channels, height, width = x_batch.size()
    patch_height = height // patch_size
    patch_width = width // patch_size
    
    # Define indices for each corner
    top_left = [(i, j) for i in range(patch_height // 2) for j in range(patch_width // 2)]
    top_right = [(i, j) for i in range(patch_height // 2) for j in range(patch_width // 2, patch_width)]
    bottom_left = [(i, j) for i in range(patch_height // 2, patch_height) for j in range(patch_width // 2)]
    bottom_right = [(i, j) for i in range(patch_height // 2, patch_height) for j in range(patch_width // 2, patch_width)]
    
    # Combine all corner indices
    all_corners = top_left + top_right + bottom_left + bottom_right
    
    # Randomly select k unique patches from all corners
    if k > len(all_corners):
        k = len(all_corners)
    selected_indices = random.sample(all_corners, k)
    
    # Create a new_x_batch tensor and initialize with the original x_batch
    new_x_batch = x_batch.clone()
    
    # Rotate the selected patches
    for idx in range(batch_size):
        for (i, j) in selected_indices:
            angle = random.choice(rotate_angles)
            patch = x_batch[idx, :, i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size]
            rotated_patch = TF.rotate(patch, angle)
            new_x_batch[idx, :, i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size] = rotated_patch
    
    return new_x_batch


def random_rotate_blocks_corners(x_batch, patch_size=4, k=20): # 对前k个位置的patch进行随机旋转
    batch_size, channels, height, width = x_batch.size()
    patch_height = height // patch_size
    patch_width = width // patch_size
    patch_height = int(patch_height)
    patch_width = int(patch_width)
    
    
    # Define indices for each corner
    top_left = [(i, j) for i in range(patch_height // 2) for j in range(patch_width // 2)]
    top_right = [(i, j) for i in range(patch_height // 2) for j in range(patch_width // 2, patch_width)]
    bottom_left = [(i, j) for i in range(patch_height // 2, patch_height) for j in range(patch_width // 2)]
    bottom_right = [(i, j) for i in range(patch_height // 2, patch_height) for j in range(patch_width // 2, patch_width)]

    # Combine all corner indices
    all_corners = top_left + top_right + bottom_left + bottom_right

    # Randomly select k unique patches from all corners
    if k > len(all_corners):
        k = len(all_corners)
    selected_indices = random.sample(all_corners, k)
    
    # Create a new_x_batch tensor and initialize with the original x_batch
    new_x_batch = x_batch.clone()
    
    # Function to rotate a patch by a given angle
    def rotate_patch(patch, angle):
        if angle == 90:
            return patch.transpose(1, 2).flip(1)
        elif angle == 180:
            return patch.flip(1).flip(2)
        elif angle == 270:
            return patch.transpose(1, 2).flip(2)
        else:  # angle == 360
            return patch

    # Rotate only the first k patches according to selected_indices
    for idx in range(batch_size):
        for (i, j) in selected_indices:
            patch = x_batch[idx, :, i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size]
            angle = random.choice([90, 180, 270, 360])
            rotated_patch = rotate_patch(patch, angle)
            
            
            new_x_batch[idx, :, i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size] = rotated_patch
    
    return new_x_batch

def add_learnable_vectors(x_batch, learnable_vectors, patch_size=4):
    batch_size, channels, height, width = x_batch.size()
    patch_height = height // patch_size
    patch_width = width // patch_size
    k = learnable_vectors.size(1)  # Number of learnable vectors for each corner

    # Define indices for each corner
    top_left = [(i, j) for i in range(patch_height // 2) for j in range(patch_width // 2)]
    top_right = [(i, j) for i in range(patch_height // 2) for j in range(patch_width // 2, patch_width)]
    bottom_left = [(i, j) for i in range(patch_height // 2, patch_height) for j in range(patch_width // 2)]
    bottom_right = [(i, j) for i in range(patch_height // 2, patch_height) for j in range(patch_width // 2, patch_width)]

    # Combine all corner indices
    all_corners = [top_left, top_right, bottom_left, bottom_right]

    # Ensure k does not exceed the number of patches
    if k > len(top_left):
        raise ValueError("The number of learnable vectors exceeds the number of available patches in each corner.")

    # Select the first k patches from each corner
    selected_indices = [corner[:k] for corner in all_corners]

    # Create a new_x_batch tensor and initialize with the original x_batch
    new_x_batch = x_batch.clone()

    # Add the learnable vectors to the selected patches
    for idx in range(batch_size):
        for corner_idx, corner in enumerate(selected_indices):
            for i, (i_patch, j_patch) in enumerate(corner):
                patch = x_batch[idx, :, i_patch * patch_size:(i_patch + 1) * patch_size, j_patch * patch_size:(j_patch + 1) * patch_size]
                new_patch = patch + learnable_vectors[corner_idx, i]
                new_x_batch[idx, :, i_patch * patch_size:(i_patch + 1) * patch_size, j_patch * patch_size:(j_patch + 1) * patch_size] = new_patch
    
    return new_x_batch

# 似乎上面不是指定前k个随机发生 而是发生k个 查看一下 这里面似乎有随机数没控制好
# Example usage
# batch_size = 2
# channels = 3
# height = 224
# width = 224
# x_batch = torch.randn(batch_size, channels, height, width)
# k = 5  # Number of learnable vectors for each corner
# learnable_vectors = torch.nn.Parameter(torch.randn(4, k, channels, 4, 4))  # 4 corners, each with k learnable vectors

# new_x_batch = add_learnable_vectors(x_batch, learnable_vectors)
# print(new_x_batch)

# Example usage
# batch_size = 2
# channels = 3
# height = 8
# width = 8
# x_batch = torch.randn(batch_size, channels, height, width)
# learnable_vectors = torch.nn.Parameter(torch.randn(4, channels, 4, 4))  # 4 learnable vectors for 4 corners

# new_x_batch = add_learnable_vectors(x_batch, learnable_vectors)
# print(new_x_batch)
# new_x_batch = random_rotate_blocks_corners(x_batch, patch_size=patch_size) # 随机旋转
# save_images(x_batch[1], new_x_batch [1], '/home/home_node7/wzb/自己算法/miro-main2/domainbed/algorithms/original_image.png', '/home/home_node7/wzb/自己算法/miro-main2/domainbed/algorithms/modified_image.png')