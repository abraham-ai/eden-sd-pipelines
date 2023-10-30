
import os, torch, random, time, glob
import numpy as np
import matplotlib.pyplot as plt
from settings import _device
import seaborn as sns

def visualize_distribution(args, savename):
    plot(args.c, savename + "_c.png")

def plot(vector, name):
    # Ensure the array is a numpy array
    vector = np.array(vector.cpu().numpy())
    
    # Plotting
    fig, axs = plt.subplots(3, 2, figsize=(24, 12))
    
    # Mean and std for each of the 77 tokens
    ax1 = axs[0, 0]
    ax1_2 = ax1.twinx()
    l1 = ax1.errorbar(range(77), np.mean(vector, axis=2)[0], fmt='o', label='Mean')[0]
    l2 = ax1_2.errorbar(range(77), np.std(vector, axis=2)[0], fmt='xr', label='Std Dev')[0]
    ax1.set_ylabel('Mean')
    ax1_2.set_ylabel('Std Dev')
    #ax1.set_ylim([0, 50])
    #ax1_2.set_ylim([0, 50])
    ax1.legend([l1, l2], ['Mean', 'Std Dev'])
    ax1.set_title("Mean and Std Dev for 77 Tokens")
    
    # Mean and std for each of the 2048 dimensions
    ax2 = axs[0, 1]
    ax2_2 = ax2.twinx()
    l1 = ax2.errorbar(range(2048), np.mean(vector, axis=1)[0], fmt='o', label='Mean')[0]
    l2 = ax2_2.errorbar(range(2048), np.std(vector, axis=1)[0], fmt='xr', label='Std Dev')[0]
    ax2.set_ylabel('Mean')
    ax2_2.set_ylabel('Std Dev')
    ax2.legend([l1, l2], ['Mean', 'Std Dev'])
    ax2.set_title("Mean and Std Dev for 2048 Dimensions")

    # Norm of each of the 77 tokens
    axs[1, 0].plot(np.linalg.norm(vector, axis=2)[0])
    axs[1, 0].set_ylim([0, 50])
    axs[1, 0].set_title("Norms of 77 Tokens")
    
    # Norm of each of the 2048 dimensions
    axs[1, 1].plot(np.linalg.norm(vector, axis=1)[0])
    axs[1, 1].set_ylim([0, 15])
    axs[1, 1].set_title("Norms of 2048 Dimensions")

    axs[2, 0].boxplot(vector[0].flatten())
    axs[2, 0].set_yscale('log')
    axs[2, 0].set_title("Boxplot for Outliers (Log Scale)")
        
    # Additional: Heatmap of the 77 tokens across 2048 dimensions
    sns.heatmap(vector[0], cmap="coolwarm", ax=axs[2, 1], vmin=-4, vmax=4)
    axs[2, 1].set_title("Heatmap of 77 Tokens Across 2048 Dimensions")
    
    plt.tight_layout()
    plt.savefig(name)
    plt.close()
    plt.clf()

try:
    raw_embeddings_path = "/data/xander/Projects/cog/eden-sd-pipelines/eden/xander/latent_hacking/prompts_mini_raw.npy"
    stats_embeddings_path = "/data/xander/Projects/cog/eden-sd-pipelines/eden/xander/latent_hacking/prompts_mini_stats.npy"
    raw_embeddings_path = "/data/xander/Projects/cog/eden-sd-pipelines/eden/xander/latent_hacking/prompts_raw.npy"
    stats_embeddings_path = "/data/xander/Projects/cog/eden-sd-pipelines/eden/xander/latent_hacking/prompts_stats.npy"
    
    raw_embeddings = torch.load(raw_embeddings_path)
    n_elem = raw_embeddings["c"].shape[0]
    print(f"Loaded {n_elem} raw embeddings")


    #stats_dict = np.load(stats_embeddings_path, allow_pickle=True).item()

    means = raw_embeddings["c"].mean(dim=0).squeeze().float()
    stds = raw_embeddings["c"].std(dim=0).squeeze().float()
    print(means.shape)
    print(stds.shape)



    # get embeddings stats:
    q_low, q_high = 0.05, 0.95
    stats_dict = {}
    for key in raw_embeddings.keys():
        stats_dict[f"{key}_std"]  = raw_embeddings[key].std(dim=0).squeeze().float()
        stats_dict[f"{key}_mean"] = raw_embeddings[key].mean(dim=0).squeeze().float()
        stats_dict[f"{key}_min"] = torch.quantile(raw_embeddings[key], q_low, dim=0).float()
        stats_dict[f"{key}_max"] = torch.quantile(raw_embeddings[key], q_high, dim=0).float()

    condition_save_dir = "/data/xander/Projects/cog/eden-sd-pipelines/eden/xander/latent_hacking/good_ip_conditions"
    ip_conditions = []
    for ip_condition_path in glob.glob(os.path.join(condition_save_dir, "*.npy")):
        ip_condition = np.load(ip_condition_path)
        ip_conditions.append(ip_condition)

    ip_conditions = np.array(ip_conditions)
    ip_means, ip_stds = np.mean(ip_conditions, axis=0), np.std(ip_conditions, axis=0)
    ip_mins, ip_maxs  = np.min(ip_conditions, axis=0), np.max(ip_conditions, axis=0)

except Exception as e:
    print("Error loading latent_magic embeddings, you can ignore this..")
    print(str(e))
    pass


def sample_random_conditioning(args):
    return sample_random_gaussian(args)
    #return mix_latents(args)
    #return sample_random_ip_conditioning(args)



def random_row_sampling(samples):
    shape = samples.shape
    n_rows = shape[0]
    feature_shape = shape[1:]

    if (len(feature_shape) > 1) and 1:
        # Generate random indices only for the first column of feature_shape
        random_indices = torch.randint(0, n_rows, (1, feature_shape[0]), device=samples.device)

        # Expand dimensions to match original tensor shape
        random_indices = random_indices.unsqueeze(-1).expand(-1, -1, *feature_shape[1:])
    else:
        # Generate random indices for each "column"
        random_indices = torch.randint(0, n_rows, (1, *feature_shape), device=samples.device)

    # Create the sampled tensor
    sample = torch.gather(samples, 0, random_indices)

    return sample

def mix_latents(args):
    """
    randomly grabs n embeddings from the raw embeddings and mixes them (using genetic crossover)

    # embeddings were saved like this (where each value is a stack torch tensor)
    embeddings = {
        "c": all_prompt_embeds,
        "uc": all_negative_prompt_embeds,
        "pc": all_pooled_prompt_embeds,
        "puc": all_negative_pooled_prompt_embeds
    }
    torch.save(embeddings, save_path_raw)
    print(f"Succesfully saved raw embeddings to {save_path_raw}")
    """

    n = random.choice([4,8,16,32,64,128])
    n = min(n, raw_embeddings["c"].shape[0])
    args.n_mixed = n
    indices = np.random.choice(raw_embeddings["c"].shape[0], n, replace=False)

    # grab n random embeddings:
    random_embeddings = {}
    for key in raw_embeddings.keys():
        random_embeddings[key] = raw_embeddings[key][indices].squeeze()

        if (random_embeddings[key].shape[0] == 77) or (random_embeddings[key].shape[0] == 1280):
            random_embeddings[key] = random_embeddings[key].unsqueeze(0)

        # sample randomly from the embeddings:
        sampled_conditioning = random_row_sampling(random_embeddings[key])

        if 1:
            # post process:
            sampled_conditioning = sampled_conditioning*args.conditioning_sigma
            if args.clamp_factor:
                sampled_conditioning = torch.clamp(sampled_conditioning, args.clamp_factor * stats_dict[f"{key}_min"], args.clamp_factor * stats_dict[f"{key}_max"])

        setattr(args, key, sampled_conditioning.to(_device))

    return args

def sample_random_gaussian(args):

    for key in ["c"]:
        mean = stats_dict[f"{key}_mean"]
        min_val = stats_dict[f"{key}_min"]
        max_val = stats_dict[f"{key}_max"]
        std = args.conditioning_sigma * stats_dict[f"{key}_std"]

        # Sample from Gaussian
        sampled_conditioning = torch.normal(mean, std).unsqueeze(0)/3

        if args.clamp_factor:
            sampled_conditioning = torch.clamp(sampled_conditioning, args.clamp_factor * stats_dict[f"{key}_min"], args.clamp_factor * stats_dict[f"{key}_max"])

        # Convert to PyTorch tensor and move to device
        setattr(args, key, sampled_conditioning.to(_device))

    return args



def save_ip_img_condition(args):
    ip_img_conditioning = args.c[0,77:,:].detach().clone().cpu().numpy()
    # save the ip_img_conditioning to disk:
    os.makedirs(condition_save_dir, exist_ok=True)
    np.save(os.path.join(condition_save_dir, f"ip_cond_{int(time.time())}.npy"), ip_img_conditioning)

def sample_random_ip_conditioning(args):
    # sample a random ip_condition:
    sigma = 1.0
    ip_condition = np.random.normal(ip_means, sigma*ip_stds)

    # uniform:
    #ip_condition = np.random.uniform(ip_mins, ip_maxs)

    args_c_clone = args.c.clone()
    args_c_clone[0, 77:, :] = torch.from_numpy(ip_condition).unsqueeze(0).to(_device)
    args.c = args_c_clone

    return args