
import os, torch, random, time, glob
import numpy as np
import matplotlib.pyplot as plt
from settings import _device

def save_plots(ip_means, ip_stds, ip_mins, ip_maxs):

    import seaborn as sns

    ip_means = ip_means.astype(np.float32)
    ip_stds = ip_stds.astype(np.float32)
    ip_mins = ip_mins.astype(np.float32)
    ip_maxs = ip_maxs.astype(np.float32)

    # Set up the figure size
    plt.figure(figsize=(12, 8))
    
    # Line Plots for Mean, Std, Min, Max
    plt.subplot(2, 2, 1)
    for i in range(ip_means.shape[0]):
        plt.plot(ip_means[i], label=f"Feature {i+1} Mean")
    plt.title('Means Across Feature Dimensions')
    plt.legend()

    plt.subplot(2, 2, 2)
    for i in range(ip_stds.shape[0]):
        plt.plot(ip_stds[i], label=f"Feature {i+1} Std")
    plt.title('Standard Deviations Across Feature Dimensions')
    plt.legend()

    # Histogram
    plt.subplot(2, 2, 3)
    sns.histplot(ip_means.flatten(), kde=True, label='Mean')
    sns.histplot(ip_stds.flatten(), kde=True, label='Std')
    plt.title('Histogram of Means and Stds')
    plt.legend()

    # Box Plot
    plt.subplot(2, 2, 4)
    sns.boxplot(data=ip_means)
    plt.title('Box Plot of Means')
    
    plt.tight_layout()
    plt.savefig('basic_stats_plots.png')
    plt.close()

    # Heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(ip_means, annot=True, fmt=".2f")
    plt.title('Heatmap of Means')
    plt.savefig('heatmap_means.png')
    plt.close()

    # Correlation Matrix
    correlation_matrix = np.corrcoef(ip_means)
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f")
    plt.title('Correlation Matrix of Means')
    plt.savefig('correlation_matrix.png')
    plt.close()


try:
    raw_embeddings_path = "/data/xander/Projects/cog/eden-sd-pipelines/eden/xander/latent_hacking/prompts_mini_raw.npy"
    stats_embeddings_path = "/data/xander/Projects/cog/eden-sd-pipelines/eden/xander/latent_hacking/prompts_mini_stats.npy"
    raw_embeddings_path = "/data/xander/Projects/cog/eden-sd-pipelines/eden/xander/latent_hacking/prompts_raw.npy"
    stats_embeddings_path = "/data/xander/Projects/cog/eden-sd-pipelines/eden/xander/latent_hacking/prompts_stats.npy"


    raw_embeddings = torch.load(raw_embeddings_path)
    n_elem = raw_embeddings["c"].shape[0]
    print(f"Loaded {n_elem} raw embeddings")


    stats_dict = np.load(stats_embeddings_path, allow_pickle=True).item()

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
    #return sample_random_gaussian(args)
    return mix_latents(args)
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

    for key in ["c", "uc", "pc", "puc"]:
        mean = stats_dict[f"{key}_mean"]
        min_val = stats_dict[f"{key}_min"]
        max_val = stats_dict[f"{key}_max"]
        std = args.conditioning_sigma * stats_dict[f"{key}_std"]

        # Sample from Gaussian
        sampled_conditioning = torch.normal(mean, std).unsqueeze(0)

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