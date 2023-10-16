

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def save_plots(ip_means, ip_stds, ip_mins, ip_maxs):

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



conditioning_dict_path = "/data/xander/Projects/cog/eden-sd-pipelines/eden/xander/text_inputs_v1.npy"
conditioning_dict = np.load(conditioning_dict_path, allow_pickle=True).item()


def sample_random_conditioning(args):
    for key in ["c", "uc", "pc", "puc"]:
        mean = conditioning_dict[f"{key}_mean"]
        std = args.conditioning_sigma * conditioning_dict[f"{key}_std"]
        min_val = conditioning_dict[f"{key}_min"]
        max_val = conditioning_dict[f"{key}_max"]

        # Sample from Gaussian
        sample = np.random.normal(mean, std)

        if args.clamp_factor:
            sample = np.clip(sample, args.clamp_factor * min_val, args.clamp_factor * max_val)

        # Convert to PyTorch tensor and move to device
        setattr(args, key, torch.from_numpy(sample).to(_device))

    return args

condition_save_dir = os.path.join(ROOT_PATH, "ip_conditions_faces")

def save_ip_img_condition(args):
    ip_img_conditioning = args.c[0,77:,:].detach().clone().cpu().numpy()
    # save the ip_img_conditioning to disk:
    os.makedirs(condition_save_dir, exist_ok=True)
    np.save(os.path.join(condition_save_dir, f"ip_cond_{int(time.time())}.npy"), ip_img_conditioning)

def sample_random_ip_conditioning(args):
    # load all the ip_conditions:
    ip_conditions = []
    for ip_condition_path in glob.glob(os.path.join(condition_save_dir, "*.npy")):
        ip_condition = np.load(ip_condition_path)
        ip_conditions.append(ip_condition)

    ip_conditions = np.array(ip_conditions)
    ip_means, ip_stds = np.mean(ip_conditions, axis=0), np.std(ip_conditions, axis=0)
    ip_mins, ip_maxs  = np.min(ip_conditions, axis=0), np.max(ip_conditions, axis=0)

    #save_plots(ip_means, ip_stds, ip_mins, ip_maxs)

    # sample a random ip_condition:
    sigma = 2.0
    ip_condition = np.random.normal(ip_means, sigma*ip_stds)

    # uniform:
    ip_condition = np.random.uniform(ip_mins, ip_maxs)

    args_c_clone = args.c.clone()
    args_c_clone[0, 77:, :] = torch.from_numpy(ip_condition).unsqueeze(0).to(_device)
    args.c = args_c_clone

    return args