import matplotlib.pyplot as plt
import os
import seaborn as sns

path = "plots"



def plot_train_val_metric(results_cnn, results_res, metric_name, ylabel, outdir):

    sns.set_theme(style="whitegrid") 

    epochs = range(1, len(results_cnn['train_' + metric_name]) + 1)

    plt.figure(figsize=(11, 6))

    # Colors 
    color_cnn = sns.color_palette("deep")[0]  
    color_res = sns.color_palette("deep")[3]  

    # CNN
    sns.lineplot(x=epochs, y=results_cnn['train_' + metric_name],
                 label=f'CNN — Train', color=color_cnn, marker='o')
    sns.lineplot(x=epochs, y=results_cnn['val_' + metric_name],
                 label=f'CNN — Validation', color=color_cnn, marker='o', linestyle='--')

    # ResNet
    sns.lineplot(x=epochs, y=results_res['train_' + metric_name],
                 label=f'ResNet — Train', color=color_res, marker='s')
    sns.lineplot(x=epochs, y=results_res['val_' + metric_name],
                 label=f'ResNet — Validation', color=color_res, marker='s', linestyle='--')

    # Labels − Title
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(f"Performance Comparison — {metric_name.upper()}",
              fontsize=18, weight='bold')
    plt.xticks(ticks=list(epochs), labels=[str(e) for e in epochs])
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True)
    plt.tight_layout()

    outfile = os.path.join(outdir, f"train_val_{metric_name}.png")
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()



def plot_all(results_cnn, results_res, outdir="plots"):

    # Create output directory if needed
    os.makedirs(outdir, exist_ok=True)

    plot_train_val_metric(results_cnn, results_res, metric_name="loss", ylabel="Loss", outdir=outdir)
    plot_train_val_metric(results_cnn, results_res, metric_name="acc", ylabel="Accuracy", outdir=outdir)

    print("Graphics downloaded")



import pickle

results_cnn = pickle.load(open("results/results_cnn.pkl", "rb"))
results_res = pickle.load(open("results/results_res.pkl", "rb"))

plot_all(results_cnn, results_res, outdir=path)