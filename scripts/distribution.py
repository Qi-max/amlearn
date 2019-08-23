import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def column_hist(data, bins=None, density=True, fraction=True,
                save_figure_to_dir=None, save_data_to_dir=None, fmt=None,
                ylim=None, yscale=None):
    bins = 20 if bins is None else bins
    fmt = '%.10e' if fmt is None else fmt

    hist, edge = np.histogram(data, bins=bins, density=density)

    if density is False:
        if fraction:
            hist = hist/hist.sum()

    if save_figure_to_dir:
        plt.figure(figsize=(6, 6))
        # alpha gives transparency
        plt.plot(edge[1:], hist, 'r--o', alpha=0.5, linewidth=1.0)
        if ylim:
            plt.ylim(*ylim)
        if yscale:
            plt.yscale(yscale)
        plt.savefig(save_figure_to_dir, dpi=100, bbox_inches='tight')
        plt.close()

    if save_data_to_dir:
        np.savetxt(save_data_to_dir, list(zip(edge[1:], hist)), fmt=fmt)


if __name__ == "__main__":
    system = ["Cu65Zr35", "qr_5plus10^10"]
    prediction_file = "xx"
    output_path = "xxx"
    output_file_header = r'{}_{}_QS'.format(*system)

    qs_col = "QS_predict"
    bin = 0.02

    df = pd.read_csv(prediction_file, index_col=0)

    column_hist(df[qs_col], bins=np.arange(0, 1.0, bin), density=True,
                save_figure_to_dir=os.path.join(output_path, "{}_density_bin_{}.png".format(output_file_header, bin)),
                save_data_to_dir=os.path.join(output_path, "{}_density_bin_{}.csv".format(output_file_header, bin)),
                fmt=["%.2f", '%.10e'])

    column_hist(df[qs_col], bins=np.arange(0, 1.0, bin), density=False,
                save_figure_to_dir=os.path.join(output_path, "{}_fraction_bin_{}.png".format(output_file_header, bin)),
                save_data_to_dir=os.path.join(output_path, "{}_fraction_bin_{}.csv".format(output_file_header, bin)),
                ylim=(0, 0.05),
                fmt=["%.2f", '%.10e'])

    column_hist(df[qs_col], bins=np.arange(0, 1.0, bin), density=False,
                save_figure_to_dir=os.path.join(output_path, "{}_fraction_bin_{}_log.png".format(output_file_header, bin)),
                save_data_to_dir=os.path.join(output_path, "{}_fraction_bin_{}_log.csv".format(output_file_header, bin)),
                ylim=(0.0001, 0.05), yscale="log",
                fmt=["%.2f", '%.10e'])
