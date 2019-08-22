import os
import copy
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams


def violin_features_pairs(prepared_df, class_col, save_to_dir=None,
                          hue_order=None, order=None,
                          dpi=500, bw=0.1, palette="Set2", orient=None,
                          figsize=None, dist_lim=None, feature_lim=None,
                          xlim=None, ylim=None,
                          fontsize=13, label_fontsize=16, split=True,
                          scale="count", inner="quartile",
                          linewidth=0.9, pad=5, aspect=1/2,
                          xrotation=0, yrotation=0, x_ha="right",
                          major_length=6, violinwidth=0.8):
    # start draw heatmap
    print("start draw violin figures")
    plt.rcParams.update({'font.family': "arial"})
    plt.figure(figsize=figsize if figsize else (10, 10))
    ax = plt.axes()

    if orient is None or orient == "v":
        sns_map = sns.violinplot(
            x="variable", y="value", hue=class_col, hue_order=hue_order,
            data=prepared_df, order=order,
            palette=palette, split=split, scale=scale, inner=inner,
            bw=bw, orient=orient, linewidth=linewidth, width=violinwidth)
        ylim = dist_lim
        xlim = feature_lim
    elif orient == "h":
        sns_map = sns.violinplot(
            y="variable", x="value", hue=class_col, hue_order=hue_order,
            data=prepared_df, order=order,
            palette=palette, split=split, scale=scale, inner=inner,
            bw=bw, orient=orient, linewidth=linewidth, width=violinwidth)
        xlim = dist_lim
        ylim = feature_lim

    set_plot_args(ax,
                  # xticklabels=xticklabels,
                  # yticks=yticks,
                  xlim=xlim, ylim=ylim,
                  top="off", bottom="on",
                  xlabel="features",
                  ylabel="normalized distribution",
                  label_fontsize=label_fontsize,
                  # yticklabels=yticklabels,
                  fontsize=fontsize, linewidth=linewidth,
                  left_linewidth=linewidth,
                  top_linewidth=linewidth,
                  bottom_linewidth=linewidth,
                  right_linewidth=linewidth,
                  major_width=linewidth,
                  major_length=6,
                  pad=pad, aspect=aspect,
                  xrotation=xrotation,
                  yrotation=yrotation,
                  x_ha=x_ha,
                  direction="out")

    if save_to_dir:
        figures = sns_map.get_figure()
        figures.savefig(save_to_dir,
                        dpi=dpi,
                        bbox_inches='tight')
        plt.close()


def prepare_df_for_violinplot(df, feature_cols, class_col,
                              class_indices=None, minmaxscale=True):
    """
    Min-max-scale the data and then melt the dataframe into the long format
    """
    if class_indices:
        df = df.loc[list(class_indices)]
    df = df[feature_cols + [class_col]]

    if minmaxscale:
        from sklearn.preprocessing.data import MinMaxScaler
        scaler = MinMaxScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])

    prepared_df = pd.melt(df, value_vars=feature_cols, id_vars=class_col)

    return prepared_df


def set_plot_args(ax, **plot_args):
    print(plot_args)
    fontsize = plot_args.get('fontsize', 12)
    fontfamily = plot_args.get('fontfamily', 'arial')
    plt.rcParams.update({'font.size': fontsize,
                         'legend.fontsize': fontsize,
                         'xtick.labelsize': fontsize,
                         'ytick.labelsize': fontsize,
                         'axes.labelsize': plot_args.get('label_fontsize',
                                                         fontsize + 2),
                         'axes.titlesize': plot_args.get('label_fontsize',
                                                         fontsize + 2),
                         'font.family': fontfamily})

    ax.set_xticks(plot_args.get('xticks', ax.get_xticks()))
    ax.set_yticks(plot_args.get('yticks', ax.get_yticks()))
    ax.set_xlim(plot_args.get('xlim', ax.get_xlim()))
    ax.set_ylim(plot_args.get('ylim', ax.get_ylim()))

    if 'xticklabels' in plot_args.keys():
        ax.set_xticklabels(plot_args['xticklabels'])
    if 'yticklabels' in plot_args.keys():
        ax.set_yticklabels(plot_args['yticklabels'])
    plt.xticks(rotation=plot_args.get('xrotation', 0),
               ha=plot_args.get('x_ha', "center"))
    plt.yticks(rotation=plot_args.get('yrotation', 0),
               ha=plot_args.get('y_ha', "right"))

    if 'xlabel' in plot_args.keys():
        ax.set_xlabel(plot_args.get('xlabel'),
                      fontsize=plot_args.get('label_fontsize', fontsize + 2))
    # ax.xaxis.label.set_size(plot_args.get('label_fontsize', fontsize + 2))

    if 'ylabel' in plot_args.keys():
        ax.set_ylabel(plot_args.get('ylabel'),
                      fontsize=plot_args.get('label_fontsize', fontsize + 2))
    # ax.yaxis.label.set_size(plot_args.get('label_fontsize', fontsize + 2))

    if 'title' in plot_args.keys():
        ax.set_title(plot_args.get('title'),
                     fontsize=plot_args.get('title_fontsize', fontsize + 2))

    ax.tick_params(direction=plot_args.get('direction', "in"),
                   bottom=plot_args.get('bottom', "on"),
                   left=plot_args.get('left', "on"),
                   top=plot_args.get('top', "off"),
                   right=plot_args.get('right', "off"),
                   labelbottom=plot_args.get('labelbottom', "on"),
                   labelleft=plot_args.get('labelleft', "on"),
                   labelsize=fontsize+2)
    cax = plt.gcf().axes[-1]
    cax.tick_params(labelsize=fontsize+2)

    l_wid = rcParams['axes.linewidth']
    print("l_wid:{}".format(l_wid))
    ax.spines['left'].set_visible(plot_args.get('left_visible', True))
    ax.spines['left'].set_linewidth(plot_args.get('left_linewidth', l_wid))

    ax.spines['bottom'].set_visible(plot_args.get('bottom_visible', True))
    ax.spines['bottom'].set_linewidth(plot_args.get('bottom_linewidth', l_wid))

    ax.spines['right'].set_visible(plot_args.get('right_visible', True))
    ax.spines['right'].set_linewidth(plot_args.get('right_linewidth', l_wid))

    ax.spines['top'].set_visible(plot_args.get('top_visible', True))
    ax.spines['top'].set_linewidth(plot_args.get('top_linewidth', l_wid))

    ax.tick_params(which='major', width=plot_args.get('major_width', 0.8))
    ax.tick_params(which='major', length=plot_args.get('major_length', 4))
    ax.tick_params(which='minor', width=plot_args.get('minor_width', 0.7))
    ax.tick_params(which='minor', length=plot_args.get('minor_length', 2.5))

    if plot_args.get('aspect', 1) != 'False':
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        ax.set_aspect(abs(x1 - x0) / abs(y1 - y0) *
                      plot_args.get('aspect', 1))

    pad = plot_args.get('pad', None)
    if pad:
        for tick in ax.get_xaxis().get_major_ticks():
            tick.set_pad(pad)
        for tick in ax.get_yaxis().get_major_ticks():
            tick.set_pad(pad)

    return ax


if __name__ == "__main__":
    feature_file = "xx"
    output_path = "xxx"
    df = pd.read_csv(feature_file, index_col="number")

    selected_features = \
        ['MRO_mean area_interstice std voro',
         'volume_interstice std voro',
         'distance_interstice min voro',
         'distance_interstice max voro',
         'MRO_mean distance_interstice mean']

    feature_names_latex = \
        {'MRO_mean area_interstice std voro': 'MRO$_{\\mathrm{mean}}$ std(a$_{\\mathrm{interstice}}$)',
         'volume_interstice std voro': 'std(V$_{\\mathrm{interstice}}$)',
         'distance_interstice min voro': 'min(d$_{\\mathrm{interstice}}$)',
         'distance_interstice max voro': 'max(d$_{\\mathrm{interstice}}$)',
         'MRO_mean distance_interstice mean': 'MRO$_{\\mathrm{mean}}$ mean(d$_{\\mathrm{interstice}}$)', }

    qs_col = "QS_predict"
    qs_lower_threshold = 0.10
    qs_higher_threshold = 0.70

    df_slice = copy.copy(df[(df[qs_col] < qs_lower_threshold) | (
                df[qs_col] > qs_higher_threshold)])

    df_slice["QS_hard_soft"] = \
        df_slice[qs_col].apply(
            lambda x: "soft" if x >= qs_higher_threshold else "hard")
    class_col = "QS_hard_soft"

    df_slice = df_slice.rename(columns=feature_names_latex)

    # prepare the dataframe for violinplots
    prepared_df = prepare_df_for_violinplot(
        df_slice, feature_cols=list(feature_names_latex.values()),
        class_col=class_col,
        class_indices=None, minmaxscale=True)

    palette = "gist_earth_r"
    soft_palette = 0
    hard_palette = 8
    bw = 0.08
    fontsize = 24
    linewidth = 2.0
    xrotation = 20
    yrotation = 0
    x_ha = "right"
    orient = "v"
    aspect = 1
    figsize = (10, 10)
    violinwidth = 1.25
    dist_lim = (-0.1, 1.1)
    margin = 0.2
    feature_lim = (-0.5 - margin, len(selected_features) - 0.5 + margin)
    # scale = "area"
    scale = "count"

    p = sns.color_palette(palette, n_colors=10)
    violin_features_pairs(
        prepared_df, class_col,
        save_to_dir=os.path.join(
            output_path,
            "{}_{}_features_font_{}_bw_{}_linewidth_{}_xrotation_{}_yrotation_{}_orient_{}_aspect_{}_figsize_{}_{}_violinwidth_{}_xlim_{}_scale_{}_margin_{}_lower{}_higher_{}_{}.png"
                .format(palette, len(selected_features),
                        fontsize, bw, linewidth,
                        xrotation, yrotation, orient,
                        aspect, figsize[0], figsize[1],
                        violinwidth, dist_lim[1],
                        scale, margin,
                        qs_lower_threshold,
                        qs_higher_threshold,
                        datetime.datetime.now().strftime(
                             '%Y-%m-%d_%H-%M-%S'))),
        hue_order=["hard", "soft"], order=None,
        dpi=300, bw=bw, xrotation=xrotation, yrotation=yrotation,
        fontsize=fontsize, linewidth=linewidth, orient=orient,
        aspect=aspect, figsize=figsize, violinwidth=violinwidth,
        dist_lim=dist_lim, feature_lim=feature_lim,
        # inner=inner,
        scale=scale,
        x_ha=x_ha,
        palette={"soft": p[soft_palette], "hard": p[hard_palette]})
