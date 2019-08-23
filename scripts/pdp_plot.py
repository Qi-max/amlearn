#-*- coding: UTF-8 -*-
from __future__ import division
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble.partial_dependence import partial_dependence, \
    plot_partial_dependence


class PDPPlot:
    def __init__(self, model, feature_df, feature_list,
                 output_path, fig_file):
        self.model = model
        self.feature_df = feature_df
        self.feature_list = feature_list
        self.output_path = output_path
        self.fig_file = fig_file

    def plot_2d(self, feature_2d, top=0.9, n_jobs=3, grid_resolution=50,
                figsize=(8, 9), subtitle=""):
        fig, axs = plot_partial_dependence(gbrt=self.model, X=self.feature_df,
                                           features=feature_2d,
                                           feature_names=self.feature_list,
                                           n_jobs=n_jobs,
                                           grid_resolution=grid_resolution,
                                           figsize=figsize)
        fig.suptitle(subtitle)
        plt.subplots_adjust(top=top, left=0.16, bottom=0.07, right=0.81,
                            wspace=0.98, hspace=0.63)
        plt.savefig(os.path.join(self.output_path, self.fig_file))
        plt.close()

    def plot_3d(self, feature_3d, top=0.9, grid_resolution=50,
                rstride=1, cstride=1, cmap="jet", edgecolor='K',
                elev=22, azim=122):
        fig = plt.figure()
        pdp, axes = partial_dependence(gbrt=self.model,
                                       target_variables=feature_3d,
                                       X=self.feature_df,
                                       grid_resolution=grid_resolution)
        XX, YY = np.meshgrid(axes[0], axes[1])
        Z = pdp[0].reshape(list(map(np.size, axes))).T
        ax = Axes3D(fig)
        surf = ax.plot_surface(XX, YY, Z, rstride=rstride, cstride=cstride,
                               cmap=cmap, edgecolor=edgecolor)
        ax.set_xlabel(self.feature_list[feature_3d[0]])
        ax.set_ylabel(self.feature_list[feature_3d[1]])
        ax.set_zlabel('Partial dependence')

        ax.view_init(elev=elev, azim=azim)
        plt.colorbar(surf)
        plt.suptitle('Partial dependence of house value on median\n'
                     'age and average occupancy')
        plt.subplots_adjust(top=top)

        plt.savefig(os.path.join(self.output_path, self.fig_file))
        plt.close()
        # plt.show()

    def plot_topn_features(self, top_n=10, top=0.9, n_jobs=3,
                           grid_resolution=50):
        if hasattr(self.model, "feature_importances_"):
            importances = sorted(zip(self.model.feature_importances_,
                                     self.feature_list), reverse=True)
            top_n = min(top_n, len(self.feature_list))
            feature_top_n = str([importances[rank][1]
                                 for rank in range(top_n)])
            self.plot_2d(feature_2d=feature_top_n, top=top, n_jobs=n_jobs,
                         grid_resolution=grid_resolution)

        else:
            raise AttributeError("This model {} has no feature_importances_"
                                 "attribute, please directly use plot_2d!".
                                 format(self.model))
