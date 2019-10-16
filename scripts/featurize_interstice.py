from amlearn.utils.data import read_lammps_dump
from amlearn.featurize.pipeline import FeaturizePipeline
from amlearn.featurize.nearest_neighbor import VoroNN, DistanceNN
from amlearn.featurize.short_range_order import \
    DistanceInterstice, VolumeAreaInterstice
from amlearn.featurize.medium_range_order import MRO

__author__ = "Qi Wang"
__email__ = "qiwang.mse@gmail.com"

"""
This is an example script of deriving interstice distribution features for 
each atom, based on relevant distance/area/volume interstice classes in 
amlearn.featurize.short_range_order,
as well as classes in amlearn.featurize.medium_range_order to further 
coarse-grain SRO features to MRO. 
"""

system = ["Cu65Zr35", "qr_5plus10^10"]

atomic_number_list = [29, 40]
stat_ops = ['mean', 'std', 'min', 'max']

lammps_file = "xxx/dump.lmp"
structure, bds = read_lammps_dump(lammps_file)

output_path = "xxx/xxx"

featurizers = [
    # neighboring analysis
    VoroNN(bds=bds, cutoff=5, output_path=output_path),
    DistanceNN(bds=bds, cutoff=4, output_path=output_path),

    # distance interstice
    DistanceInterstice(atomic_number_list=atomic_number_list,
                       dependent_class='voro', stat_ops=stat_ops,
                       output_path=output_path),
    DistanceInterstice(atomic_number_list=atomic_number_list,
                       dependent_class='dist', stat_ops=stat_ops,
                       output_path=output_path),

    # area and volume interstice
    VolumeAreaInterstice(atomic_number_list=atomic_number_list,
                         stat_ops=stat_ops, output_path=output_path),

    # from SRO to MRO
    MRO(stats_types=[0, 1, 1, 1, 1, 0], output_path=output_path)]

# defining a featurize_pipeline
featurize_pipeline = FeaturizePipeline(featurizers=featurizers,
                                       output_path=output_path)

# featurization
feature_df = featurize_pipeline.fit_transform(X=structure, bds=bds,
                                              lammps_df=structure)
