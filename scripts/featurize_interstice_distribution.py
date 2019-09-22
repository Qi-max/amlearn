from amlearn.utils.data import read_lammps_dump
from amlearn.featurize.featurizers.medium_range_order import MRO
from amlearn.featurize.featurizers.nearest_neighbor import VoroNN, DistanceNN
from amlearn.featurize.featurizers.featurizer_pipeline import FeaturizerPipeline
from amlearn.featurize.featurizers.short_range_order import \
    VolumeAreaInterstice, ClusterPackingEfficiency, \
    AtomicPackingEfficiency, DistanceInterstice

"""
This is an example script of deriving features for each atom, based on the 
Fortran source codes in amlearn/featurize/featurizers/src/ and classes in 
short/medium_range_order. Please make sure to compile the Fortran code using
f2py before running this script. 
"""

system = ["Cu65Zr35", "qr_5plus10^10"]
CuZr_atomic_number_list = [29, 40]

lammps_file = "xxx/dump.lmp"
output_path = "xxx/xxx"
structure, bds = read_lammps_dump(lammps_file)

featurizers = [
    VoroNN(Bds=bds, cutoff=5, output_path=output_path),
    DistanceNN(Bds=bds, cutoff=4, output_path=output_path),
    DistanceInterstice(type_to_atomic_number_list=CuZr_atomic_number_list,
                       dependent_class='voro', output_path=output_path),
    DistanceInterstice(type_to_atomic_number_list=CuZr_atomic_number_list,
                       dependent_class='dist', output_path=output_path),
    VolumeAreaInterstice(type_to_atomic_number_list=CuZr_atomic_number_list,
                         output_path=output_path),
    ClusterPackingEfficiency(type_to_atomic_number_list=CuZr_atomic_number_list,
                             output_path=output_path),
    AtomicPackingEfficiency(type_to_atomic_number_list=CuZr_atomic_number_list,
                            output_path=output_path),
    MRO(output_path=output_path)]
multi_featurizer = FeaturizerPipeline(featurizers=featurizers,
                                      output_path=output_path)
multi_featurizer.fit_transform(X=structure, Bds=bds, lammps_df=structure)
