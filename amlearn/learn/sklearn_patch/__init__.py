from ._validation import cross_validate
from ._validation import cross_val_score
from ._validation import cross_val_predict
from ._validation import permutation_test_score
from ._validation import learning_curve
from ._validation import validation_curve
from ._validation import calc_scores

__all__ = ['cross_validate', 'cross_val_score', 'cross_val_predict',
           'permutation_test_score', 'learning_curve',
           'calc_scores', 'validation_curve']
