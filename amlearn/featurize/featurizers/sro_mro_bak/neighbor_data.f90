SUBROUTINE add_neighbor_features(pred_last, neigh_voro, probs, features_new, n_atoms, n_neighs, n_probs, n_features)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Important arrays:
! pred_last:    (n_atoms) array,
!               y_pred values from last layer,
!               previouslly cannot be (n_atoms) as atom ids are not continuous
!               currently should fill the lost values as 0
!
! neigh_voro:   (n_atoms, n_neighs) array,
!               neighbor_list by Voronoi,
!               should deal with the lost atoms due to removing boundaries
!               currently should fill the lost values as 0
!
! probs:        (n_probs) array/list,
!               probs used as prob_thresholds
!
! features_new: (n_atoms, n_features) array,
!               new features returned
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

integer :: n_atoms, n_neighs, n_probs, n_features
integer :: atom, i, j, m, neighs
REAL(8) :: pred_neigh, mean_nn, sum_nn, std_nn, min_nn, max_nn
REAL(8), dimension(n_atoms):: pred_last
integer, dimension(n_atoms, n_neighs):: neigh_voro
REAL(4), dimension(n_probs):: probs
REAL(8), dimension(n_atoms, n_features):: features_new

!f2py intent(in) pred_last, neigh_voro, n_atoms, n_neighs
!f2py intent(in) probs, n_probs
!f2py intent(in) n_features
!f2py intent(in,out) features_new


do atom = 1, n_atoms
    if (neigh_voro(atom, 1) .ne. 0) then
         mean_nn = 0
         sum_nn = 0
         std_nn = 0
         min_nn = pred_last(neigh_voro(atom, 1))
         max_nn = min_nn
         i = 1
         do while ((i <= n_neighs).and.(neigh_voro(atom, i).ne.0))
             pred_neigh = pred_last(neigh_voro(atom, i))
             sum_nn = sum_nn + pred_neigh
             if (pred_neigh > max_nn) then
               max_nn = pred_neigh
             end if
             if (pred_neigh < min_nn) then
               min_nn = pred_neigh
             end if
             do m = 1, n_probs
                  if (pred_neigh >= probs(m)) then
                  features_new(atom, 5 + m) = features_new(atom, 5 + m) + 1
                  end if
             end do
             i = i + 1
         end do
         neighs = i - 1
         mean_nn = sum_nn / neighs
         features_new(atom, 1) = mean_nn
         features_new(atom, 3) = min_nn
         features_new(atom, 4) = max_nn
         features_new(atom, 5) = sum_nn
         do j = 1, neighs
             std_nn = std_nn + (pred_last(neigh_voro(atom, j)) - mean_nn)**2
         end do
         std_nn = sqrt(std_nn / neighs)
         features_new(atom, 2) = std_nn
    end if
end do
return
END SUBROUTINE
