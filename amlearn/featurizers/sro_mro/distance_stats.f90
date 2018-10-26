    subroutine CN_dist(CN, n_neighbor_list)
        integer, allocatable :: CN, n_neighbor_list

!f2py   intent(out) :: CN
!f2py   intent(in) :: n_neighbor_list

        CN = n_neighbor_list

    end subroutine CN_dist



    subroutine cutoff_distance_stats(distance_stats, &
        n_atoms, n_neighbor_limit, n_neighbor_list, neighbor_distance_lists)

        use :: a_stats

        integer :: n_atoms, n_neighbor_limit
        integer, dimension(n_atoms) :: n_neighbor_list
        REAL(8), dimension(n_atoms, n_neighbor_limit) :: neighbor_distance_lists
        REAL(8), dimension(n_atoms, 5) :: distance_stats

!f2py   intent(in) :: n_atoms, n_neighbor_limit
!f2py   intent(in) :: n_neighbor_list
!f2py   intent(in) :: neighbor_distance_lists
!f2py   intent(in) :: edge_min, edge_max
!f2py   intent(out) :: distance_list

        integer  :: atom

        do atom = 1, n_atoms
            distance_stats(atom, :) = all_stats(neighbor_distance_lists(atom, :), n_neighbor_list(atom))
        end do

    end subroutine cutoff_distance_stats



