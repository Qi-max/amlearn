    subroutine CN_dist(CN, n_neighbor_list)
        integer, allocatable :: CN, n_neighbor_list

!f2py   intent(out) :: CN
!f2py   intent(in) :: n_neighbor_list

        CN = n_neighbor_list

    end subroutine CN_dist



    subroutine cutoff_distance_stats(distance_stats, &
                                     n_neighbor_list, neighbor_distance_lists &
                                     n_atoms, n_neighbor_limit)

        use :: a_stats, only : all_stats

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



    subroutine cutoff_csro(csro, &
                           atom_type, element_comp_list, &
                           n_neighbor_list, neighbor_lists, &
                           n_atoms, n_neighbor_limit, n_elements)
        integer :: n_atoms
        integer, dimension(n_atoms) :: atom_type, n_neighbor_list
        REAL(8), dimension(n_atoms, n_elements) :: element_comp_list
        REAL(8), dimension(n_atoms, n_neighbor_limit) :: neighbor_lists
        REAL(8), dimension(n_atoms, 1 + 3*n_elements) :: csro

!f2py   intent(in) :: n_atoms, n_neighbor_limit, n_elements
!f2py   intent(in) :: atom_type
!f2py   intent(in) :: element_comp_list
!f2py   intent(in) :: n_neighbor_list, neighbor_lists
!f2py   intent(in, out) :: csro

        integer :: i
        integer, dimension(n_elements) :: element_count

        csro(:, 1) = atom_type(:)

        do atom = 1, n_atoms
            element_count = 0
            do i = 1, n_neighbor_list(atom)
                element_count(atom_type(neighbor_lists(i))) = element_count(atom_type(neighbor_lists(i))) + 1
            end do
            do j = 1, n_elements
                csro(atom, j + 1) = element_count(j)
            end do
        end do

        do j = 1, n_elements
            csro(:, j + 1 + n_elements) = csro(:, j + 1) / n_neighbor_list(:)
        end do

        do j = 1, n_elements
            csro(:, j + 1 + n_elements*2) = (csro(:, j + 1 + n_elements) - element_comp_list(j)) / element_comp_list(j)
        end do

    end subroutine cutoff_csro