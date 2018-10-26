    subroutine sro_to_mro(sro_feature, n_atoms, n_neighbor_list, &
        n_neighbor_limit, neighbor_lists, stats_types, sum_stats_types, mro_feature)

        use :: c_stats

        integer :: n_atoms, sum_stats_types
        REAL(8), dimension(n_atoms):: sro_feature
        integer, dimension(n_atoms):: n_neighbor_list
        integer, dimension(n_atoms, n_neighbor_limit):: neighbor_lists
        integer, dimension(6) :: stats_types
        REAL(8), dimension(n_atoms, sum_stats_types):: mro_feature

!f2py   intent(in) sro_feature, n_atoms, n_neighbor_list, n_neighbor_limit, neighbor_lists
!f2py   intent(in) stats_types, sum_stats_types
!f2py   intent(in, out) mro_feature
        integer :: atom, i, j
        REAL(8), allocatable :: list(:)

        mro_feature = 0
        list = 0
        do atom = 1, n_atoms
          allocate(list(n_neighbor_list(atom)))
          do i = 1, n_neighbor_list(atom)
            j = neighbor_lists(atom, i)
            list(i) = sro_feature(j)
          end do
          mro_feature(atom, :) = customize_stats(list, stats_types, sum_stats_types, sro_feature(atom))
          deallocate(list)
        end do

    end subroutine sro_to_mro
