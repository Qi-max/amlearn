

    subroutine cn_voro(cn, n_atoms, n_neighbor_list)
        integer, dimension(n_atoms) :: cn, n_neighbor_list

!f2py   intent(in, out) :: cn
!f2py   intent(in) :: n_atoms, n_neighbor_list

        cn = n_neighbor_list

    end subroutine cn_voro



    SUBROUTINE voronoi_index(voronoi_index_list, n_atoms, n_neighbor_limit, &
                             n_neighbor_list, neighbor_edge_lists, &
                             edge_min, edge_max, include_beyond_edge_max)

        integer :: n_atoms, n_neighbor_limit
        integer, dimension(n_atoms) :: n_neighbor_list
        integer, dimension(n_atoms, n_neighbor_limit) :: neighbor_edge_lists
        integer :: include_beyond_edge_max
        integer :: edge_min, edge_max
        integer, dimension(n_atoms, edge_max - edge_min + 1) :: voronoi_index_list

!f2py   intent(in) :: n_atoms, n_neighbor_limit
!f2py   intent(in) :: n_neighbor_list
!f2py   intent(in) :: neighbor_edge_lists
!f2py   intent(in) :: edge_min, edge_max, include_beyond_edge_max
!f2py   intent(in, out) :: voronoi_index_list

        integer  :: atom, i, edge
        voronoi_index_list=0
        do atom = 1, n_atoms
          do i = 1, n_neighbor_list(atom)
            edge = neighbor_edge_lists(atom, i)
            if ((edge >= edge_min) .AND. (edge <= edge_max)) then
              voronoi_index_list(atom, edge - edge_min + 1) = voronoi_index_list(atom, edge - edge_min + 1) + 1
            else if ((edge > edge_max) .AND. (include_beyond_edge_max == 1)) then
              voronoi_index_list(atom, edge_max - edge_min + 1) = voronoi_index_list(atom, edge_max - edge_min + 1) + 1
            end if
          end do
        end do
    return
    end SUBROUTINE voronoi_index


    subroutine i_fold_symmetry(i_symm_list, n_atoms, n_neighbor_limit, n_neighbor_list, &
        neighbor_edge_lists, edge_min, edge_max, include_beyond_edge_max)

        integer :: n_atoms, n_neighbor_limit
        integer, dimension(n_atoms) :: n_neighbor_list
        integer, dimension(n_atoms, n_neighbor_limit) :: neighbor_edge_lists
        integer :: include_beyond_edge_max
        integer :: edge_min, edge_max
        REAL(8), dimension(n_atoms, edge_max - edge_min + 1) :: i_symm_list

!f2py   intent(in) :: n_atoms, n_neighbor_limit
!f2py   intent(in) :: n_neighbor_list
!f2py   intent(in) :: neighbor_edge_lists
!f2py   intent(in) :: edge_min, edge_max, include_beyond_edge_max
!f2py   intent(in, out) :: i_symm_list

        integer :: atom
        integer, dimension(n_atoms, edge_max - edge_min + 1) :: voronoi_index_list

        call voronoi_index(voronoi_index_list, n_atoms, n_neighbor_limit, n_neighbor_list, &
            neighbor_edge_lists, edge_min, edge_max, include_beyond_edge_max)

        do atom = 1, n_atoms
            i_symm_list(atom, :) = 1.0 * voronoi_index_list(atom, :) / n_neighbor_list(atom)
        end do

    end subroutine i_fold_symmetry



    subroutine line_percent(percent_list, value_list, n_atoms, n_values)
        integer :: n_atoms, n_values
        integer, dimension(n_atoms, n_values) :: value_list
        REAL(8), dimension(n_atoms, n_values) :: percent_list

!f2py   intent(in) :: n_atoms, n_values
!f2py   intent(in) :: value_list
!f2py   intent(in, out) :: percent_list

        integer :: atom

        do atom = 1, n_atoms
            percent_list(atom, :) = 1.0 * value_list(atom, :) / SUM(value_list(atom, :))
        end do
    end subroutine line_percent


    subroutine character_motif(motif_one_hot, voronoi_index_list, n_atoms, n_voro, &
                               edge_min, target_voro_idx, n_target, frank_kasper)
        integer :: n_atoms, edge_min, frank_kasper
        integer, dimension(n_target, n_voro) :: target_voro_idx
        integer, dimension(n_atoms, n_voro) :: voronoi_index_list
        integer, dimension(n_atoms, n_target + frank_kasper) :: motif_one_hot

!f2py   intent(in) :: voronoi_index_list
!f2py   intent(in) :: edge_min
!f2py   intent(in) :: target_voro_idx
!f2py   intent(in) :: frank_kasper
!f2py   intent(in, out) :: motif_one_hot
        write(*, *) target_voro_idx
        write(*, *) n_target + frank_kasper
!        motif_one_hot = 0
        do atom = 1, n_atoms
            do target_idx = 1, n_target
                if(all(voronoi_index_list(atom, :)-target_voro_idx(target_idx, :)==0)) then
                    motif_one_hot(atom, target_idx) = 1
                end if
            end do
            if((frank_kasper==1).AND.((2 * voronoi_index_list(atom, 5-edge_min) + voronoi_index_list(atom, 6-edge_min)).EQ.12)) then
                motif_one_hot(atom, n_target + frank_kasper) = 1
            end if
        end do
    end subroutine character_motif
!
!    subroutine special_polyhedron(special_polyhedron_list, i_symm_list, n_atoms)
!
!        integer :: n_atoms
!        REAL(8), dimension(n_atoms, 4) :: i_symm_list
!        REAL(8), dimension(n_atoms, 4) :: special_polyhedron_list
!
!!f2py   intent(in) :: n_atoms
!!f2py   intent(in) :: i_symm_list
!!f2py   intent(in, out) :: special_polyhedron_list
!
!        integer :: atom, is_00120, is_00124, is_00120_00124, is_stab_12
!
!        do atom = 1, n_atoms
!            if(i_symm_list(atom, 3) == 12) then
!                is_stab_12 = 1
!                if((i_symm_list(atom, 1) == 0).AND.((i_symm_list(atom, 2) == 0))) then
!
!                end if
!            else
!                is_00120 = 0
!                is_00124 = 0
!                is_00120_00124 = 0
!                is_stab_12 = 0
!            end if
!            special_polyhedron_list(atom, :) = (/is_00120, is_00124, is_00120_00124, is_stab_12/)
!        end do
!
!    end subroutine special_polyhedron


    subroutine area_wt_i_fold_symmetry(area_wt_i_symm_list, n_atoms, n_neighbor_limit, n_neighbor_list, &
        neighbor_edge_lists, neighbor_area_lists, edge_min, edge_max, include_beyond_edge_max)

        integer :: n_atoms, n_neighbor_limit
        integer, dimension(n_atoms) :: n_neighbor_list
        integer, dimension(n_atoms, n_neighbor_limit) :: neighbor_edge_lists
        REAL(8), dimension(n_atoms, n_neighbor_limit) :: neighbor_area_lists
        integer :: include_beyond_edge_max
        integer :: edge_min, edge_max
        REAL(8), dimension(n_atoms, edge_max - edge_min + 1) :: area_wt_i_symm_list

!f2py   intent(in) :: n_atoms, n_neighbor_limit
!f2py   intent(in) :: n_neighbor_list
!f2py   intent(in) :: neighbor_edge_lists, neighbor_area_lists
!f2py   intent(in) :: edge_min, edge_max, include_beyond_edge_max
!f2py   intent(in, out) :: area_wt_i_symm_list

        integer  :: atom, i, edge
        REAL(8) :: area_sum

        do atom = 1, n_atoms
          area_sum = 0
          do i = 1, n_neighbor_list(atom)
            edge = neighbor_edge_lists(atom, i)
            area_sum = area_sum + neighbor_area_lists(atom, i)
            if ((edge >= edge_min) .AND. (edge <= edge_max)) then
              area_wt_i_symm_list(atom, edge - edge_min + 1) = &
                  area_wt_i_symm_list(atom, edge - edge_min + 1) + neighbor_area_lists(atom, i)
            else if ((edge > edge_max) .AND. (include_beyond_edge_max == 1)) then
              area_wt_i_symm_list(atom, edge_max - edge_min + 1) = &
                  area_wt_i_symm_list(atom, edge_max - edge_min + 1) + neighbor_area_lists(atom, i)
            end if
          end do
          area_wt_i_symm_list(atom, :) = area_wt_i_symm_list(atom, :) / area_sum
        end do

    end subroutine area_wt_i_fold_symmetry


    subroutine vol_wt_i_fold_symmetry(vol_wt_i_symm_list, n_atoms, n_neighbor_limit, n_neighbor_list, &
        neighbor_edge_lists, neighbor_vol_lists, edge_min, edge_max, include_beyond_edge_max)

        integer :: n_atoms, n_neighbor_limit
        integer, dimension(n_atoms) :: n_neighbor_list
        integer, dimension(n_atoms, n_neighbor_limit) :: neighbor_edge_lists
        REAL(8), dimension(n_atoms, n_neighbor_limit) :: neighbor_vol_lists
        integer :: include_beyond_edge_max
        integer :: edge_min, edge_max
        REAL(8), dimension(n_atoms, edge_max - edge_min + 1) :: vol_wt_i_symm_list

!f2py   intent(in) :: n_atoms, n_neighbor_limit
!f2py   intent(in) :: n_neighbor_list
!f2py   intent(in) :: neighbor_edge_lists, neighbor_vol_lists
!f2py   intent(in) :: edge_min, edge_max, include_beyond_edge_max
!f2py   intent(in, out) :: vol_wt_i_symm_list

        call area_wt_i_fold_symmetry(vol_wt_i_symm_list, n_atoms, n_neighbor_limit, n_neighbor_list, &
        neighbor_edge_lists, neighbor_vol_lists, edge_min, edge_max, include_beyond_edge_max)

    end subroutine vol_wt_i_fold_symmetry



    subroutine voronoi_area_stats(area_stats, &
        n_atoms, n_neighbor_limit, n_neighbor_list, neighbor_area_lists)

        use :: a_stats, only: all_stats

        integer :: n_atoms, n_neighbor_limit
        integer, dimension(n_atoms) :: n_neighbor_list
        REAL(8), dimension(n_atoms, n_neighbor_limit) :: neighbor_area_lists
        REAL(8), dimension(n_atoms, 5) :: area_stats

!f2py   intent(in) :: n_atoms, n_neighbor_limit
!f2py   intent(in) :: n_neighbor_list
!f2py   intent(in) :: neighbor_area_lists
!f2py   intent(in, out) :: area_stats

        integer  :: atom

        do atom = 1, n_atoms
            area_stats(atom, :) = all_stats(neighbor_area_lists(atom, :), n_neighbor_list(atom))
        end do

    end subroutine voronoi_area_stats


    subroutine voronoi_area_stats_separate(area_stats_separate, &
        n_atoms, n_neighbor_limit, n_neighbor_list, neighbor_edge_lists, &
        neighbor_area_lists, edge_min, edge_max, include_beyond_edge_max)

        use :: a_stats, only: all_stats

        integer :: n_atoms, n_neighbor_limit
        integer, dimension(n_atoms) :: n_neighbor_list
        integer, dimension(n_atoms, n_neighbor_limit) :: neighbor_edge_lists
        REAL(8), dimension(n_atoms, n_neighbor_limit) :: neighbor_area_lists
        integer :: edge_min, edge_max, include_beyond_edge_max
        REAL(8), allocatable:: atom_area_stats_separate(:,:)
        integer, dimension(edge_max - edge_min + 1) :: atom_count_list_separate
        REAL(8), dimension(n_atoms, 5 * (edge_max - edge_min + 1)) :: area_stats_separate

!f2py   intent(in) :: n_atoms, n_neighbor_limit
!f2py   intent(in) :: n_neighbor_list
!f2py   intent(in) :: neighbor_edge_lists, neighbor_area_lists
!f2py   intent(in) :: edge_min, edge_max, include_beyond_edge_max
!f2py   intent(in, out) :: area_stats_separate

        integer  :: atom, i, j, edge
        REAL(8) :: area

        do atom = 1, n_atoms
          atom_count_list_separate = 0
          allocate(atom_area_stats_separate(edge_max - edge_min + 1, n_neighbor_list(atom)))
          atom_area_stats_separate = 0
          do i = 1, n_neighbor_list(atom)
            edge = neighbor_edge_lists(atom, i)
            area = neighbor_area_lists(atom, i)
            if ((edge >= edge_min) .AND. (edge <= edge_max)) then
              atom_count_list_separate(edge - edge_min + 1) = atom_count_list_separate(edge - edge_min + 1) + 1
              atom_area_stats_separate(edge - edge_min + 1, atom_count_list_separate(edge - edge_min + 1)) = area
            else if ((edge > edge_max) .AND. (include_beyond_edge_max == 1)) then
              atom_count_list_separate(edge_max - edge_min + 1) = atom_count_list_separate(edge_max - edge_min + 1) + 1
              atom_area_stats_separate(edge_max - edge_min + 1, atom_count_list_separate(edge_max - edge_min + 1)) = area
            end if
          end do

          do j = 1, edge_max - edge_min + 1
            area_stats_separate(atom, (j - 1) * 5 + 1 : j * 5) = &
              all_stats(atom_area_stats_separate(j, 1 : atom_count_list_separate(j)), atom_count_list_separate(j))
          end do
          deallocate(atom_area_stats_separate)
        end do
    end subroutine voronoi_area_stats_separate

    subroutine voronoi_vol_stats(vol_stats, &
        n_atoms, n_neighbor_limit, n_neighbor_list, neighbor_vol_lists)
        integer :: n_atoms, n_neighbor_limit
        integer, dimension(n_atoms) :: n_neighbor_list
        REAL(8), dimension(n_atoms, n_neighbor_limit) :: neighbor_vol_lists
        REAL(8), dimension(n_atoms, 5) :: vol_stats

!f2py   intent(in) :: n_atoms, n_neighbor_limit
!f2py   intent(in) :: n_neighbor_list
!f2py   intent(in) :: neighbor_vol_lists
!f2py   intent(in, out) :: vol_stats

        call voronoi_area_stats(vol_stats, n_atoms, n_neighbor_limit, n_neighbor_list, neighbor_vol_lists)
    end subroutine voronoi_vol_stats


    subroutine voronoi_vol_stats_separate(vol_stats_separate, &
            n_atoms, n_neighbor_limit, n_neighbor_list, neighbor_edge_lists, &
            neighbor_vol_lists, edge_min, edge_max, include_beyond_edge_max)
        integer :: n_atoms, n_neighbor_limit
        integer, dimension(n_atoms) :: n_neighbor_list
        integer, dimension(n_atoms, n_neighbor_limit) :: neighbor_edge_lists
        REAL(8), dimension(n_atoms, n_neighbor_limit) :: neighbor_vol_lists
        integer :: edge_min, edge_max, include_beyond_edge_max
        REAL(8), allocatable:: atom_vol_stats_separate(:,:)
        integer, dimension(edge_max - edge_min + 1) :: atom_count_list_separate
        REAL(8), dimension(n_atoms, 5 * (edge_max - edge_min + 1)) :: vol_stats_separate

!f2py   intent(in) :: n_atoms, n_neighbor_limit
!f2py   intent(in) :: n_neighbor_list
!f2py   intent(in) :: neighbor_edge_lists, neighbor_vol_lists
!f2py   intent(in) :: edge_min, edge_max, include_beyond_edge_max
!f2py   intent(in, out) :: vol_stats_separate

        call voronoi_area_stats_separate(vol_stats_separate, &
            n_atoms, n_neighbor_limit, n_neighbor_list, neighbor_edge_lists, &
            neighbor_vol_lists, edge_min, edge_max, include_beyond_edge_max)

    end subroutine voronoi_vol_stats_separate



    subroutine voronoi_distance_stats(distance_stats, &
        n_atoms, n_neighbor_limit, n_neighbor_list, neighbor_distance_lists)

        use :: a_stats, only : all_stats
        use :: c_stats, only : customize_stats

        integer :: n_atoms, n_neighbor_limit
        integer, dimension(n_atoms) :: n_neighbor_list
        REAL(8), dimension(n_atoms, n_neighbor_limit) :: neighbor_distance_lists
        REAL(8), dimension(n_atoms, 5) :: distance_stats

!f2py   intent(in) :: n_atoms, n_neighbor_limit
!f2py   intent(in) :: n_neighbor_list
!f2py   intent(in) :: neighbor_distance_lists
!f2py   intent(in, out) :: distance_stats

        integer  :: atom, sum_stats_types
        REAL(8) :: distance_mean
        integer, dimension(6) :: stats_types

        do atom = 1, n_atoms
            distance_stats(atom, :) = all_stats(neighbor_distance_lists(atom, :), n_neighbor_list(atom))
        end do
    end subroutine voronoi_distance_stats
