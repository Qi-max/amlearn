!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! INPUT & OUTPUT:
! n_atoms:                    int
! atom_type:                  int (n_atoms)
! atom_coords:                real(n_atoms, 3)
! distance_cutoff:                     real
! allow_neighbor_limit:       int, e.g. 80
! n_neighbor_limit:           int, e.g. 50
! pbc:                        int(3), e.g. [1, 1, 1]
! Bds:                        real(3, 2)
!
! n_neighbor_list:            int (n_atoms),
! neighbor_lists:             int (n_atoms, n_neighbor_limit)
! neighbor_distance_lists:    real(n_atoms, n_neighbor_limit)
! n_neighbor_max:             int
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


module distance_nn

  use :: distance
  use :: quicksort

contains
    ! if final neighbors> n_neighbor_limit choose the smallest ones
    subroutine distance_neighbor(n_atoms, atom_type, atom_coords, distance_cutoff, &
        allow_neighbor_limit, n_neighbor_limit, pbc, Bds, &
        n_neighbor_max, n_neighbor_list, neighbor_lists, neighbor_distance_lists)

        integer :: n_atoms, allow_neighbor_limit, n_neighbor_limit, n_neighbor_max
        REAL(8) :: distance_cutoff
        integer, dimension(n_atoms):: atom_type, n_neighbor_list
        REAL(8), dimension(n_atoms, 3):: atom_coords
        integer, dimension(n_atoms, n_neighbor_limit):: neighbor_lists
        REAL(8), dimension(n_atoms, n_neighbor_limit):: neighbor_distance_lists
        integer, dimension(3) :: pbc
        REAL(8), dimension(3, 2) :: Bds

!f2py   intent(in) n_atoms, atom_type, atom_coords, distance_cutoff
!f2py   intent(in) allow_neighbor_limit, n_neighbor_limit, pbc, Bds
!f2py   intent(in, out) n_neighbor_list, n_neighbor_max
!f2py   intent(in, out) neighbor_lists, neighbor_distance_lists

        integer :: atom, i, j, possible_n_neighbor, pop_n_neighbor
        REAL(8), dimension(n_neighbor_limit, 2) :: possible_list
        REAL(8) :: d
        REAL(8), dimension(3) :: r

        n_neighbor_max = 0

        do atom = 1, n_atoms
          write(*, *) 'atom is: ', atom
!          if (atom>5) then
!              exit
!          end if
          possible_list = 0

          possible_n_neighbor = 0
          pop_n_neighbor = 0
          do i = 1, n_atoms
            call distance_info(atom_coords(atom, :), atom_coords(i, :), Bds, pbc, r, d)
            if((i /= atom).and.(d < distance_cutoff)) then
              possible_n_neighbor = possible_n_neighbor + 1
              possible_list(possible_n_neighbor, 1) = d   !! distance
              possible_list(possible_n_neighbor, 2) = i   !! neighbor_id
            end if
          end do

          if(possible_n_neighbor > allow_neighbor_limit)    then
            write(*,*) "possible_n_neighbor OUT of allow_neighbor_limit"
          end if

          if(possible_n_neighbor > n_neighbor_max) then
            n_neighbor_max = possible_n_neighbor
          end if
!          write(*, *) 'in 1'
          if(possible_n_neighbor <= n_neighbor_limit) then
!              write(*, *) 'in 21f'
!              write(*, *) 'possible_list', possible_list(:, 2)

            neighbor_distance_lists(atom, :) = possible_list(:, 1)
            neighbor_lists(atom, :) = possible_list(:, 2)
            n_neighbor_list(atom) = possible_n_neighbor
!                        write(*, *) 'in 21'

          else
!                                      write(*, *) 'in 22f'

            call quick_sort(possible_list, 1, 1, possible_n_neighbor)
!!sort_col,can be used as multiple array?? (:, 1)??
            neighbor_distance_lists(atom, :) = possible_list(1:n_neighbor_limit, 1)
            neighbor_lists(atom, :) = possible_list(1:n_neighbor_limit, 2)
            n_neighbor_list(atom) = n_neighbor_limit
!                        write(*, *) 'in 22'

          end if
        end do

    end subroutine distance_neighbor

end module distance_nn