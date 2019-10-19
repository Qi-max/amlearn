!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! INPUT & OUTPUT:
! n_atoms:                    int
! atom_type:                  int (n_atoms)
! atom_coords:                real(n_atoms, 3)
! distance_cutoff:                     real
! allow_neighbor_limit:       int, e.g. 80
! n_neighbor_limit:           int, e.g. 50
! pbc:                        int(3), e.g. [1, 1, 1]
! bds:                        real(3, 2)
!
! n_neighbor_list:            int (n_atoms),
! neighbor_lists:             int (n_atoms, n_neighbor_limit)
! neighbor_distance_lists:    real(n_atoms, n_neighbor_limit)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


! if final neighbors> n_neighbor_limit choose the smallest ones
subroutine distance_neighbor(n_atoms, atom_type, atom_coords, distance_cutoff, &
    allow_neighbor_limit, n_neighbor_limit, pbc, bds, &
    n_neighbor_list, neighbor_lists, neighbor_distance_lists, &
    print_freq)

    use :: distance
    use :: quicksort

    integer :: n_atoms, allow_neighbor_limit, n_neighbor_limit
    REAL(8) :: distance_cutoff
    integer, dimension(n_atoms):: atom_type, n_neighbor_list
    REAL(8), dimension(n_atoms, 3):: atom_coords
    integer, dimension(n_atoms, n_neighbor_limit):: neighbor_lists
    REAL(8), dimension(n_atoms, n_neighbor_limit):: neighbor_distance_lists
    integer, dimension(3) :: pbc
    REAL(8), dimension(3, 2) :: bds
    integer :: print_freq

!f2py   intent(in) n_atoms, atom_type, atom_coords, distance_cutoff
!f2py   intent(in) allow_neighbor_limit, n_neighbor_limit, pbc, bds, print_freq
!f2py   intent(in, out) n_neighbor_list
!f2py   intent(in, out) neighbor_lists, neighbor_distance_lists

    integer :: atom, i, j, possible_n_neighbor, pop_n_neighbor
    REAL(8), dimension(n_neighbor_limit, 2) :: possible_list
    REAL(8) :: d
    REAL(8), dimension(3) :: r

    do atom = 1, n_atoms
      if (i == 0) then
        write(*, *) "start distance nn"
      else if (mod(i, print_freq) == 0) then
        write(*, *) "distance nn: atom", i
      end if

      possible_list = 0
      possible_n_neighbor = 0
      pop_n_neighbor = 0
      do i = 1, n_atoms
        call distance_info(atom_coords(atom, :), atom_coords(i, :), bds, pbc, r, d)
        if((i /= atom).and.(d < distance_cutoff)) then
          possible_n_neighbor = possible_n_neighbor + 1
          possible_list(possible_n_neighbor, 1) = d   !! distance
          possible_list(possible_n_neighbor, 2) = i   !! neighbor_id
        end if
      end do

      if(possible_n_neighbor > allow_neighbor_limit)    then
        write(*,*) "possible_n_neighbor OUT of allow_neighbor_limit"
      end if

!      if(possible_n_neighbor > n_neighbor_max) then
!        n_neighbor_max = possible_n_neighbor
!      end if

      if(possible_n_neighbor <= n_neighbor_limit) then
        neighbor_distance_lists(atom, :) = possible_list(:, 1)
        neighbor_lists(atom, :) = possible_list(:, 2)
        n_neighbor_list(atom) = possible_n_neighbor
      else
        call quick_sort(possible_list, 1, 1, possible_n_neighbor)
        neighbor_distance_lists(atom, :) = possible_list(1:n_neighbor_limit, 1)
        neighbor_lists(atom, :) = possible_list(1:n_neighbor_limit, 2)
        n_neighbor_list(atom) = n_neighbor_limit
      end if
    end do
    write(*, *) "finish distance nn for atoms: ", n_atoms
end subroutine distance_neighbor
