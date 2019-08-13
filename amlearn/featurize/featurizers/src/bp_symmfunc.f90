! center_atom_ids: (n_center_atoms)
! center_atom_coords: (n_center_atoms, 3)
! atom_ids: (n_atoms)
! atom_types: (n_atoms)
! atom_type_symbols: (n_atom_types)
! atom_coords: (n_atoms, 3)
! radial_distances: (n_radial_distances), make sure this is sorted from low-high
!                   such as np.arange(0, 5.1, 0.1), pls include the highest bound
! radial funcs: (n_center_atoms, (n_radial_distances-1) * n_atom_types)
! for example, for 100 center atoms with 51 radial_distances and A-B-binary atoms,
! the returned radial funcs would be (100, 50 * 2)


subroutine bp_radial(n_center_atoms, center_atom_ids, center_atom_coords, &
    n_atoms, atom_ids, atom_types, atom_type_symbols, n_atom_types, atom_coords, &
    pbc, Bds, &
    cutoff_s, cutoff_r, delta_r, n_r, radial_funcs)

    implicit none
    interface
     function average(list, len)
        integer, intent(in) :: len
        REAL(8), dimension(len), intent(in) :: list
        REAL(8) :: average
     end function
  	end interface

    integer :: n_center_atoms
    integer, dimension(n_center_atoms):: center_atom_ids
    REAL(8), dimension(n_center_atoms, 3):: center_atom_coords

    integer :: n_atoms, n_atom_types
    integer, dimension(n_atoms):: atom_ids, atom_types
    integer, dimension(n_atom_types):: atom_type_symbols
    REAL(8), dimension(n_atoms, 3):: atom_coords
    integer, dimension(3) :: pbc
    REAL(8), dimension(3, 2) :: Bds
    REAL(8) :: cutoff_s, cutoff_r, delta_r
    integer :: n_r
    REAL(8), dimension(n_center_atoms, n_r*n_atom_types):: radial_funcs
    ! dimension: bin_num, types of atom center

!f2py   intent(in) n_center_atoms, center_atom_ids, center_atom_coords
!f2py   intent(in) n_atoms, atom_ids, atom_types, atom_type_symbols, n_atom_types, atom_coords, pbc, Bds
!f2py   intent(in) cutoff_s, cutoff_r, delta_r, n_r
!f2py   intent(in, out) radial_funcs

    integer :: atom, i, j, range, type
    REAL(8) :: rij, r_ref
    REAL(8), dimension(3) :: r

    radial_funcs = 0

!    distance_cutoff = radial_distances(-1)
!    write(*, *) "Calculated distance cutoff", distance_cutoff

    do atom = 1, n_center_atoms
        write(*, *) atom, center_atom_ids(atom)
        do i = 1, n_atoms
            if (atom_ids(i) /= center_atom_ids(atom)) then
                call distance_info(center_atom_coords(atom, :), atom_coords(i, :), Bds, pbc, r, rij)
                if(rij <= cutoff_s) then
!                    write(*, *) "neighbor", i, rij
                    do type = 1, n_atom_types
                        if (atom_types(i) == atom_type_symbols(type)) then
                            do j = 1, n_r
                                r_ref = (j - 1) * delta_r
!                                write(*, *) "atom type", atom_types(i), j + (type-1) * n_r
                                radial_funcs(atom, j + (type-1) * n_r) = &
                                    radial_funcs(atom, j + (type-1) * n_r) + &
                                    exp(-1 / (2* delta_r ** 2) * (rij - r_ref) ** 2)
!                                write(*, *) "center", atom, "neighbor", i, j, radial_funcs(atom, j + (type-1) * n_r)
                            end do
                        end if
                    end do
                end if
            end if
        end do
    end do

end subroutine bp_radial


subroutine bp_angular(n_center_atoms, center_atom_ids, center_atom_coords, &
    n_atoms, atom_ids, atom_types, atom_type_symbols, n_atom_types, atom_coords, &
    pbc, Bds, &
    ksaais, lambdas, zetas, n_params, cutoff_s, angular_funcs)

    implicit none
    interface
     function average(list, len)
        integer, intent(in) :: len
        REAL(8), dimension(len), intent(in) :: list
        REAL(8) :: average
     end function
  	end interface

    integer :: n_center_atoms
    integer, dimension(n_center_atoms):: center_atom_ids
    REAL(8), dimension(n_center_atoms, 3):: center_atom_coords

    integer :: n_atoms, n_atom_types
    integer, dimension(n_atoms):: atom_ids, atom_types
    integer, dimension(n_atom_types):: atom_type_symbols
    REAL(8), dimension(n_atoms, 3):: atom_coords
    integer, dimension(3) :: pbc
    REAL(8), dimension(3, 2) :: Bds
    integer :: n_params
    REAL(8), dimension(n_params):: ksaais, lambdas, zetas
    REAL(8) :: cutoff_s
    REAL(8), dimension(n_center_atoms, n_atom_types*(n_atom_types+1) / 2 *n_params):: angular_funcs

    ! dimension: bin_num, types of atom center

!f2py   intent(in) n_center_atoms, center_atom_ids, center_atom_coords
!f2py   intent(in) n_atoms, atom_ids, atom_types, atom_type_symbols, n_atom_types, atom_coords, pbc, Bds
!f2py   intent(in) ksaais, lambdas, zetas, n_params, cutoff_s
!f2py   intent(in, out) angular_funcs

    integer :: atom, i, j, k, range, type, type1, type2, n_neigh, p
    REAL(8) :: rij, rjk, rik, r_ref, cosijk, func
    REAL(8), dimension(3) :: r
    REAL(8), dimension(n_atom_types, 60, 4):: neigh_dists
    integer, dimension(n_atom_types, 60):: neigh_ids
    integer, dimension(n_atom_types):: neigh_nums

    angular_funcs = 0

!    distance_cutoff = radial_distances(-1)
!    write(*, *) "Calculated distance cutoff", distance_cutoff

    do i = 1, n_center_atoms
        write(*, *) i, center_atom_ids(i)
        neigh_nums = 0
        neigh_dists = 0
        neigh_ids = 0
        do atom = 1, n_atoms
            if (atom_ids(atom) /= center_atom_ids(i)) then
                call distance_info(center_atom_coords(i, :), atom_coords(atom, :), Bds, pbc, r, rij)
                if(rij <= cutoff_s) then
!                    write(*, *) "neighbor", atom, rij
                    do type = 1, n_atom_types
                        if (atom_types(atom) == atom_type_symbols(type)) then
                            neigh_nums(type) = neigh_nums(type) + 1
                            neigh_dists(type, neigh_nums(type), 1:3) = r
                            neigh_dists(type, neigh_nums(type), 4) = rij
                            neigh_ids(type, neigh_nums(type)) = atom
                        end if
                    end do
                end if
            end if
        end do
        do p = 1, n_params
            do type1 = 1, n_atom_types
                do type2 = type1, n_atom_types
                    range = type1 + type2 - 1 ! determine the range in the angular_func, valid for ternary?
                    if (type1 == type2) then
                        do j = 1, neigh_nums(type1)
                            do k = j + 1, neigh_nums(type2)
                                rij = neigh_dists(type1, j, 4)
                                rik = neigh_dists(type2, k, 4)
                                call distance_info(atom_coords(neigh_ids(type1, j), :), &
                                        atom_coords(neigh_ids(type2, k), :), Bds, pbc, r, rjk)
!                                dot_product = neigh_dists(type1, j, 1) * neigh_dists(type2, k, 1) + &
!                                        neigh_dists(type1, j, 2) * neigh_dists(type2, k, 2) + &
!                                        neigh_dists(type1, j, 3) * neigh_dists(type2, k, 3)
!                                cosijk = dot_product / (rij * rik)
                                cosijk = DOT_PRODUCT(neigh_dists(type1, j, 1:3), neigh_dists(type2, k, 1:3)) / (rij * rik)
                                func = exp(-(rij**2 + rik**2 + rjk**2)/(ksaais(p)**2)) * (1 + lambdas(p)*cosijk)**zetas(p)
                                angular_funcs(i, p + (range-1)*n_params) = &
                                  angular_funcs(i, p + (range-1)*n_params) + func
!                                write(*, *) p, type1, type2, neigh_ids(type1, j), neigh_ids(type1, k), rij, rik, rjk, cosijk, func
                            end do
                        end do
                    else
                        do j = 1, neigh_nums(type1)
                            do k = 1, neigh_nums(type2)
                                rij = neigh_dists(type1, j, 4)
                                rik = neigh_dists(type2, k, 4)
                                call distance_info(atom_coords(neigh_ids(type1, j), :), &
                                        atom_coords(neigh_ids(type2, k), :), Bds, pbc, r, rjk)
!                                dot_product = neigh_dists(type1, j, 1) * neigh_dists(type2, k, 1) + &
!                                        neigh_dists(type1, j, 2) * neigh_dists(type2, k, 2) + &
!                                        neigh_dists(type1, j, 3) * neigh_dists(type2, k, 3)
!                                cosijk = dot_product / (rij * rik)
                                cosijk = DOT_PRODUCT(neigh_dists(type1, j, 1:3), neigh_dists(type2, k, 1:3)) / (rij * rik)
                                func = exp(-(rij**2 + rik**2 + rjk**2)/(ksaais(p)**2)) * (1 + lambdas(p)*cosijk)**zetas(p)
                                angular_funcs(i, p + (range-1)*n_params) = &
                                  angular_funcs(i, p + (range-1)*n_params) + func
!                                write(*, *) p, type1, type2, neigh_ids(type1, j), neigh_ids(type1, k), rij, rik, rjk, cosijk, func
                            end do
                        end do
                    end if
                end do
            end do
        end do
    end do

end subroutine bp_angular


subroutine bp_radial_qw(n_center_atoms, center_atom_ids, center_atom_coords, &
    n_atoms, atom_ids, atom_types, atom_type_symbols, n_atom_types, atom_coords, &
    pbc, Bds, &
    cutoff_r, delta_r, n_r, radial_funcs)

    implicit none
    interface
     function average(list, len)
        integer, intent(in) :: len
        REAL(8), dimension(len), intent(in) :: list
        REAL(8) :: average
     end function
  	end interface

    integer :: n_center_atoms
    integer, dimension(n_center_atoms):: center_atom_ids
    REAL(8), dimension(n_center_atoms, 3):: center_atom_coords

    integer :: n_atoms, n_atom_types
    integer, dimension(n_atoms):: atom_ids, atom_types
    integer, dimension(n_atom_types):: atom_type_symbols
    REAL(8), dimension(n_atoms, 3):: atom_coords
    integer, dimension(3) :: pbc
    REAL(8), dimension(3, 2) :: Bds
    REAL(8) :: cutoff_r, delta_r
    integer :: n_r
    REAL(8), dimension(n_center_atoms, n_r*n_atom_types):: radial_funcs
    ! dimension: bin_num, types of atom center

!f2py   intent(in) n_center_atoms, center_atom_ids, center_atom_coords
!f2py   intent(in) n_atoms, atom_ids, atom_types, atom_type_symbols, n_atom_types, atom_coords, pbc, Bds
!f2py   intent(in) cutoff_r, delta_r, n_r
!f2py   intent(in, out) radial_funcs

    integer :: atom, i, j, range, type
    REAL(8) :: rij, r_ref
    REAL(8), dimension(3) :: r

    radial_funcs = 0

!    distance_cutoff = radial_distances(-1)
!    write(*, *) "Calculated distance cutoff", distance_cutoff

    do atom = 1, n_center_atoms
        write(*, *) atom, center_atom_ids(atom)
        do i = 1, n_atoms
            if (atom_ids(i) /= center_atom_ids(atom)) then
                call distance_info(center_atom_coords(atom, :), atom_coords(i, :), Bds, pbc, r, rij)
!                if(rij <= cutoff_s) then
!                    write(*, *) "neighbor", i, rij
                if(rij <= cutoff_r) then
                    range = ceiling(rij / delta_r)
                    r_ref = (range - 1) * delta_r ! get the lower r of the shell
!                    write(*, *) "neighbor", i, atom_types(i), rij, range, r_ref
                    do type = 1, n_atom_types
                        if (atom_types(i) == atom_type_symbols(type)) then
!                            write(*, *) "atom type", atom_types(i), range + (type-1) * n_r
                            radial_funcs(atom, range + (type-1) * n_r) = &
                                radial_funcs(atom, range + (type-1) * n_r) + &
                                    exp(-1 / (2* delta_r ** 2) * (rij - r_ref) ** 2)
!                                write(*, *) "center", atom, "neighbor", i, j, radial_funcs(atom, j + (type-1) * n_r)
!                            end do
                        end if
                    end do
                end if
            end if
        end do
    end do

end subroutine bp_radial_qw


subroutine distance_info(atom_coords_i, atom_coords_j, Bds, pbc, r, d)

	REAL(8), dimension(3), intent(in) :: atom_coords_i, atom_coords_j
	REAL(8), dimension(3), intent(out):: r
	REAL(8), intent(out) :: d
	REAL(8), dimension(3, 2) :: Bds
	REAL(8), dimension(3) :: Lens
	integer, dimension(3) :: pbc

	Lens(1) = Bds(1, 2) - Bds(1, 1)
	Lens(2) = Bds(2, 2) - Bds(2, 1)
	Lens(3) = Bds(3, 2) - Bds(3, 1)

	do m = 1, 3
	r(m) = atom_coords_i(m) - atom_coords_j(m)
	!            write(*, *) r(m), atom_coords_i(m), atom_coords_j(m)
	if (pbc(m) == 1) then
	  if (r(m) > Lens(m)*0.5) then
	    r(m) = (r(m) - Lens(m))
	  else if (r(m) < Lens(m)*(-0.5)) then
	    r(m) = (r(m) + Lens(m))
	  else
	    r(m) = r(m)  ! why divided by 2?
	  end if
	end if
	end do

	d = sqrt(r(1)**2 + r(2)**2 + r(3)**2)

end subroutine distance_info



function average(list, len) result(mean)
    REAL(8), dimension(len) :: list
    integer :: len, atom
    REAL(8) :: mean
    mean = 0
    if (len.eq.0) then
        len = 1
    end if

    do atom = 1, len
        mean = mean + list(atom)
    end do

    mean = mean / len

end function average
