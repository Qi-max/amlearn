
subroutine fractal(n_atoms, atom_type, atom_coords, pbc, bds, &
    cutoff, bin, bin_num, center_types, center_types_num, &
    fractal_distancewise, fractal_accumulative)

    implicit none
    interface 
     function average(list, len)
        integer, intent(in) :: len
        REAL(8), dimension(len), intent(in) :: list
        REAL(8) :: average
     end function
  	end interface

    integer :: n_atoms
    integer, dimension(n_atoms):: atom_type
    REAL(8), dimension(n_atoms, 3):: atom_coords
    integer, dimension(3) :: pbc
    REAL(8), dimension(3, 2) :: bds
    REAL(8) :: cutoff, bin
    integer :: bin_num, center_types_num
    integer, dimension(center_types_num):: center_types
    ! dimension: types of atom center
    REAL(8), dimension(bin_num, center_types_num):: fractal_distancewise, fractal_accumulative
    ! dimension: bin_num, types of atom center

!f2py   intent(in) n_atoms, atom_type, atom_coords, pbc, bds
!f2py   intent(in) cutoff, bin, bin_num, center_types, center_types_num
!f2py   intent(in, out) fractal_distancewise, fractal_accumulative

    integer :: atom, i, j, range, type
    REAL(8) :: d
    integer(2), dimension(n_atoms, n_atoms) :: distance_ranges
    integer, dimension(center_types_num):: fractal_atoms  ! dimension: types of atom center
    REAL(8), dimension(3) :: r

    distance_ranges = -1
    fractal_atoms= 0
    fractal_distancewise = 0
    fractal_accumulative = 0

    do atom = 1, n_atoms
        write(*, *) atom
        do type = 1, center_types_num
            if((center_types(type) == 777) .or. (atom_type(atom) == center_types(type))) then
                fractal_atoms(type) = fractal_atoms(type) + 1
            end if
        end do

        do i = 1, atom-1
            range = distance_ranges(i, atom)
            do type = 1, center_types_num
                if((center_types(type) == 777) .or. (atom_type(atom) == center_types(type))) then
                    if (range>=0) then
                        fractal_distancewise(range, type) = fractal_distancewise(range, type) + 1
                    end if
                end if
            end do
        end do

        do i = atom+1, n_atoms
            call distance_info(atom_coords(atom, :), atom_coords(i, :), bds, pbc, r, d)
            if(d <= cutoff) then
                range = floor(d/bin)
                distance_ranges(atom, i) = range
                do type = 1, center_types_num
                    if((center_types(type) == 777) .or. (atom_type(atom) == center_types(type))) then
                        fractal_distancewise(range, type) = fractal_distancewise(range, type) + 1
                    end if
                end do
            end if
        end do
    end do

    do i = 1, center_types_num
        fractal_distancewise(:, i) = fractal_distancewise(:, i) / fractal_atoms(i)
    end do

    do i = 2, bin_num
        fractal_accumulative(i, :) = fractal_accumulative(i-1, :) + fractal_distancewise(i, :)
    end do

end subroutine fractal


subroutine fractal_intense(n_atoms, atom_type, atom_coords, pbc, bds, &
    cutoff, bin, bin_num, center_types, center_types_num, &
    fractal_distancewise, fractal_accumulative)

    implicit none
    interface
     function average(list, len)
        integer, intent(in) :: len
        REAL(8), dimension(len), intent(in) :: list
        REAL(8) :: average
     end function
  	end interface

    integer :: n_atoms
    integer, dimension(n_atoms):: atom_type
    REAL(8), dimension(n_atoms, 3):: atom_coords
    integer, dimension(3) :: pbc
    REAL(8), dimension(3, 2) :: bds
    REAL(8) :: cutoff, bin
    integer :: bin_num, center_types_num
    integer, dimension(center_types_num):: center_types   ! dimension: types of atom center
    REAL(8), dimension(bin_num, center_types_num):: fractal_distancewise, fractal_accumulative
    ! dimension: bin_num, types of atom center

!f2py   intent(in) n_atoms, atom_type, atom_coords, pbc, bds
!f2py   intent(in) cutoff, bin, bin_num, center_types, center_types_num
!f2py   intent(in, out) fractal_distancewise, fractal_accumulative

    integer :: atom, i, j, range, type
    REAL(8) :: d
    integer, dimension(center_types_num):: fractal_atoms  ! dimension: types of atom center
    REAL(8), dimension(3) :: r

    fractal_atoms= 0
    fractal_distancewise = 0
    fractal_accumulative = 0

    do atom = 1, n_atoms
        write(*, *) atom
        do type = 1, center_types_num
            if((center_types(type) == 777) .or. (atom_type(atom) == center_types(type))) then
                fractal_atoms(type) = fractal_atoms(type) + 1
            end if
        end do

        do i = 1, n_atoms
            if (i/=atom) then
                call distance_info(atom_coords(atom, :), atom_coords(i, :), bds, pbc, r, d)
                if(d <= cutoff) then
                    range = floor(d/bin)
                    do type = 1, center_types_num
                        if((center_types(type) == 777) .or. (atom_type(atom) == center_types(type))) then
                            fractal_distancewise(range, type) = fractal_distancewise(range, type) + 1
                        end if
                    end do
                end if
            end if
        end do
    end do

    do i = 1, center_types_num
        fractal_distancewise(:, i) = fractal_distancewise(:, i) / fractal_atoms(i)
    end do

    do i = 2, bin_num
        fractal_accumulative(i, :) = fractal_accumulative(i-1, :) + fractal_distancewise(i, :)
    end do

end subroutine fractal_intense


subroutine distance_info(atom_coords_i, atom_coords_j, bds, pbc, r, d)

	REAL(8), dimension(3), intent(in) :: atom_coords_i, atom_coords_j
	REAL(8), dimension(3), intent(out):: r
	REAL(8), intent(out) :: d
	REAL(8), dimension(3, 2) :: bds
	REAL(8), dimension(3) :: Lens
	integer, dimension(3) :: pbc

	Lens(1) = bds(1, 2) - bds(1, 1)
	Lens(2) = bds(2, 2) - bds(2, 1)
	Lens(3) = bds(3, 2) - bds(3, 1)

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
