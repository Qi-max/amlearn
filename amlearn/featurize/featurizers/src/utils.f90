    module determinant_func
    CONTAINS
        subroutine determinant(matrix, deter_value)
            REAL(8), dimension(3, 3), intent(in) :: matrix
            REAL(8), intent(out) :: deter_value
            deter_value=(matrix(1,1)*matrix(2,2)*matrix(3,3)+matrix(1,2)*matrix(2,3)*matrix(3,1)&
                    +matrix(1,3)*matrix(2,1)*matrix(3,2)&
                    -matrix(3,1)*matrix(2,2)*matrix(1,3)-matrix(2,1)*matrix(1,2)*matrix(3,3)&
                    -matrix(1,1)*matrix(3,2)*matrix(2,3))
        end subroutine determinant
    end module determinant_func


    module distance

    CONTAINS
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
    end module distance

    module quicksort
    CONTAINS
      recursive subroutine quick_sort(array, sort_col, &
          sort_index_min, sort_index_max)
      implicit none
      integer :: sort_col, sort_index_min, sort_index_max
      REAL(8), dimension(:, :) :: array

      REAL(8), dimension(size(array, 2)) :: trans
      REAL(8) :: x
      integer i, j

      x = array((sort_index_min + sort_index_max) / 2, sort_col)   !!integer division
      i = sort_index_min
      j = sort_index_max

      do
        do while (array(i, sort_col) < x)
          i = i + 1
        end do
        do while (x < array(j, sort_col))
          j = j - 1
        end do

        if (i >= j) then
            exit
        end if

        trans = array(i, :)
        array(i, :) = array(j, :)
        array(j, :) = trans

        i = i + 1
        j = j - 1
      end do

      if (sort_index_min < i - 1) then
        call quick_sort(array, sort_col, sort_index_min, i - 1)
      end if

      if (j + 1 < sort_index_max) then
        call quick_sort(array, sort_col, j + 1, sort_index_max)
      end if

      end subroutine quick_sort
    end module quicksort



    ! for boop
    module CG_function
    CONTAINS
      function calculate_CG(half_l_min, half_l_max) result(CG)
          integer, intent(in) :: half_l_min, half_l_max
          integer:: l_min, l_max
          REAL(8), dimension(half_l_min:half_l_max, -half_l_max*2:half_l_max*2, &
              -half_l_max*2:half_l_max*2, -half_l_max*2:half_l_max*2):: CG

          integer :: i, j, l, m1, m2, m3, v, s
          REAL(8) :: v0, v1, v2
          REAL(8), dimension(0:100) :: fac

          do i = 0, 50
            if(i == 0) then
              fac(i)=0
            else
              do j = 1, i
                fac(i) = fac(i) + log10(1.0*j)
              end do
            end if
          end do


          CG = 0
          l_min = half_l_min * 2
          l_max = half_l_max * 2

          do l = l_min, l_max, 2
            do m1 = -l, l
              do m2 = max(-l, -l-m1), min(l, l-m1)
                m3 = -(m1 + m2)
                s = 0
                v2 = 0
                do v = max(0, -m1, m2), min(l, l-m1, l+m2)
                  s = s + 1
                  v1 = fac(v) + fac(l-v) + fac(l-m1-v) + fac(l+m2-v) + fac(-m2+v) + fac(m1+v)
                  if(s.eq.1) then
                    v0 = v1
                  end if
                  v2 = v2 + (-1)**(-v) * 10**(1.0*(v0-v1))
                end do
                CG(l/2, m1, m2, m3) = 10**(1.0/2 * &
                    (log10(1.0*2*l+1) + 3*fac(l)-fac(3*l+1) + &
                    fac(l+m1)+fac(l-m1)+fac(l+m2)+fac(l-m2)+fac(l+m3)+fac(l-m3))-v0) * v2 * &
                    (-1)**(-m3) / sqrt(1.0*(2*l+1))
              end do
            end do
          end do

      end function calculate_CG
    end module



    ! for boop
    module angle

    CONTAINS
      subroutine angle_info(atom_coords_i, atom_coords_j, Bds, pbc, ksai, theta)
      REAL(8), dimension(3), intent(in) :: atom_coords_i, atom_coords_j
      REAL(8), intent(out) :: ksai, theta
      REAL(8), dimension(3, 2), intent(in) :: Bds
      integer, dimension(3), intent(in) :: pbc
      REAL(8), dimension(3) :: r
      REAL(8) :: d
      REAL(8), dimension(3) :: Lens
      integer :: m

      Lens(1) = Bds(1, 2) - Bds(1, 1)
      Lens(2) = Bds(2, 2) - Bds(2, 1)
      Lens(3) = Bds(3, 2) - Bds(3, 1)

      do m = 1, 3
        r(m) = atom_coords_i(m) - atom_coords_j(m)
        if (pbc(m) == 1) then
          if (r(m) > Lens(m)*0.5) then
            r(m) = (r(m) - Lens(m)) / 2.0
          else if (r(m) < Lens(m)*(-0.5)) then
            r(m) = (r(m) + Lens(m)) / 2.0
          else
            r(m) = r(m) / 2.0  ! why divided by 2?
          end if
        end if
      end do

      d = sqrt(r(1)**2 + r(2)**2 + r(3)**2)
      ksai = atan(r(2) / r(1))
      theta = acos(r(3) / d)

      end subroutine angle_info
    end module



    module c_stats
    contains
    function customize_stats(list, stats_types, stats_types_sum, self_prop) result(stats)
        REAL(8), dimension(:) :: list
        integer, dimension(:) :: stats_types
        REAL(8) :: self_prop
        integer:: list_len, atom, i, j, stats_types_sum
        REAL(8), dimension(6) :: stats_all
        REAL(8), dimension(stats_types_sum) :: stats
        REAL(8) :: prop, sum_stat, mean_stat, std_stat, min_stat, max_stat, diff_stat

        list_len = size(list)
        min_stat = list(1)
        max_stat = min_stat
        sum_stat = 0
        mean_stat = 0
        std_stat = 0

        do atom = 1, list_len
            prop = list(atom)
            sum_stat = sum_stat + prop
            if (prop > max_stat) then
                max_stat = prop
            end if

            if (prop < min_stat) then
                min_stat = prop
            end if
        end do

        mean_stat = sum_stat / list_len

        if (stats_types(3) == 1) then
          do atom = 1, list_len
            std_stat = std_stat + (list(atom) - mean_stat)**2
          end do
          std_stat = sqrt(std_stat / list_len)
        end if

        if (stats_types(6) == 1) then
          diff_stat = mean_stat - self_prop
        end if

        stats_all(1) = sum_stat
        stats_all(2) = mean_stat
        stats_all(3) = std_stat
        stats_all(4) = min_stat
        stats_all(5) = max_stat
        stats_all(6) = diff_stat

        i = 0
        do j = 1, 6
          if (stats_types(j) == 1) then
            i = i + 1
            stats(i) = stats_all(j)
          end if
        end do

    end function customize_stats
    end module c_stats


    module a_stats
    contains
    function all_stats(list, len) result(stats)
        REAL(8), dimension(len) :: list(len)
        integer :: len, atom
        REAL(8) :: s, mean, sum, min, max
        REAL(16) :: std
        REAL(8), dimension(5) :: stats

        min = list(1)
        max = min
        mean = 0
        sum = 0
        std = 0
        if (len.eq.0) then
            len = 1
        end if
        do atom = 1, len
            s = list(atom)
            sum = sum + s
            if (s > max) then
                max = s
            end if

            if (s < min) then
                min = s
            end if
        end do

        mean = sum / len

        do atom = 1, len
            std = std + (list(atom) - mean)**2
        end do
        std = sqrt(std / len)


        stats(1) = sum
        stats(2) = mean
        stats(3) = std
        stats(4) = min
        stats(5) = max

    end function all_stats
    end module a_stats