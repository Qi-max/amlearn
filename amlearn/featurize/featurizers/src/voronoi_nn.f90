!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! INPUT & OUTPUT:
! n_atoms:                    int
! atom_type:                  int (n_atoms)
! atom_coords:                real(n_atoms, 3)
! cutoff:                     real
! allow_neighbor_limit:       int, e.g. 80
! n_neighbor_limit:           int, e.g. 50
! small_face_thres:           real e.g. 0.05
! pbc:                        int(3), e.g. [1, 1, 1]
! Bds:                        real(3, 2)
!
! n_neighbor_list:            int (n_atoms),
! neighbor_lists:             int (n_atoms, n_neighbor_limit)
! neighbor_area_lists:        real(n_atoms, n_neighbor_limit)
! neighbor_vol_lists:         real(n_atoms, n_neighbor_limit)
! neighbor_distance_lists:    real(n_atoms, n_neighbor_limit)
! neighbor_edge_lists:        int (n_atoms, n_neighbor_limit)
! n_neighbor_max:             int
! n_edge_max:                 int
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



!module voronoi_nn
!!
!!!  use :: distance
!!
!contains

subroutine determinant(face, deter_value)
    REAL(16), dimension(3, 3), intent(in) :: face
    REAL(16), intent(out) :: deter_value
    deter_value=(face(1,1)*face(2,2)*face(3,3)+face(1,2)*face(2,3)*face(3,1)&
            +face(1,3)*face(2,1)*face(3,2)&
            -face(3,1)*face(2,2)*face(1,3)-face(2,1)*face(1,2)*face(3,3)&
            -face(1,1)*face(3,2)*face(2,3))
    return
END subroutine determinant


subroutine voronoi(n_atoms, atom_type, atom_coords, cutoff, allow_neighbor_limit, &
    n_neighbor_limit, small_face_thres, pbc, Bds, &
    n_neighbor_list, neighbor_lists, &
    neighbor_area_lists, neighbor_vol_lists, neighbor_distance_lists, neighbor_edge_lists, &
    n_neighbor_max, n_edge_max)

    use :: distance

    integer :: n_atoms, allow_neighbor_limit, n_neighbor_limit, n_neighbor_max, n_edge_max
    REAL(8) :: cutoff, small_face_thres
    integer, dimension(3) :: pbc
    REAL(8), dimension(3, 2) :: Bds
    integer, dimension(n_atoms):: atom_type, n_neighbor_list
    REAL(8), dimension(n_atoms, 3):: atom_coords
    integer, dimension(n_atoms, n_neighbor_limit):: neighbor_lists, neighbor_edge_lists
    REAL(8), dimension(n_atoms, n_neighbor_limit):: neighbor_area_lists
    REAL(8), dimension(n_atoms, n_neighbor_limit):: neighbor_vol_lists
    REAL(8), dimension(n_atoms, n_neighbor_limit):: neighbor_distance_lists

!f2py   intent(in) n_atoms, atom_type, atom_coords, cutoff, small_face_thres
!f2py   intent(in) allow_neighbor_limit, n_neighbor_limit, pbc, Bds
!f2py   intent(in, out) n_neighbor_list
!f2py   intent(in, out) neighbor_lists, neighbor_edge_lists
!f2py   intent(in, out) neighbor_area_lists, neighbor_vol_lists, neighbor_distance_lists
!f2py   intent(in, out) n_neighbor_max, n_edge_max

    integer :: atom, i, j, l, k, s, possible_n_neighbor, CN_test
    integer, dimension(allow_neighbor_limit) :: bool, possible_neighbor_list, vertex_face
    integer, dimension(allow_neighbor_limit, allow_neighbor_limit) :: mianvoro
    REAL(8) :: d, d_temp, vertex(allow_neighbor_limit, 3)
    REAL(16) :: deter_value1, deter_value2, deter_value
    REAL(8) :: area_sum
    REAL(8), dimension(3) :: r, vv, a, vertex_x
    REAL(8), dimension(3,3) :: v
    REAL(16), dimension(3,3) :: face, face_2
    REAL(16), dimension(n_neighbor_limit):: neighbor_area_list_with_small
    REAL(16), dimension(n_neighbor_limit):: neighbor_vol_list_with_small
    integer, dimension(n_atoms) :: n_vertex_list
    integer, dimension(n_neighbor_limit, 3) :: vertex_info

!    write(*,*) "call fortran"

    n_neighbor_max = 0
    n_edge_max = 0

    do atom = 1, n_atoms
!      if (atom > 5) then
!              exit
!      end if
      write(*,*) "atom is : ", atom

      vertex = 0
      vertex_info = 0
      mianvoro = 0
      bool = 0
      possible_n_neighbor=0
      possible_neighbor_list = 0
      neighbor_area_list_with_small = 0
      neighbor_vol_list_with_small = 0
      vertex_face = 0
      do i = 1, n_atoms
        call distance_info(atom_coords(atom, :), atom_coords(i, :), Bds, pbc, r, d)
!        write(*,*) r, d
        if((i /= atom).and.(d < cutoff)) then
          possible_n_neighbor = possible_n_neighbor + 1
          possible_neighbor_list(possible_n_neighbor) = i
        end if
      end do
!      write(*,*) possible_n_neighbor
!      write(*,*) possible_neighbor_list
      if(possible_n_neighbor > allow_neighbor_limit)    then
        write(*,*) "possible_n_neighbor OUT of allow_neighbor_limit"
      end if

      n_vertex_list(atom) = 0

      do i = 1, possible_n_neighbor
        call distance_info(atom_coords(atom, :), atom_coords(possible_neighbor_list(i), :), Bds, pbc, r, d)
!        write(*, *) r
        v(1, :) = r(:)
        do j = i+1,possible_n_neighbor
          call distance_info(atom_coords(atom, :), atom_coords(possible_neighbor_list(j), :), Bds, pbc, r, d)
          v(2, :) = r(:)
          do k = j+1,possible_n_neighbor
            call distance_info(atom_coords(atom, :), atom_coords(possible_neighbor_list(k), :), Bds, pbc, r, d)
            v(3, :) = r(:)

            vv = 0
            do m = 1, 3
              do n = 1, 3
                vv(m) = vv(m) + v(m,n)*v(m,n)  !sum of |v|^2
              end do
            end do
!            write(*, *) vv

            do m = 1, 3
              do n = 1, 3
                face(m,n) = 2.0 * v(m,n) / vv(m)    !reciprocal space?
              end do
            end do

            face_2 = face
            do m = 1, 3
              face_2(m, 1)=1.0                      !set x = 1
            end do

            call determinant(face_2(1:3,1:3), deter_value1)
            call determinant(face(1:3,1:3), deter_value2)
!            deter_value1 = determinant(face_2(1:3,1:3))
!            deter_value2 = determinant(face(1:3,1:3))
            vertex_x(1)=deter_value1/deter_value2

            face_2 = face
            do m = 1, 3
              face_2(m, 2)=1.0                      !set y = 1
            end do

            call determinant(face_2(1:3,1:3), deter_value1)

!            deter_value1 = determinant(face_2(1:3,1:3))
!            deter_value2 = determinant(face(1:3,1:3))
            vertex_x(2)=deter_value1/deter_value2

            face_2 = face
            do m = 1, 3
              face_2(m, 3)=1.0                      !set y = 1
            end do


            call determinant(face_2(1:3,1:3), deter_value1)

!            deter_value1 = determinant(face_2(1:3, 1:3))
!            deter_value2 = determinant(face(1:3, 1:3))
            vertex_x(3)=deter_value1/deter_value2

            l=0
            s=0
            do l = 1, possible_n_neighbor
              call distance_info(atom_coords(atom, :), atom_coords(possible_neighbor_list(l), :), Bds, pbc, r, d_temp)
              a(:) = vertex_x(:) * r(:) / d_temp**2
!              write(*,*) vertex_x, r, a

              !if found the fourth possible_neighbor_list, set s=1 and skip the following iteration
              if((a(1) + a(2) + a(3)) > 0.5000001D0) then
                s = 1
                exit
              end if
            end do

            if(s == 0) then
              bool(i) = 1
              bool(j) = 1
              bool(k) = 1
              n_vertex_list(atom) = n_vertex_list(atom) + 1
              vertex(n_vertex_list(atom), :) = vertex_x(:)
              vertex_info(n_vertex_list(atom), 1) = i
              vertex_info(n_vertex_list(atom), 2) = j
              vertex_info(n_vertex_list(atom), 3) = k
            end if

          end do
        end do
      end do
!      write(*, *) n_vertex_list(atom)
!      write(*, *) vertex_info(n_vertex_list(atom),:)
!        write(*, *) mianvoro
      ! calculate the area and vol of each part of the Voronoi polyhedra(cluster)
      !! start of i-iteration
      j=0
      k=0
      s=0
      n_neighbor_list(atom)=0
      area_sum = 0
      CN_test = 0

      do i = 1, possible_n_neighbor
        if (bool(i).eq.1) then
          CN_test = CN_test + 1
!          write(*,*) CN_test
          k = 0
          do l = 1, n_vertex_list(atom)
            if ((vertex_info(l, 1).eq.i) .or. (vertex_info(l, 2).eq.i) .or. (vertex_info(l, 3).eq.i)) then
              k = k + 1
              mianvoro(i, k) = l        ! the edge number of atom-i is k

!              write(*,*) 'k', k
!              write(*,*) 'mianvoro', mianvoro(i, k)
              exit
            end if
          end do

          j = l
          l = 1
!          write(*,*) 'j', j
!            write(*,*) 'k', k
!            write(*,*) 'vertex_info is ', vertex_info
          do while(l <= n_vertex_list(atom))

            if (((vertex_info(l,1).eq.i &
              .and.((vertex_info(l,2).eq.vertex_info(j,1)) &
              .or.(vertex_info(l,2).eq.vertex_info(j,2)) &
              .or.(vertex_info(l,2).eq.vertex_info(j,3)) &
              .or.(vertex_info(l,3).eq.vertex_info(j,1)) &
              .or.(vertex_info(l,3).eq.vertex_info(j,2)) &
              .or.(vertex_info(l,3).eq.vertex_info(j,3)))) &
             .or.(vertex_info(l,2).eq.i &
              .and.((vertex_info(l,1).eq.vertex_info(j,1)) &
              .or.(vertex_info(l,1).eq.vertex_info(j,2)) &
              .or.(vertex_info(l,1).eq.vertex_info(j,3)) &
              .or.(vertex_info(l,3).eq.vertex_info(j,1)) &
              .or.(vertex_info(l,3).eq.vertex_info(j,2)) &
              .or.(vertex_info(l,3).eq.vertex_info(j,3)))) &
             .or.(vertex_info(l,3).eq.i &
              .and.((vertex_info(l,1).eq.vertex_info(j,1)) &
              .or.(vertex_info(l,1).eq.vertex_info(j,2)) &
              .or.(vertex_info(l,1).eq.vertex_info(j,3)) &
              .or.(vertex_info(l,2).eq.vertex_info(j,1)) &
              .or.(vertex_info(l,2).eq.vertex_info(j,2)) &
              .or.(vertex_info(l,2).eq.vertex_info(j,3))))) &
             .and.(l.ne.mianvoro(i, 1)) &
             .and.(l.ne.mianvoro(i, k + 1*merge(-1, 0, k.ne.1))) &
             .and.(l.ne.j)) then
!                write(*, *) 'l', l
              k = k + 1
!              if(k.gt.(n_atoms)) then
              if(k.gt.(allow_neighbor_limit)) then

                exit
              end if
              mianvoro(i, k) = l
!              write(*, *) 'goto 200'
              j = l
              l = 0
            end if
            l = l + 1
          end do
          vertex_face(i)=k

          neighbor_vol_list_with_small(i) = 0
          do l = 2, k - 1
            face_2(1, :) = vertex(mianvoro(i, 1), :)
            do m = 2, 3
              face_2(m, :)=vertex(mianvoro(i, l + m-2), :)
            end do

            call determinant(face_2(1:3, 1:3), deter_value)
!            deter_value = determinant(face_2(1:3, 1:3))
            neighbor_vol_list_with_small(i) = neighbor_vol_list_with_small(i) + abs(deter_value/6.0)
!            write(*, *) 'neighbor_vol_list_with_small', neighbor_vol_list_with_small(i)

          end do
!          write(*, *) 'here 24'

          call distance_info(atom_coords(atom, :), atom_coords(possible_neighbor_list(i), :), Bds, pbc, r, d)
          neighbor_area_list_with_small(i) = 6.0 * neighbor_vol_list_with_small(i)/d
          area_sum = area_sum + neighbor_area_list_with_small(i)
        end if
      !!! missing one end if???????
      end do
!      write(*, *) 'here 3'

      if (small_face_thres > 0.0) then
        s = 0
        do i = 1, possible_n_neighbor
          if (bool(i).eq.1) then
            if (neighbor_area_list_with_small(i) .lt. small_face_thres * area_sum / CN_test) then  !5% percent
              bool(i) = 0
              CN_test = CN_test - 1
            else
!              write(*,*) "atom is : ", atom
!              write(*,*) n_neighbor_list(atom)

              s = s + 1;
              n_neighbor_list(atom) = n_neighbor_list(atom) + 1
              neighbor_lists(atom, s) = possible_neighbor_list(i)
              neighbor_area_lists(atom, s) = neighbor_area_list_with_small(i)
              neighbor_vol_lists(atom, s) = neighbor_vol_list_with_small(i)
              neighbor_edge_lists(atom, s) = vertex_face(i)
              call distance_info(atom_coords(atom, :), atom_coords(possible_neighbor_list(i), :), Bds, pbc, r, d)
              neighbor_distance_lists(atom, s) = d
            end if
          end if
        end do
!        write(*,*) "atom is : ", atom
!        write(*,*) "__________________________"
!        write(*,*) n_neighbor_list(atom)
!        write(*,*) "__________________________"
!        write(*,*) neighbor_lists(atom, :)
!        write(*,*) "__________________________"
!        write(*,*) neighbor_area_lists(atom, :)
!        write(*,*) "__________________________"
!        write(*,*) neighbor_vol_lists(atom, :)
!        write(*,*) "__________________________"
!        write(*,*) neighbor_edge_lists(atom, :)
!        write(*,*) "__________________________"
      end if

!      write(*, *) 'here 4'

      if (n_neighbor_list(atom) > n_neighbor_max) then
        n_neighbor_max = n_neighbor_list(atom)
      end if

      do m = 1, n_neighbor_list(atom)
        if (neighbor_edge_lists(atom, m) > n_neighbor_max) then
          n_edge_max = neighbor_edge_lists(atom, m)
        end if
      end do
!      write(*, *) 'here 5'

    end do
end subroutine voronoi


!end module voronoi_nn
