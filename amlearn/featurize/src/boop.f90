
    subroutine calculate_boop(n_atoms, atom_coords, pbc, bds, &
        n_neighbor_limit, n_neighbor_list, neighbor_lists, &
        low_order, higher_order, coarse_lower_order, coarse_higher_order, &
        Ql, Wlbar, coarse_Ql, coarse_Wlbar)

        use :: angle
!        use :: CG_function

        integer :: n_atoms, n_neighbor_limit
        integer, dimension(n_atoms):: n_neighbor_list
        REAL(8), dimension(n_atoms, 3):: atom_coords
        integer, dimension(3) :: pbc
        REAL(8), dimension(3, 2) :: bds
        integer, dimension(n_atoms, n_neighbor_limit):: neighbor_lists
        integer :: low_order, higher_order, coarse_lower_order, coarse_higher_order
        REAL(8), dimension(n_atoms, 4):: Ql, Wlbar, coarse_Ql, coarse_Wlbar
        REAL(16), dimension(n_atoms, 4):: Ql2, Wlbar2, coarse_Ql2, coarse_Wlbar2

!f2py   intent(in) n_atoms, atom_coords, pbc, bds
!f2py   intent(in) n_neighbor_limit, n_neighbor_list, neighbor_lists
!f2py   intent(in) low_order, higher_order, coarse_lower_order, coarse_higher_order
!f2py   intent(in, out) Ql, Wlbar, coarse_Ql, coarse_Wlbar

        integer :: atom, i, j, l, m, m1, m2, m3, neighs
        REAL(8) :: ksai, theta
        COMPLEX(16) ::  qtemp, comp
        COMPLEX(16), allocatable:: Q2(:,:,:), Wl(:,:)
        COMPLEX(16), allocatable:: QQ2(:,:,:), WWl(:,:)
        INTEGER err_mesg

        REAL(16), dimension(2:5, -10:10, -10:10, -10:10) :: CG


        integer :: v, s
        REAL(16) :: v0, v1, v2
        REAL(16), dimension(0:100) :: fac

      !!!!!!!!!!!!!!!!!!! calculate Ql !!!!!!!!!!!!!!!!!!!!!
        allocate(Q2(n_atoms, 2:5, -10:10), stat = err_mesg)
        Q2 = 0

        comp=(0.,1.)

        do atom = 1, n_atoms
          do i = 1, n_neighbor_list(atom)
            j = neighbor_lists(atom, i)
            call angle_info(atom_coords(atom, :), atom_coords(j, :), bds, pbc, ksai, theta)
            do l = 4, 10, 2
              do m = -l, l
                  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!Q4!!!!!!!!!!!!!!!!!!!!!!!!!!!
                  if((l == 4).and.(m == (-4))) then
                    qtemp = 3.0/16.0 * sqrt(35.0/(2.0*3.1415926)) * exp(-4*comp*ksai) * &
                            (sin(theta))**4

                  else if((l == 4).and.(m == (-3))) then
                    qtemp = 3.0/8.0 * sqrt(35.0/(3.1415926)) * exp(-3*comp*ksai) * &
                            (sin(theta))**3 * cos(theta)

                  else if((l == 4).and.(m == (-2))) then
                    qtemp = 3.0/8.0 * sqrt(5.0/(2.0*3.1415926)) * exp(-2*comp*ksai) * &
                            (sin(theta))**2 * (7.0*(cos(theta))**2 - 1.0)

                  else if((l == 4).and.(m == (-1))) then
                    qtemp = 3.0/8.0 * sqrt(5.0/(3.1415926))* exp(-1*comp*ksai) * &
                            (sin(theta)) * (7.0*(cos(theta))**3-3.0*cos(theta))

                  else if((l == 4).and.(m == (0))) then
                    qtemp = 3.0/16.0 * sqrt(1.0/(3.1415926))* &
                            (35.0*(cos(theta))**4 - 30.0*(cos(theta))**2+3)


                  else if((l == 4).and.(m == 1)) then
                    qtemp = -3.0/8.0 * sqrt(5.0/(3.1415926))* exp(1*comp*ksai) * &
                            (sin(theta)) * (7.0*(cos(theta))**3-3.0*cos(theta))

                  else if((l == 4).and.(m == 2)) then
                    qtemp = 3.0/8.0 * sqrt(5.0/(2.0*3.1415926)) * exp(2*comp*ksai)* &
                            (sin(theta))**2* (7.0*(cos(theta))**2-1.0)

                  else if((l == 4).and.(m == 3)) then
                    qtemp = -3.0/8.0 * sqrt(35.0/(3.1415926)) * exp(3*comp*ksai)* &
                            (sin(theta))**3*cos(theta)

                  else if((l == 4).and.(m == 4)) then
                    qtemp = 3.0/16.0 * sqrt(35.0/(2.0*3.1415926)) * exp(4*comp*ksai)* &
                            (sin(theta))**4

                  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!Q6!!!!!!!!!!!!!!!!!!!!!!!!!!!

                  else if((l == 6).and.(m == (-6))) then
                    qtemp = 1.0/64.0 * sqrt(3003.0/(3.1415926)) * exp(-6*comp*ksai)* &
                            (sin(theta))**6

                  else if((l == 6).and.(m == (-5))) then
                    qtemp = 3.0/32.0 * sqrt(1001.0/(3.1415926)) * exp(-5*comp*ksai)* &
                            (sin(theta))**5*cos(theta)

                  else if((l == 6).and.(m == (-4))) then
                    qtemp = 3.0/32.0 * sqrt(91.0/(2.0*3.1415926)) * exp(-4*comp*ksai)* &
                            (sin(theta))**4* (11.0*(cos(theta))**2-1)

                  else if((l == 6).and.(m == (-3))) then
                    qtemp = 1.0/32.0 * sqrt(1365.0/(3.1415926)) * exp(-3*comp*ksai)* &
                            (sin(theta))**3* (11.0*(cos(theta))**3-3.0*cos(theta))

                  else if((l == 6).and.(m == (-2))) then
                    qtemp = 1.0/64.0 * sqrt(1365.0/(3.1415926)) * exp(-2*comp*ksai)* &
                            (sin(theta))**2 * (33.0*(cos(theta))**4 - 18.0*(cos(theta))**2 + 1)

                  else if((l == 6).and.(m == (-1))) then
                    qtemp = 1.0/16.0 * sqrt(273.0/(2*3.1415926)) * exp(-1*comp*ksai)* &
                            (sin(theta))* (33.0*(cos(theta))**5 - 30.0*(cos(theta))**3 + &
                            5.0*cos(theta))

                  !! no exp(-1*comp*ksai)?
                  else if((l == 6).and.(m == (0))) then
                    qtemp = 1.0/32.0 * sqrt(13.0/(3.1415926))* (231.0*(cos(theta))**6 - &
                            315.0*(cos(theta))**4 + 105.0*(cos(theta))**2 - 5)

                  else if((l == 6).and.(m == (1))) then
                    qtemp = -1.0/16.0 * sqrt(273.0/(2*3.1415926)) * exp(1*comp*ksai)* &
                            (sin(theta)) * (33.0*(cos(theta))**5-30.0*(cos(theta))**3 + &
                            5.0*cos(theta))

                  else if((l == 6).and.(m == (2))) then
                    qtemp = 1.0/64.0 * sqrt(1365.0/(3.1415926))* exp(2*comp*ksai)* &
                            (sin(theta))**2 * (33.0*(cos(theta))**4 - 18.0*(cos(theta))**2 + 1)

                  else if((l == 6).and.(m == (3))) then
                    qtemp = -1.0/32.0 * sqrt(1365.0/(3.1415926)) * exp(3*comp*ksai)* &
                            (sin(theta))**3 * (11.0*(cos(theta))**3 - 3.0*cos(theta))

                  else if((l == 6).and.(m == (4))) then
                    qtemp = 3.0/32.0 * sqrt(91.0/(2.0*3.1415926)) * exp(4*comp*ksai)* &
                            (sin(theta))**4 * (11.0*(cos(theta))**2-1)

                  else if((l == 6).and.(m == (5))) then
                    qtemp = -3.0/32.0 * sqrt(1001.0/(3.1415926)) * exp(5*comp*ksai)* &
                            (sin(theta))**5 * cos(theta)


                  else if((l == 6).and.(m == (6))) then
                    qtemp = 1.0/64.0 * sqrt(3003.0/(3.1415926)) * exp(6*comp*ksai)* &
                            (sin(theta))**6

                  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!Q8!!!!!!!!!!!!!!!!!!!!!!!!!!!

                  else if((l == 8).and.(m == (-8))) then
                    qtemp = 3.0/256.0 * sqrt(12155.0/(2*3.1415926)) * exp(-8*comp*ksai)* &
                            (sin(theta))**8

                  else if((l == 8).and.(m == (-7))) then
                    qtemp = 3.0/64.0 * sqrt(12155.0/(2*3.1415926)) * exp(-7*comp*ksai)* &
                            (sin(theta))**7 * cos(theta)

                  else if((l == 8).and.(m == (-6))) then
                    qtemp = 1.0/128.0 * sqrt(7293.0/(3.1415926)) * exp(-6.0*comp*ksai)* &
                            (sin(theta))**6 * (15.0*(cos(theta))**2 - 1)

                  else if((l == 8).and.(m == (-5))) then
                    qtemp = 3.0/64.0 * sqrt(17017.0/(2*3.1415926)) * exp(-5.0*comp*ksai)* &
                            (sin(theta))**5 * (5.0*(cos(theta))**3 - 1.0*cos(theta))

                  else if((l == 8).and.(m == (-4))) then
                    qtemp = 3.0/128.0 * sqrt(1309.0/(2*3.1415926)) * exp(-4.0*comp*ksai)* &
                            (sin(theta))**4 * (65.0*(cos(theta))**4 - 26.0*(cos(theta))**2 + 1)

                  else if((l == 8).and.(m == (-3))) then
                    qtemp = 1.0/64.0 * sqrt(19635.0/(2*3.1415926)) * exp(-3.0*comp*ksai)* &
                            (sin(theta))**3 * (39.0*(cos(theta))**5 - 26.0*(cos(theta))**3 + &
                            3.0*cos(theta))

                  else if((l == 8).and.(m == (-2))) then
                    qtemp = 3.0/128.0 * sqrt(595.0/(3.1415926)) * exp(-2.0*comp*ksai)* &
                            (sin(theta))**2 * (143.0*(cos(theta))**6 - 143.0*(cos(theta))**4 + &
                            33.0*(cos(theta))**2 - 1)

                  else if((l == 8).and.(m == (-1))) then
                    qtemp = 3.0/64.0 * sqrt(17.0/(2.0*3.1415926)) * exp(-1.0*comp*ksai)* &
                            (sin(theta))**1 * (715.0*(cos(theta))**7-1001.0*(cos(theta))**5 + &
                            385.0*(cos(theta))**3 - 35.0*cos(theta))

                  else if((l == 8).and.(m == (0))) then
                    qtemp = 1.0/256.0 * sqrt(17.0/(3.1415926)) * (6435.0*(cos(theta))**8 - &
                            12012.0*(cos(theta))**6 + 6930.0*(cos(theta))**4 - &
                            1260.0*(cos(theta))**2 + 35.0)

                  else if((l == 8).and.(m == (1))) then
                    qtemp = -3.0/64.0 * sqrt(17.0/(2.0*3.1415926)) * exp(1.0*comp*ksai)* &
                            (sin(theta))**1 * (715.0*(cos(theta))**7 - 1001.0*(cos(theta))**5 + &
                            385.0*(cos(theta))**3 - 35.0*cos(theta))

                  else if((l == 8).and.(m == (2))) then
                    qtemp = 3.0/128.0 * sqrt(595.0/(3.1415926)) * exp(2.0*comp*ksai)* &
                            (sin(theta))**2 * (143.0*(cos(theta))**6 - 143.0*(cos(theta))**4 + &
                            33.0*(cos(theta))**2-1)

                  else if((l == 8).and.(m == (3))) then
                    qtemp = -1.0/64.0 * sqrt(19635.0/(2.0*3.1415926)) * exp(3.0*comp*ksai)* &
                            (sin(theta))**3 * (39.0*(cos(theta))**5-26.0*(cos(theta))**3 + &
                            3.0*cos(theta))

                  else if((l == 8).and.(m == (4))) then
                    qtemp = 3.0/128.0 * sqrt(1309.0/(2.0*3.1415926)) * exp(4.0*comp*ksai)* &
                            (sin(theta))**4* (65.0*(cos(theta))**4 - 26.0*(cos(theta))**2 + 1)

                  else if((l == 8).and.(m == (5))) then
                    qtemp = -3.0/64.0 * sqrt(17017.0/(2.0*3.1415926)) * exp(5.0*comp*ksai)* &
                            (sin(theta))**5* (5.0*(cos(theta))**3 - 1.0*cos(theta))

                  else if((l == 8).and.(m == (6))) then
                    qtemp = 1.0/128.0 * sqrt(7293.0/(3.1415926)) * exp(6.0*comp*ksai)* &
                            (sin(theta))**6 * (15.0*(cos(theta))**2-1)

                  else if((l == 8).and.(m == (7))) then
                    qtemp = -3.0/64.0 * sqrt(12155.0/(2.0*3.1415926)) * exp(7.0*comp*ksai)* &
                            (sin(theta))**7 * cos(theta)

                  else if((l == 8).and.(m == (8))) then
                    qtemp = 3.0/256.0 * sqrt(12155.0/(2.0*3.1415926)) * exp(8.0*comp*ksai)* &
                            (sin(theta))**8

                  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!Q10!!!!!!!!!!!!!!!!!!!!!!!!!!!

                  else if((l == 10).and.(m == (-10))) then
                    qtemp = 1.0/1024.0 * sqrt(969969.0/(3.1415926)) * exp(-10.0*comp*ksai)* &
                            (sin(theta))**10

                  else if((l == 10).and.(m == (-9))) then
                    qtemp = 1.0/512.0 * sqrt(4849845.0/(3.1415926)) * exp(-9.0*comp*ksai)* &
                            (sin(theta))**9 * cos(theta)

                  else if((l == 10).and.(m == (-8))) then
                    qtemp = 1.0/512.0 * sqrt(255255.0/(2.0*3.1415926)) * exp(-8.0*comp*ksai)* &
                            (sin(theta))**8 * (19.0*(cos(theta))**2-1)

                  else if((l == 10).and.(m == (-7))) then
                    qtemp = 3.0/512.0 * sqrt(85085.0/(3.1415926)) * exp(-7.0*comp*ksai)* &
                            (sin(theta))**7 * (19.0*(cos(theta))**3-3.0*cos(theta))

                  else if((l == 10).and.(m == (-6))) then
                    qtemp = 3.0/1024.0 * sqrt(5005.0/(3.1415926)) * exp(-6.0*comp*ksai)* &
                            (sin(theta))**6 * (323.0*(cos(theta))**4-102.0*(cos(theta))**2 + 3)

                  else if((l == 10).and.(m == (-5))) then
                    qtemp = 3.0/256.0 * sqrt(1001.0/(3.1415926)) * exp(-5.0*comp*ksai)* &
                            (sin(theta))**5 * (323.0*(cos(theta))**5 - 170.0*(cos(theta))**3 + &
                            15.0*cos(theta))

                  else if((l == 10).and.(m == (-4))) then
                    qtemp = 3.0/256.0 * sqrt(5005.0/(2.0*3.1415926)) * exp(-4.0*comp*ksai)* &
                            (sin(theta))**4 * (323.0*(cos(theta))**6 - 255.0*(cos(theta))**4 + &
                            45.0*(cos(theta))**2-1)

                  else if((l == 10).and.(m == (-3))) then
                    qtemp = 3.0/256.0 * sqrt(5005.0/(3.1415926)) * exp(-3.0*comp*ksai)* &
                            (sin(theta))**3 * (323.0*(cos(theta))**7 - 357.0*(cos(theta))**5 + &
                            105.0*(cos(theta))**3-7.0*cos(theta))

                  else if((l == 10).and.(m == (-2))) then
                    qtemp = 3.0/512.0 * sqrt(385.0/(2.0*3.1415926)) * exp(-2.0*comp*ksai)* &
                            (sin(theta))**2 * (4199.0*(cos(theta))**8 - 6188.0*(cos(theta))**6 + &
                            2730.0*(cos(theta))**4 - 364.0*(cos(theta))**2 + 7)

                  else if((l == 10).and.(m == (-1))) then
                    qtemp = 1.0/256.0 * sqrt(1155.0/(2.0*3.1415926)) * exp(-1.0*comp*ksai)* &
                            (sin(theta))**1 * (4199.0*(cos(theta))**9 - 7956.0*(cos(theta))**7 + &
                            4914.0*(cos(theta))**5 - 1092.0*(cos(theta))**3 + 63.0*cos(theta))

                  else if((l == 10).and.(m == (0))) then
                    qtemp = 1.0/512.0 * sqrt(21.0/(3.1415926)) * (46189.0*(cos(theta))**10 - &
                            109395.0*(cos(theta))**8 + 90090.0*(cos(theta))**6 - 30030.0*(cos(theta))**4 + &
                            3465.0*(cos(theta))**2-63)

                  else if((l == 10).and.(m == (1))) then
                    qtemp = -1.0/256.0 * sqrt(1155.0/(2.0*3.1415926)) * exp(1.0*comp*ksai)* &
                            (sin(theta))**1 * (4199.0*(cos(theta))**9 - 7956.0*(cos(theta))**7 + &
                            4914.0*(cos(theta))**5 - 1092.0*(cos(theta))**3 + 63.0*cos(theta))

                  else if((l == 10).and.(m == (2))) then
                    qtemp = 3.0/512.0 * sqrt(385.0/(2.0*3.1415926)) * exp(2.0*comp*ksai)* &
                            (sin(theta))**2 * (4199.0*(cos(theta))**8 - 6188.0*(cos(theta))**6 + &
                            2730.0*(cos(theta))**4 - 364.0*(cos(theta))**2 + 7)

                  else if((l == 10).and.(m == (3))) then
                    qtemp = -3.0/256.0 * sqrt(5005.0/(3.1415926)) * exp(3.0*comp*ksai)* &
                            (sin(theta))**3 * (323.0*(cos(theta))**7 - 357.0*(cos(theta))**5 + &
                            105.0*(cos(theta))**3 - 7.0*cos(theta))

                  else if((l == 10).and.(m == (4))) then
                    qtemp = 3.0/256.0 * sqrt(5005.0/(2.0*3.1415926)) * exp(4.0*comp*ksai)* &
                            (sin(theta))**4 * (323.0*(cos(theta))**6 - 255.0*(cos(theta))**4 + &
                            45.0*(cos(theta))**2 - 1)

                  else if((l == 10).and.(m == (5))) then
                    qtemp = -3.0/256.0 * sqrt(1001.0/(3.1415926)) * exp(5.0*comp*ksai)* &
                            (sin(theta))**5 * (323.0*(cos(theta))**5 - 170.0*(cos(theta))**3 + &
                            15.0*cos(theta))

                  else if((l == 10).and.(m == (6))) then
                    qtemp = 3.0/1024.0 * sqrt(5005.0/(3.1415926)) * exp(6.0*comp*ksai)* &
                            (sin(theta))**6 * (323.0*(cos(theta))**4 - 102.0*(cos(theta))**2 + 3)

                  else if((l == 10).and.(m == (7))) then
                    qtemp = -3.0/512.0 * sqrt(85085.0/(3.1415926)) * exp(7.0*comp*ksai)* &
                            (sin(theta))**7 * (19.0*(cos(theta))**3 - 3.0*cos(theta))

                  else if((l == 10).and.(m == (8))) then
                    qtemp = 1.0/512.0 * sqrt(255255.0/(2.0*3.1415926)) * exp(8.0*comp*ksai)* &
                            (sin(theta))**8 * (19.0*(cos(theta))**2-1)

                  else if((l == 10).and.(m == (9))) then
                    qtemp = -1.0/512.0 * sqrt(4849845.0/(3.1415926)) * exp(9.0*comp*ksai)* &
                            (sin(theta))**9 * cos(theta)

                  else if((l == 10).and.(m == (10))) then
                    qtemp = 1.0/1024.0 * sqrt(969969.0/(3.1415926)) * exp(10.0*comp*ksai)* &
                            (sin(theta))**10

                  end if

                Q2(atom, l/2, m) = Q2(atom, l/2, m) + qtemp

              end do
            end do
	      end do

	      do l = 4, 10, 2
	        do m = -l, l
		      Q2(atom, l/2, m) = Q2(atom, l/2, m) / (1.0*n_neighbor_list(atom))
	          Ql2(atom, l/2-1) = Ql2(atom, l/2-1) + (abs(Q2(atom, l/2, m)))**2
	          Ql(atom, l/2-1) = Ql2(atom, l/2-1)
	        end do
	        Ql2(atom, l/2-1) = sqrt(Ql2(atom, l/2-1) * 4.0 * 3.1415926 / (1.0 * (2.0*l + 1.0)))
            Ql(atom, l/2-1) = Ql2(atom, l/2-1)
!              write(*,*) "calc ql", Ql(atom, l/2-1), Ql2(atom, l/2-1), Ql2(atom, l/2-1)
          end do

	    end do

      !!!!!!!!!!!!!!!!!!! calculate Wl !!!!!!!!!!!!!!!!!!!!!
      if (higher_order == 1) then

        allocate(Wl(n_atoms, 2:5), stat = err_mesg)

!        CG = calculate_CG(2, 5)


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

        do l = 4, 10, 2
        do m1 = -l, l
          do m2 = max(-l, -l-m1), min(l, l-m1)
            m3 = -(m1 + m2)
            s = 0
            v2 = 0
            do v = max(0, -m1, m2), min(l, l-m1, l+m2)
              s = s + 1
              v1 = fac(v) + fac(l-v) + fac(l-m1-v) + fac(l+m2-v) + fac(-m2+v) + fac(m1+v)
              if(s == 1) then
                v0 = v1
              end if
              v2 = v2 + (-1)**(-v) * 10**(1.0*(v0-v1))
            end do
            CG(l/2, m1, m2, m3) = 10**(1.0/2 * &
                (log10(1.0*2*l+1) + 3*fac(l)-fac(3*l+1) + &
                fac(l+m1)+fac(l-m1)+fac(l+m2)+fac(l-m2)+fac(l+m3)+fac(l-m3))-v0) * v2 * &
                (-1)**(-m3) / sqrt(1.0*(2*l+1))
!            write(*,*) CG(l/2, m1, m2, m3)
          end do
        end do
        end do

        do atom = 1, n_atoms
!            if(atom > 10) then
!                exit
!            end if

          do l = 4, 10, 2
            do m1 = -l, l
              do m2 = max(-l,-l-m1), min(l,l-m1)
                m3 = -(m1+m2)
                Wl(atom, l/2) = Wl(atom, l/2) + CG(l/2, m1, m2, m3) * &
                    Q2(atom, l/2, m1) * Q2(atom, l/2, m2) * Q2(atom, l/2, m3)

!                write(*,*) CG(l/2, m1, m2, m3), Q2(atom, l/2, m1), Q2(atom, l/2, m2), Q2(atom, l/2, m3)

              end do
            end do
            Wlbar2(atom, l/2-1) = Wl(atom, l/2) / (Ql2(atom, l/2-1)/sqrt(4.0*3.1415926/(1.0*(2*l+1))))**3

            Wlbar(atom, l/2-1) = Wlbar2(atom, l/2-1)
!            write(*,*) Ql(atom, l/2-1), Ql2(atom, l/2-1)
!            write(*,*) Wl(atom, l/2)
!            write(*,*) Wlbar(atom, l/2-1), Wlbar2(atom, l/2-1)

          end do
        end do

      end if

      !!!!!!!!!!!!!!!!!!! calculate coarse_Ql !!!!!!!!!!!!!!!!!!!!!
      if ((coarse_lower_order == 1) .OR. (coarse_higher_order == 1)) then

        allocate(QQ2(n_atoms, 2:5, -10:10), stat = err_mesg)

        do atom = 1, n_atoms
          do l = 4, 10, 2
            do m = -l, l
              QQ2(atom, l/2, m) = QQ2(atom, l/2, m) + Q2(atom, l/2, m)
              do i = 1, n_neighbor_list(atom)
	            j = neighbor_lists(atom, i)
		        QQ2(atom, l/2, m) = QQ2(atom, l/2, m) + Q2(j, l/2,m)
		      end do
		      QQ2(atom, l/2, m) = QQ2(atom, l/2, m) / (1.0*n_neighbor_list(atom)+1.0)
	          coarse_Ql2(atom, l/2-1) = coarse_Ql2(atom, l/2-1) + (abs(QQ2(atom, l/2, m)))**2
	          coarse_Ql(atom, l/2-1) = coarse_Ql2(atom, l/2-1)
	        end do
	        coarse_Ql2(atom, l/2-1)= sqrt(coarse_Ql2(atom, l/2-1)*4.0*3.1415926/(1.0*(2.0*l+1.0)))
	        coarse_Ql(atom, l/2-1)= coarse_Ql2(atom, l/2-1)
	      end do
	    end do

	  end if

        
      !!!!!!!!!!!!!!!!!!! calculate coarse_Wlbar !!!!!!!!!!!!!!!!!!!!!
      if (coarse_higher_order == 1) then

        allocate(WWl(n_atoms, 2:5), stat = err_mesg)

!        CG = calculate_CG(2, 5)
        fac = 0
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


        do l = 4, 10, 2
        do m1 = -l, l
          do m2 = max(-l, -l-m1), min(l, l-m1)
            m3 = -(m1 + m2)
            s = 0
            v2 = 0
            do v = max(0, -m1, m2), min(l, l-m1, l+m2)
              s = s + 1
              v1 = fac(v) + fac(l-v) + fac(l-m1-v) + fac(l+m2-v) + fac(-m2+v) + fac(m1+v)
              if(s == 1) then
                v0 = v1
              end if
              v2 = v2 + (-1)**(-v) * 10**(1.0*(v0-v1))
            end do
            CG(l/2, m1, m2, m3) = 10**(1.0/2 * &
                (log10(1.0*2*l+1) + 3*fac(l)-fac(3*l+1) + &
                fac(l+m1)+fac(l-m1)+fac(l+m2)+fac(l-m2)+fac(l+m3)+fac(l-m3))-v0) * v2 * &
                (-1)**(-m3) / sqrt(1.0*(2*l+1))
!            write(*,*) CG(l/2, m1, m2, m3)
          end do
        end do
        end do

        do atom = 1, n_atoms
          do l = 4, 10, 2
            do m1 = -l, l
              do m2 = max(-l,-l-m1), min(l,l-m1)
                m3 = -(m1+m2)
                WWl(atom, l/2) = WWl(atom, l/2) + CG(l/2, m1, m2, m3) * &
                    QQ2(atom, l/2, m1) * QQ2(atom, l/2, m2) * QQ2(atom, l/2, m3)
              end do
            end do
           coarse_Wlbar2(atom, l/2-1) = WWl(atom, l/2) / (coarse_Ql2(atom, l/2-1)/sqrt(4.0*3.1415926/(1.0*(2*l+1))))**3
           coarse_Wlbar(atom, l/2-1) = coarse_Wlbar2(atom, l/2-1)
          end do
        end do

      end if

      deallocate(Q2, Wl)
      deallocate(QQ2, WWl)

    end subroutine calculate_boop

!
!    function calculate_CG(half_l_min, half_l_max) result(CG)
!    integer, intent(in) :: half_l_min, half_l_max
!    integer:: l_min, l_max
!    REAL(16), dimension(half_l_min:half_l_max, -half_l_max*2:half_l_max*2, &
!      -half_l_max*2:half_l_max*2, -half_l_max*2:half_l_max*2):: CG
!
!    integer :: i, j, l, m1, m2, m3, v, s
!    REAL(16) :: v0, v1, v2
!    REAL(16), dimension(0:100) :: fac
!
!    do i = 0, 50
!    if(i == 0) then
!      fac(i)=0
!    else
!      do j = 1, i
!        fac(i) = fac(i) + log10(1.0*j)
!      end do
!    end if
!    end do
!
!
!    CG = 0
!
!    do l = 4, 10, 2
!    do m1 = -l, l
!      do m2 = max(-l, -l-m1), min(l, l-m1)
!        m3 = -(m1 + m2)
!        s = 0
!        v2 = 0
!        do v = max(0, -m1, m2), min(l, l-m1, l+m2)
!          s = s + 1
!          v1 = fac(v) + fac(l-v) + fac(l-m1-v) + fac(l+m2-v) + fac(-m2+v) + fac(m1+v)
!          if(s == 1) then
!            v0 = v1
!          end if
!          v2 = v2 + (-1)**(-v) * 10**(1.0*(v0-v1))
!        end do
!        CG(l/2, m1, m2, m3) = 10**(1.0/2 * &
!            (log10(1.0*2*l+1) + 3*fac(l)-fac(3*l+1) + &
!            fac(l+m1)+fac(l-m1)+fac(l+m2)+fac(l-m2)+fac(l+m3)+fac(l-m3))-v0) * v2 * &
!            (-1)**(-m3) / sqrt(1.0*(2*l+1))
!        write(*,*) CG(l/2, m1, m2, m3)
!      end do
!    end do
!    end do
!
!    end function calculate_CG