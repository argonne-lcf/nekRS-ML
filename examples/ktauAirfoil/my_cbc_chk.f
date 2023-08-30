c-----------------------------------------------------------------------
      subroutine my_cbc_chk(s3)
c     This print types of current BC, counting as well
      implicit none
      include 'SIZE'
      include 'TOTAL'

      integer ncbc,lcbc,i,j,e,f,inid,mtype,idummy,ncbc_tmp,ifld,nel
     $      , nfldt
      parameter (lcbc=10) ! max # unique CBC
      integer cbc_count(lcbc),cbc_count_tmp(lcbc)
      character*3 cbc_list(lcbc),cbc_list_tmp(lcbc),cbc3,s3,tag3
      character*2 s2tmp
      logical cbc_in_list

      nfldt = nfield
      if (ifmhd) nfldt = ifldmhd   ! mhd always uses nscal+3, with or without temperature

      do ifld=0,nfldt

        if (ifld.eq.0) then
          tag3='PR '
          nel = nelv
        elseif (ifld.eq.1) then
          tag3='VEL'
          nel = nelv
        elseif (ifmhd.AND.ifld.eq.ifldmhd) then
          tag3='MHD'
          nel = nelv
        elseif (ifld.gt.1) then
          write(s2tmp,'(I2.2)') ifld-2
          tag3='S'//s2tmp
          nel = nelt
        endif

        call izero(cbc_count,lcbc)

        ncbc = 1                 ! # unique CBC
        cbc_list(1) = 'E  ' ! internal BC

        do e=1,nel
        do f=1,2*ldim
          cbc3 = cbc(f,e,ifld)
          cbc_in_list = .false.

          if (cbc3.eq.'E  '.or.cbc3.eq.'   ') then
            cbc_count(1) = cbc_count(1) + 1
            cbc_in_list = .true.
          else
            do i=2,ncbc ! go throught the registered CBC
              if (cbc3.eq.cbc_list(i)) then
                cbc_count(i) = cbc_count(i) + 1
                cbc_in_list = .true.
              endif
            enddo
          endif
          if (.not.cbc_in_list) then
            ncbc = ncbc + 1
            if (ncbc.gt.lcbc)
     $        call exitti('BCchk: Increase lcbc$',ncbc)

            cbc_list(ncbc) = cbc3
            cbc_count(ncbc) = 1
          endif
        enddo
        enddo

        call nekgsync()

        ! All reduce to nid=0
        if (nid.eq.0) then
          do inid=1,np-1
            mtype = inid
            call csend(mtype,idummy,4,inid,0) ! handshake

            call crecv(mtype,ncbc_tmp,4)
            call crecv(mtype,cbc_list_tmp(1),3*ncbc_tmp,0,0)
            call crecv(mtype,cbc_count_tmp(1),4*ncbc_tmp,0,0)

            cbc_count(1) = cbc_count(1)+cbc_count_tmp(1)
            do j=2,ncbc_tmp
              cbc_in_list = .false.
              do i=2,ncbc
                if (cbc_list(i).eq.cbc_list_tmp(j)) then
                  cbc_count(i) = cbc_count(i) + cbc_count_tmp(j)
                  cbc_in_list = .true.
                endif
              enddo
              if (.not.cbc_in_list) then
                ncbc = ncbc + 1
                if (ncbc.gt.lcbc)
     $            call exitti('BCchk: Increase lcbc$',ncbc)

                cbc_list(ncbc) = cbc_list_tmp(j)
                cbc_count(ncbc) = cbc_count_tmp(j)
              endif
            enddo
          enddo
        else
          mtype = nid
          call crecv(mtype,idummy,4) ! ! handshake

          call csend(mtype,ncbc,4,0,0)
          call csend(mtype,cbc_list,3*ncbc,0,0)
          call csend(mtype,cbc_count,4*ncbc,0,0)
        endif

        call nekgsync()

        ! print
        if (nid.eq.0) then
          do i=1,ncbc
            write(6,41) s3,ifld,tag3,cbc_list(i),cbc_count(i)
          enddo
        endif

      enddo ! ifld

   41 format(a3,' BC: 'i3,' ',a3,a5,i10)

      return
      end
c-----------------------------------------------------------------------
