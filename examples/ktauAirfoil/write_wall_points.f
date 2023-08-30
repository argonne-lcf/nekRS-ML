c-----------------------------------------------------------------------
      subroutine write_wall_points(bcint)
c     This print types of current BC, counting as well
      implicit none
      include 'SIZE'
      include 'TOTAL'

      INTEGER iel, ifc, inid
      INTEGER ix, iy, iz
      INTEGER kx1, kx2
      INTEGER ky1, ky2
      INTEGER kz1, kz2
      REAL    x, y, z
      INTEGER bcint
      CHARACTER filename*200
      CHARACTER boundary_type*3


      if(bcint.eq.1)then
        boundary_type='v  '
        filename="inlet_boundary_points.txt"
      endif
      if(bcint.eq.2)then
        boundary_type='o  '
        filename="outlet_boundary_points.txt"
      endif
      if(bcint.eq.3)then
        boundary_type='W  '
        filename="wall_boundary_points.txt"
      endif

      do inid=0,np-1
        if(inid.eq.nid)then
          if (inid.eq.0) then
            open(unit=29,file=trim(filename),status='REPLACE')
          else
            open(unit=29,file=trim(filename),status='OLD',
     &           position='append')
          endif
          do iel=1,nelv
            do ifc=1,2*ldim
              if (cbc(ifc,iel,1).eq.boundary_type) then
                call facind(KX1,KX2,KY1,KY2,KZ1,KZ2,lx1,ly1,lz1,ifc)
                do iz=kz1,kz2
                  do iy=ky1,ky2
                    do ix=kx1,kx2
                      x = xm1(ix,iy,iz,iel)
                      y = ym1(ix,iy,iz,iel)
                      z = zm1(ix,iy,iz,iel)
                      if(z.le.0.0)then
                         write(29,*)nid,lglel(iel),ix,iy,iz,x,y,z
                      endif
                    enddo
                  enddo
                enddo
              endif
            enddo
          enddo
          close(29)
        endif
        call nekgsync()
      enddo

      return
      end
c-----------------------------------------------------------------------
