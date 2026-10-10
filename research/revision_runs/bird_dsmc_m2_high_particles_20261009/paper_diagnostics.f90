module paper_diagnostics
  use iso_fortran_env, only: real64
  use, intrinsic :: ieee_arithmetic
  implicit none
  integer, parameter :: ncells=800, nmom=56, ncollision_moments=35
  real(real64) :: raw_sum(nmom,ncells)=0, coll_sum(ncollision_moments,ncells)=0
  real(real64) :: stab_sum(35,ncells)=0
  integer :: powers(3,nmom), sample_count=0, coll_count=0
  integer :: ma(55)=0,inext=0,inextp=31,iff=0
  integer :: run_seed,run_warm,run_total,run_chunk,first_block=-1
  real(real64) :: run_mach,run_ppc,run_dt,velocity_unit,length_unit,time_unit,viscosity_unit,shift_sum=0
contains
  subroutine read_config
    integer :: a,b,c,d,k
    open(91,file='case.in',status='old')
    read(91,*) run_mach,run_seed,run_ppc,run_dt,run_warm,run_total,run_chunk
    close(91)
    velocity_unit=sqrt(1.380622d-23*293.d0/6.64d-26)
    viscosity_unit=0.5d0*sqrt(6.64d-26*1.380622d-23*293.d0/acos(-1.d0))/(4.092d-10)**2
    time_unit=viscosity_unit/(1.d20*1.380622d-23*293.d0)
    length_unit=time_unit*velocity_unit
    k=0
    do d=0,5
      do a=0,d
        do b=0,d-a
          c=d-a-b; k=k+1; powers(:,k)=[a,b,c]
        end do
      end do
    end do
    if(k/=nmom) stop 81
    open(91,file='units.txt',status='replace')
    write(91,*) velocity_unit,length_unit,time_unit,viscosity_unit
    close(91)
    open(91,file='powers.txt',status='replace')
    do k=1,nmom
      write(91,*) powers(:,k)
    end do
    close(91)
  end subroutine
  subroutine monomials(v,z)
    real(real64),intent(in)::v(3)
    real(real64),intent(out)::z(nmom)
    real(real64)::p(0:5,3)
    integer::a,k
    p(0,:)=1.d0; p(1,:)=v/velocity_unit
    do a=2,5
      p(a,:)=p(a-1,:)*p(1,:)
    end do
    do k=1,nmom
      z(k)=p(powers(1,k),1)*p(powers(2,k),2)*p(powers(3,k),3)
    end do
  end subroutine
  subroutine sample_paper(n,v,ip)
    integer,intent(in)::n,ip(*)
    real(real64),intent(in)::v(3,*)
    real(real64)::z(nmom)
    integer::j,cell
    sample_count=sample_count+1
    do j=1,n
      cell=(ip(j)-1)/6+1
      if(cell<1.or.cell>ncells) stop 82
      call monomials(v(:,j),z)
      raw_sum(:,cell)=raw_sum(:,cell)+z
    end do
  end subroutine
  subroutine coll_paper(cell,v,w,sgn)
    integer,intent(in)::cell,sgn
    real(real64),intent(in)::v(3),w(3)
    real(real64)::a(nmom),b(nmom)
    call monomials(v,a); call monomials(w,b)
    coll_sum(:,cell)=coll_sum(:,cell)+sgn*(a(:ncollision_moments)+b(:ncollision_moments))
    if(sgn==1) coll_count=coll_count+1
  end subroutine
  subroutine stabilization_paper(n,x,v,geom,sgn)
    integer,intent(in)::n,sgn
    real(real64),intent(in)::x(*),v(3,*),geom(3,ncells)
    real(real64)::z(nmom)
    integer::j,cell
    do j=1,n
      if(x(j)<0) then
        cell=int((x(j)-geom(1,1))/geom(3,1)+0.99999d0)
      else
        cell=400+int(x(j)/geom(3,401)+0.99999d0)
      end if
      cell=max(1,min(ncells,cell))
      call monomials(v(:,j),z)
      stab_sum(:,cell)=stab_sum(:,cell)+sgn*z(:35)
    end do
  end subroutine
  subroutine write_block(block,warm,t,n,weight,vol,geom,dt,nis,nsp)
    integer,intent(in)::block,warm,n,nis,nsp
    real(real64),intent(in)::t,weight,vol(ncells),geom(3,ncells),dt
    character(100)::fname,tmp
    integer::i
    if(.not.all(ieee_is_finite(raw_sum))) stop 83
    if(.not.all(ieee_is_finite(coll_sum))) stop 84
    write(fname,'("blocks/block_",I6.6,".dat")') block
    tmp=trim(fname)//'.tmp'
    open(91,file=trim(tmp),status='replace')
    write(91,*) block,t/time_unit,n,sample_count,coll_count,shift_sum/length_unit,merge(1,0,block>warm)
    do i=1,ncells
      write(91,'(*(ES25.16E3,1X))') (geom(1,i)+geom(2,i))/(2*length_unit), &
         raw_sum(:,i)*weight/(vol(i)*1.d20*sample_count), &
         coll_sum(:,i)*weight*time_unit/(vol(i)*1.d20*dt*nis*nsp), &
         stab_sum(:,i)*weight*time_unit/(vol(i)*1.d20*dt*nis*nsp)
    end do
    close(91)
    call rename(trim(tmp),trim(fname))
    raw_sum=0;coll_sum=0;stab_sum=0;sample_count=0;coll_count=0;shift_sum=0
    open(91,file='progress.txt.tmp',status='replace')
    write(91,*) block,t/time_unit,n,run_total
    close(91)
    call rename('progress.txt.tmp','progress.txt')
  end subroutine
  subroutine check_stop(block,total)
    integer,intent(in)::block,total
    logical::exists
    if(first_block<0) first_block=block
    if(block>=total) then
      open(91,file='FINAL_TIME_REACHED',status='replace');write(91,*) block;close(91)
      stop
    end if
    inquire(file='STOP_REQUEST',exist=exists)
    if(exists.or.block-first_block+1>=run_chunk) stop
  end subroutine
end module
