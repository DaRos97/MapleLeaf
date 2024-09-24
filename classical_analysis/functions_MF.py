import numpy as np
import cmath

def R_z(th):
    return np.array([[np.cos(th),-np.sin(th),0],[np.sin(th),np.cos(th),0],[0,0,1]])

#Spins FM
s1 = np.array([0,0,1/2])
S_FM = {'a': [s1,s1,s1],'b': [s1,s1,s1],'c': [s1,s1,s1],'d': [s1,s1,s1],'e': [s1,s1,s1,s1,s1,s1],'f':[s1,s1,s1,s1,s1,s1],'i':[s1,s1,s1,s1],'j':[s1,s1,s1,s1]}

#Spins Neel
s1 = np.array([0,0,1/2])
S_Neel = {'a': [-s1,-s1,-s1],'b': [s1,s1,s1],'c': [-s1,s1,-s1],'d': [s1,-s1,s1],'e': [s1,-s1,s1,-s1,s1,-s1],'f':[s1,-s1,s1,-s1,s1,-s1],'i':[-s1,s1,-s1,-s1],'j':[s1,s1,s1,-s1]}

#Spins coplanar order
alpha_ = np.pi/7+1
s1 = []
s2 = []
for i in range(3):
    s1.append(np.matmul(R_z(2*np.pi/3*i),np.array([1/2,0,0])))
    s2.append(np.matmul(R_z(2*np.pi/3*i),np.array([np.cos(alpha_)/2,np.sin(alpha_)/2,0])))

S_Coplanar1 = {
        'a': s2, 
        'b': [s1[0],s1[2],s1[1]], 
        'c': [s2[0],s1[0],s2[1]], 
        'd': [s1[1],s2[1],s1[0]], 
        'e': [s1[0],s2[0],s1[0],s2[0],s1[0],s2[0]],
        'f': [s1[2],s2[0],s1[0],s2[1],s1[1],s2[2]],
        'i': [s2[0],s1[0],s2[1],s2[2]],
        'j': [s1[0],s1[2],s1[1],s2[1]]
        }

S_Coplanar2 = {
        'a': [s2[0],s2[2],s2[1]], 
        'b': [s1[0],s1[0],s1[0]], 
        'c': [s2[0],s1[0],s2[2]], 
        'd': [s1[0],s2[2],s1[0]], 
        'e': [s1[0],s2[0],s1[1],s2[1],s1[2],s2[2]],
        'f': [s1[1],s2[0],s1[0],s2[2],s1[2],s2[1]],
        'i': [s2[0],s1[0],s2[2],s2[1]],
        'j': [s1[0],s1[0],s1[0],s2[2]]
        }

#Spins non-coplanar1 order
R_nc = np.array([[0,0,1],[1,0,0],[0,1,0]])
flip_x = np.array([[1,0,0],[0,-1,0],[0,0,-1]])
flip_y = np.array([[-1,0,0],[0,1,0],[0,0,-1]])
flip_xy = np.array([[-1,0,0],[0,-1,0],[0,0,1]])
theta_ = np.pi/7+0.5
phi_ = np.pi/5+1
s0 = np.array([np.sin(theta_)*np.cos(phi_),np.sin(theta_)*np.sin(phi_),np.cos(theta_)])/2
s1 = []
s1x = []
s1y = []
s1xy = []
for i in range(3):
    s1.append(np.matmul(np.linalg.matrix_power(R_nc,i),s0))
    s1x.append(np.matmul(flip_x,s1[i]))
    s1y.append(np.matmul(flip_y,s1[i]))
    s1xy.append(np.matmul(flip_xy,s1[i]))

S_NonCoplanar1 = {
        'a': [s1[2],s1y[0],s1x[1]], 
        'b': [s1[0],s1xy[1],s1y[2]], 
        'c': [s1[2],s1[0],s1y[0]], 
        'd': [s1y[2],s1y[0],s1[0]], 
        'e': [s1[0],s1[2],s1[1],s1[0],s1[2],s1[1]],
        'f': [s1x[2],s1[2],s1[0],s1y[0],s1y[1],s1x[1]],
        'i': [s1[2],s1[0],s1y[0],s1x[1]],
        'j': [s1[0],s1xy[1],s1y[2],s1y[0]]
        }

S_NonCoplanar2 = {
        'a': [-s1[2],-s1y[0],-s1x[1]], 
        'b': [s1[0],s1xy[1],s1y[2]], 
        'c': [-s1[2],s1[0],-s1y[0]], 
        'd': [s1y[2],-s1y[0],s1[0]], 
        'e': [s1[0],-s1[2],s1[1],-s1[0],s1[2],-s1[1]],
        'f': [s1x[2],-s1[2],s1[0],-s1y[0],s1y[1],-s1x[1]],
        'i': [-s1[2],s1[0],-s1y[0],-s1x[1]],
        'j': [s1[0],s1xy[1],s1y[2],-s1y[0]]
        }

####
S_order = {'FM': S_FM, 'Neel': S_Neel, 'Coplanar1': S_Coplanar1, 'Coplanar2': S_Coplanar2, 'NonCoplanar1': S_NonCoplanar1, 'NonCoplanar2': S_NonCoplanar2}


#A^dag*A
def ada(S):
    return complex(1/2 + S[2],0)
def aad(S):
    return complex(1/2 + S[2],0)
#B^dag*B
def bdb(S):
    return complex(1/2 - S[2],0)
def bbd(S):
    return complex(1/2 - S[2],0)
#A^dag*B
def adb(S):
    return complex(S[0],S[1])
def bda(S):
    return complex(S[0],-S[1])

fun_s = {'ada': ada,'aad': aad,'bdb': bdb,'bbd': bbd,'adb': adb,'bda': bda,     'abd':bda,'bad':adb}

#Pairing and hopping "parameters". Need to put the * at the end only on the intermediate ones
def Adag(i,j,end=False):
    t1 = 'ad'+str(i)+'*bd'+str(j)
    t2 = '-bd'+str(i)+'*ad'+str(j)
    if end:
        return [t1,t2]
    else:
        return [t1+'*',t2+'*']
def A(i,j,end=False):
    t1 = 'a'+str(i)+'*b'+str(j)
    t2 = '-b'+str(i)+'*a'+str(j)
    if end:
        return [t1,t2]
    else:
        return [t1+'*',t2+'*']
def B(i,j,end=False):
    t1 = 'ad'+str(i)+'*a'+str(j)
    t2 = 'bd'+str(i)+'*b'+str(j)
    if end:
        return [t1,t2]
    else:
        return [t1+'*',t2+'*']

def alpha(n):
    """
    Operator loop type -> BB..
    """
    res = []
    for i in range(1,n):
        res.append(B(i,i+1))
    res.append(B(n,1,end=True))
    return res

def beta(n):
    """
    Operator loop type -> ad a ad a ad a for n==6 or ad a b for n==3
    """
    res = []
    for i in range(1,n,2):
        res.append(Adag(i,i+1))
        if i+1 < n:
            res.append(A(i+1,i+2))
    if n == 3:
        res.append(B(n,1,end=True))
    else:
        res.append(A(n,1,end=True))
    return res

def gamma(n):
    """
    Operator loop type -> ad a b ad a b for n==6
    """
    res = []
    for i in [1,4]:
        res.append(Adag(i,i+1))
        res.append(A(i+1,i+2))
        if i == 4:
            res.append(B(n,1,end=True))
        else:
            res.append(B(3,4))
    return res


fun_operators = {'alpha': alpha, 'beta': beta, 'gamma': gamma}

def compute_loop(loop, type_op, order,pr=False):
    """
    loop -> a,b,c,d or e
    type_op(erators) -> alpha or beta or gamma 
    order -> coplanar, ecc
    pr -> print stuff
    """
    spins = S_order[order][loop]
    ls = len(spins)
    if type_op in ['beta1','beta2','beta3']:
        ind = int(type_op[-1])-1
        type_op = 'beta'
        spins = spins[ind:] + spins[:ind]
    op = fun_operators[type_op](len(spins))
    #Save in r all the products of a and b
    r = op[0]
    for i1 in range(ls-1):
        temp = op[i1+1]
        r2 = []
        for i2 in range(len(r)):
            for i3 in range(len(temp)):
                if (r[i2][0] == '-' and temp[i3][0] == '-'):
                    r2.append(r[i2][1:]+temp[i3][1:])
                elif r[i2][0] == '-':
                    r2.append(r[i2]+temp[i3])
                elif temp[i3][0] == '-':
                    r2.append('-'+r[i2]+temp[i3][1:])
                else:
                    r2.append(r[i2]+temp[i3])
        r = r2
    #Now calculate each term in r one by one
    Calculus = complex(0,0)
    for num,res in enumerate(r):
        temp_C = complex(1,0)
        if res[0] == '-':       #remove the minus in front if it is there
            res = res[1:]
            sign = -1
        else:
            sign = 1
        #Split the terms in the single multiplication
        terms = res.split('*')
        #For each term consider the spin sites one by one
        val = []
        for i in range(1,ls+1):
            temp = ''
            for t in terms:             #extract terms with same lattice position
                if t[-1] == str(i):
                    temp += t[:-1]
            val.append(fun_s[temp](spins[i-1]))
            #For each spin site compute the associated ada, bdb, adb, ecc
            temp_C *= val[-1]
        if pr:
            if not temp_C == 0:
                print(res)
            print(temp_C)
            print(val)
            print('-------------------------------------')
        #Sum the term to the result
        Calculus += temp_C * sign
    
    return Calculus/2**ls

def Bh(res,order):
    k1 = abs(res['alpha']['b'])
    k2 = abs(res['alpha']['a'])
    k3 = abs(res['alpha']['e'])
    k4 = abs(res['alpha']['c'])
    k5 = abs(res['alpha']['d'])
    if k4 == 0 and k5 == 0:
        return 0.
    return (k4/k5)**(1/2)*(k1*k3/k2)**(1/6)
def Bhp(res,order):
    k1 = abs(res['alpha']['b'])
    k2 = abs(res['alpha']['a'])
    k3 = abs(res['alpha']['e'])
    k4 = abs(res['alpha']['c'])
    k5 = abs(res['alpha']['d'])
    if k3 == 0:
        return 0.
    if k4 == 0 and k5 == 0:
        return 0.
    return (k5/k4)**(1/2)*(k2*k3/k1)**(1/6)
def Bt(res,order):
    k1 = abs(res['alpha']['b'])
    return k1**(1/3)
def Btp(res,order):
    k2 = abs(res['alpha']['a'])
    return k2**(1/3)
def Bd(res,order):
    k1 = abs(res['alpha']['b'])
    k2 = abs(res['alpha']['a'])
    k3 = abs(res['alpha']['e'])
    k4 = abs(res['alpha']['c'])
    k5 = abs(res['alpha']['d'])
    if k3 == 0:
        return 0.
    return (k4*k5)**(1/2)*(k1*k3*k2)**(-1/6)


def Ah(res,order):
    l3 = abs(res['beta']['e'])
    l4 = abs(res['beta1']['c'])
    l5 = abs(res['beta1']['d'])
    if l3 == 0:
        return 0.
    if l4 == 0 and l5 == 0:
        return l3**(1/6)
    return (Bt(res,order)*l4/Btp(res,order)/l5)**(1/2)*l3**(1/6)
def Ahp(res,order):
    l3 = abs(res['beta']['e'])
    l4 = abs(res['beta1']['c'])
    l5 = abs(res['beta1']['d'])
    if l3 == 0:
        return 0.
    if l4 == 0 and l5 == 0:
        return l3**(1/6)
    return (Btp(res,order)*l5/Bt(res,order)/l4)**(1/2)*l3**(1/6)
def At(res,order):
    l1 = abs(res['beta']['b'])
    return (l1/Bt(res,order))**(1/2)
def Atp(res,order):
    l2 = abs(res['beta']['a'])
    return (l2/Btp(res,order))**(1/2)
def Ad(res,order):
    l3 = abs(res['beta']['e'])
    l4 = abs(res['beta1']['c'])
    l5 = abs(res['beta1']['d'])
    if l5 == 0 and l4 == 0:
        return 0.
    return (l4*l5/Bt(res,order)/Btp(res,order))**(1/2)*l3**(-1/6)


# phi is phase of pairing A
def phi_h(res,order):
    if Ah(res,order) == 0:
        return np.nan
    return 0
def phi_hp(res,order):
    if Ahp(res,order) == 0:
        return np.nan
    if abs(res['beta']['i'])>0 and abs(res['beta']['j'])>0:
        s10 = cmath.phase(res['beta']['i'])
        s11 = cmath.phase(res['beta']['j'])
        return (s10-s11+np.pi)%(2*np.pi)
    else:
        s4 = compute_loop('d','beta1',order)
        if abs(s4)>0:
            print('phi_hp at second order')
            return (-cmath.phase(s4)+phi_d(res,order)-psi_t(res,order))%(2*np.pi)
        else:
            print('wtf')
def phi_t(res,order):
    if At(res,order) == 0:
        return np.nan
    if abs(res['beta']['i'])>0 and abs(res['beta2']['d'])>0 and abs(res['gamma']['e'])>0:
        s5 = cmath.phase(res['beta2']['d'])
        s10 = cmath.phase(res['beta']['i'])
        r3 = cmath.phase(res['gamma']['e'])
        return (s5+s10+np.pi+r3)%(2*np.pi)
    else:
        return 10
def phi_tp(res,order):
    if Atp(res,order) == 0:
        return np.nan
    if abs(res['beta']['i'])>0 and abs(res['beta2']['c'])>0:
        s2 = cmath.phase(res['beta2']['c'])
        s10 = cmath.phase(res['beta']['i'])
        return (s2+s10)%(2*np.pi)
    else:
        return 10
def phi_d(res,order):
    if Ad(res,order) == 0:
        return np.nan
    if abs(res['beta']['i'])>0:
        s10 = cmath.phase(res['beta']['i'])
        return (s10+np.pi)%(2*np.pi)
    else:
        s1 = compute_loop('c','beta1',order)
        if abs(s1)>0:
            print('phi_d at second order')
            return (cmath.phase(s1)+psi_tp(res,order))%(2*np.pi)
        else:
            print('wtf')

# psi is phase of pairing A
def psi_h(res,order):
    if Bh(res,order) == 0:
        return np.nan
    return 0
def psi_hp(res,order):
    if Bhp(res,order) == 0:
        return np.nan
    if abs(res['gamma']['e'])>0:
        r3 = cmath.phase(res['gamma']['e'])
        return r3%(2*np.pi)
    else:
        g3 = compute_loop('e','alpha',order)
        if abs(g3)>0:
            print('psi_hp at second order')
            return (cmath.phase(g3)/3)%(2*np.pi)
        else:
            print('wtf')
def psi_t(res,order):
    if Bt(res,order) == 0:
        return np.nan
    if abs(res['beta']['b'])>0:
        r1 = cmath.phase(res['beta']['b'])
        return r1%(2*np.pi)
    else:
        g1 = compute_loop('b','alpha',order)
        if abs(g1)>0:
            print('psi_t at second order')
            return (cmath.phase(g1)/3)%(2*np.pi)
        else:
            print('wtf')
def psi_tp(res,order):
    if Btp(res,order) == 0:
        return np.nan
    if abs(res['beta']['a'])>0:
        r2 = cmath.phase(res['beta']['a'])
        return r2%(2*np.pi)
    else:
        g2 = compute_loop('a','alpha',order)
        if abs(g2)>0:
            print('psi_tp at second order')
            return (cmath.phase(g2)/3)%(2*np.pi)
        else:
            print('wtf')
def psi_d(res,order):
    if Bd(res,order) == 0:
        return np.nan
    if abs(res['gamma']['f'])>0:
        r5 = cmath.phase(res['gamma']['f'])
        return r5%(2*np.pi)
    else:
        return 10



























