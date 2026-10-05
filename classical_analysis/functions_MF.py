import numpy as np
import cmath
import functions_ssf as fs_ssf

dic_spin_coord = {
    'L1': [(0,0,5), (1,0,1), (1,1,3)],
    'L2': [(0,0,0), (0,0,1), (0,0,2), (0,0,3), (0,0,4), (0,0,5)],
    'L3': [(0,0,4), (0,0,5), (1,0,1)],
}
def get_spins_loop(loop,lattice):
    spins = []
    coords = dic_spin_coord[loop]
    for i in range(len(coords)):
        UCx,UCy,iUC = coords[i]
        spins.append(lattice[UCx,UCy,iUC])
    return spins

#A^dag*A
def ada(S):
    return complex(1/2 + S[2],0)
#B^dag*B
def bdb(S):
    return complex(1/2 - S[2],0)
#A^dag*B
def adb(S):
    return complex(S[0],S[1])
def bda(S):
    return complex(S[0],-S[1])

fun_s = {'ada': ada,'aad': ada,'bdb': bdb,'bbd': bdb,'adb': adb,'bda': bda,'abd': bda,'bad': adb}   #take the normal order

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

def compute_loop(loop, type_op, ans, lattice, disp=False):
    """
    loop -> L1 (triangle 't'), L2 (hexagon 'h'), L3 (triangle 'htd', 1)
    type_op(erators) -> alpha or beta or gamma
    disp -> print stuff
    """
    spins = get_spins_loop(loop,lattice)
    op = fun_operators[type_op](len(spins))
    #Save in r all the products of a and b
    r = op[0]
    for i1 in range(len(spins)-1):
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
    result = complex(0,0)
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
        for i in range(1,len(spins)+1):     #sites go from 1 to n+1
            temp = ''
            for t in terms:             #extract terms with same lattice position
                if t[-1] == str(i):
                    temp += t[:-1]
            val.append(fun_s[temp](spins[i-1]))
            #For each spin site compute the associated ada, bdb, adb, ecc..
            temp_C *= val[-1]
        #Sum the term to the result
        result += temp_C * sign
    #
    result = result/(2**len(spins))
    #
    if disp:
        print(loop,': ',type_op)
        print(result)
        input()
    return result

######################################################################################
def Ah(res):
    m3 = np.absolute(res['beta']['L2'])
    return m3**(1/6)
def At(res):
    m1 = np.absolute(res['alpha']['L1'])
    m2 = np.absolute(res['beta']['L1'])
    if m1==0:
        return 0.
    return m1**(-1/6) * m2**(1/2)
def Ad(res):
    m1 = np.absolute(res['alpha']['L1'])
    m2 = np.absolute(res['beta']['L1'])
    m4 = np.absolute(res['alpha']['L2'])
    m6 = np.absolute(res['beta']['L3'])
    if m2==0 or m4==0:
        return 0.
    return m1**(1/6) * m2**(-1/2) * m4**(-1/6) * m6

def Bh(res):
    m4 = np.absolute(res['alpha']['L2'])
    return m4**(1/6)
def Bt(res):
    m1 = np.absolute(res['alpha']['L1'])
    return m1**(1/3)
def Bd(res):
    m1 = np.absolute(res['alpha']['L1'])
    m4 = np.absolute(res['alpha']['L2'])
    m5 = np.absolute(res['alpha']['L3'])
    if m1==0 or m4==0:
        return 0.
    return m1**(-1/3) * m4**(-1/6) * m5

######################################################################################

# phi is phase of pairing A
def phi_h(res):
    if Ah(res)<1e-6:
        return np.nan
    return 0
def phi_t(res):
    if At(res)<1e-6:
        return np.nan
def phi_d(res):
    if Ad(res)<1e-6:
        return np.nan

# psi is phase of hopping B
def psi_h(res):
    if Bh(res,ans)<1e-6:
        return np.nan
    return 0
def psi_t(res):
    if Bt(res,ans)<1e-6:
        return np.nan
def psi_d(res):
    if Bd(res)<1e-6:
        return np.nan



























