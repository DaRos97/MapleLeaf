import numpy as np

def R_z(th):
    return np.array([[np.cos(th),-np.sin(th),0],[np.sin(th),np.cos(th),0],[0,0,1]])

#Spins FM
s1 = np.array([0,0,1/2])
S_FM = {'a': [s1,s1,s1],'b': [s1,s1,s1],'c': [s1,s1,s1],'d': [s1,s1,s1],'e': [s1,s1,s1,s1,s1,s1]}

#Spins Neel
s1 = np.array([0,0,1/2])
S_Neel = {'a': [-s1,-s1,-s1],'b': [s1,s1,s1],'c': [-s1,s1,-s1],'d': [s1,s1,-s1],'e': [s1,-s1,s1,-s1,s1,-s1]}

#Spins coplanar order
alpha_ = np.pi/2
s1 = []
s2 = []
for i in range(3):
    s1.append(np.matmul(R_z(2*np.pi/3*i),np.array([1/2,0,0])))
    s2.append(np.matmul(R_z(2*np.pi/3*i),np.array([np.cos(alpha_)/2,np.sin(alpha_)/2,0])))

S_Coplanar = {'a': s2, 'b': [s1[0],s1[2],s1[1]], 'c': [s2[0],s1[0],s2[1]], 'd': [s1[0],s1[1],s2[1]], 'e': [s1[0],s2[0],s1[1],s2[1],s1[2],s2[2]]}

####
S_order = {'FM': S_FM, 'Neel': S_Neel, 'Coplanar': S_Coplanar}


#A^dag*A
def ada(S):
    return complex(1/2 + S[2],0)
def aad(S):
    return complex(3/2 + S[2],0)
#B^dag*B
def bdb(S):
    return complex(1/2 - S[2],0)
def bbd(S):
    return complex(3/2 - S[2],0)
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
    t1 = 'a'+str(i)+'*a'+str(j)
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

fun_operators = {'alpha': alpha, 'beta': beta}

def compute_loop(loop, type_op, order,pr=False):
    """
    loop -> a,b,c,d or e
    type_op(erators) -> alpha or beta or gamma 
    order -> coplanar, ecc
    pr -> print stuff
    """
    spins = S_order[order][loop]
    ls = len(spins)
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
        for i in range(1,ls+1):
            temp = ''
            for t in terms:             #extract terms with same lattice position
                if t[-1] == str(i):
                    temp += t[:-1]
            val = fun_s[temp](spins[i-1])
            #For each spin site compute the associated ada, bdb, adb, ecc
            temp_C *= val
        if pr:
            print(temp_C)
            if not temp_C == 0:
                print(res)
        #Sum the term to the result
        Calculus += temp_C * sign
    
    return Calculus/2**ls

def Bh(res):
    k1 = abs(res['alpha']['b'])
    k2 = abs(res['alpha']['a'])
    k3 = abs(res['alpha']['e'])
    k4 = abs(res['alpha']['c'])
    k5 = abs(res['alpha']['d'])
    return (k4/k5)**(1/2)*(k1*k3/k2)**(1/6)
def Bhp(res):
    k1 = abs(res['alpha']['b'])
    k2 = abs(res['alpha']['a'])
    k3 = abs(res['alpha']['e'])
    k4 = abs(res['alpha']['c'])
    k5 = abs(res['alpha']['d'])
    return (k5/k4)**(1/2)*(k2*k3/k1)**(1/6)
def Bt(res):
    k1 = abs(res['alpha']['b'])
    return k1**(1/3)
def Btp(res):
    k2 = abs(res['alpha']['a'])
    return k2**(1/3)
def Bd(res):
    k1 = abs(res['alpha']['b'])
    k2 = abs(res['alpha']['a'])
    k3 = abs(res['alpha']['e'])
    k4 = abs(res['alpha']['c'])
    k5 = abs(res['alpha']['d'])
    return (k4*k5)**(1/2)*(k1*k3*k2)**(-1/6)


def Ah(res):
    l3 = abs(res['beta']['e'])
    l4 = abs(res['beta']['c'])
    l5 = abs(res['beta']['d'])
    if l3 == 0:
        return 0.
    if l4 == 0 and l5 == 0:
        return l3**(1/6)
    return (Bt(res)*l4/Btp(res)/l5)**(1/2)*l3**(1/6)
def Ahp(res):
    l3 = abs(res['beta']['e'])
    l4 = abs(res['beta']['c'])
    l5 = abs(res['beta']['d'])
    if l3 == 0:
        return 0.
    if l4 == 0 and l5 == 0:
        return l3**(1/6)
    return (Btp(res)*l5/Bt(res)/l4)**(1/2)*l3**(1/6)
def At(res):
    l1 = abs(res['beta']['b'])
    return (l1/Bt(res))**(1/2)
def Atp(res):
    l2 = abs(res['beta']['a'])
    return (l2/Btp(res))**(1/2)
def Ad(res):
    l3 = abs(res['beta']['e'])
    l4 = abs(res['beta']['c'])
    l5 = abs(res['beta']['d'])
    if l5 == 0 and l4 == 0:
        return 0
    return (l4*l5/Bt(res)/Btp(res))**(1/2)*l3**(1/6)































