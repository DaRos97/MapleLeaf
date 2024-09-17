import numpy as np
from numpy import cos as C
from numpy import sin as S

def E_FM(*args):
    h,t,d = args
    return 6*h+6*t+3*d

def E_Neel(*args):
    h,t,d = args
    return -6*h+6*t-3*d
#
def E_aC6(a,*args):
    h,t,d = args
    return np.sin(a)**2*(6*h+6*t+3*d)+3*np.cos(a)**2*(h-t-d)

def E_aK6(a,*args): #K=\bar(C)
    h,t,d = args
    return np.sin(a)**2*(6*t-6*h-3*d)+3*np.cos(a)**2*(d-h-t)

def E_aC3(a,*args):
    h,t,d = args
    return 6*np.sin(a)**2*(h+t)-3*np.cos(a)**2*(h+t)+3*d

def E_aK3(a,*args):
    h,t,d = args
    return 6*np.sin(a)**2*(t-h)-3*np.cos(a)**2*(t-h)-3*d

def E_aC2(a,*args):
    h,t,d = args
    return (6*h+3*d)*(np.sin(a)**2-np.cos(a)**2)+6*t

def E_aK2(a,*args):
    h,t,d = args
    return (6*h+3*d)*(np.cos(a)**2-np.sin(a)**2)+6*t

E1s = [E_aC6,E_aK6,E_aC3,E_aK3,E_aC2,E_aK2]

def E_NC2(*args):
    h,t,d = args
    return -3*d

def E_NK2(*args):
    h,t,d = args
    return 3*d

E0s = [E_FM, E_Neel,E_NC2,E_NK2]

def E_NC3(pars,*args):
    tt,ph = pars
    h,t,d = args
    return 3*np.sin(tt)**2*(2*h*np.cos(2*ph) -t +d*np.cos(2*ph-2*np.pi/3)) - 3*np.cos(tt)**2*(2*h - 2*t + d)

def E_NK3(pars,*args):
    h,t,d = args
    args2 = (-h,t,-d)
    return E_NC3(pars,*args2)

E2s = [E_NC3, E_NK3]

def E_aA3(pars,*args):
    tt, tp, ph = pars
    h,t,d = args
    return (6*h+3*d)*(S(tt)*S(tp)*C(ph)+C(tt)*C(tp)) + 6*t

def E_bA3(pars,*args):
    tt, tp, ph = pars
    h,t,d = args
    return 3*C(tt)*C(tp)*(2*h+d) + 3*S(tt)*S(tp)*C(ph)*(d-h) + 3/2*t*(3*C(tt)**2+3*C(tp)**2-2)

def E_3NC9a(pars,*args):
    tt, tp, ph = pars
    h,t,d = args
    return 3*C(tt)*C(tp)*(2*h + d) + 3*S(tt)*S(tp)*( 2*h*C(ph) + d*C(ph-2*np.pi/3)) + 3/2*t*(3*C(tt)**2+3*C(tp)**2 -2)

def E_3NC9b(pars,*args):
    tt, tp, ph = pars
    h,t,d = args
    return 3*C(tt)*C(tp)*(2*h + d) + 3*S(tt)*S(tp)*(h*C(ph) + (h+d)*C(ph+2*np.pi/3)) + 3*t*(1 + C(tp)**2 - S(tp)**2/2)

E3s = [E_aA3, E_bA3, E_3NC9a, E_3NC9b ]

def E_3NC2a(pars,*args):
    tt, tp, ph, pp = pars
    h,t,d = args
    return (3*h*(S(tt)*S(tp)*C(ph-pp) + C(tt)*C(tp) + S(tt)*C(tp)*S(ph) + C(tt)*S(tp)*C(pp) + S(tt)*S(tp)*C(ph)*S(pp)) +
            3*t*(- S(tt)*C(tt)*C(ph) + S(tt)*C(tt)*S(ph) - S(tt)**2*S(ph)*C(ph) - S(tp)*C(tp)*C(pp) - S(tp)*C(tp)*S(pp) + S(tp)**2*S(pp)*C(pp)) + 
            3*d*(S(tt)*S(tp)*S(ph)*C(pp) - S(tt)*C(tp)*C(ph) - C(tt)*S(tp)*S(pp)))

def E_3NC2b(pars,*args):
    tt, tp, ph, pp = pars
    h,t,d = args
    return (3*h*(S(tt)*S(tp)*C(ph-pp) + C(tt)*C(tp) - S(tt)*C(tp)*S(ph) - C(tt)*S(tp)*C(pp) + S(tt)*S(tp)*C(ph)*S(pp)) +
            3*t*(S(tt)*C(tt)*C(ph) - S(tt)*C(tt)*S(ph) - S(tt)**2*S(ph)*C(ph) + S(tp)*C(tp)*C(pp) + S(tp)*C(tp)*S(pp) + S(tp)**2*S(pp)*C(pp)) + 
            3*d*(S(tt)*S(tp)*S(ph)*C(pp) + S(tt)*C(tp)*C(ph) + C(tt)*S(tp)*S(pp)))

def E_3NC2c(pars,*args):
    tt, tp, ph, pp = pars
    h,t,d = args
    return (3*h*(S(tt)*S(tp)*C(ph-pp) + C(tt)*C(tp) + S(tt)*C(tp)*S(ph) - C(tt)*S(tp)*C(pp) - S(tt)*S(tp)*C(ph)*S(pp)) +
            3*t*(S(tt)*C(tt)*C(ph) + S(tt)*C(tt)*S(ph) + S(tt)**2*S(ph)*C(ph) + S(tp)*C(tp)*C(pp) - S(tp)*C(tp)*S(pp) - S(tp)**2*S(pp)*C(pp)) + 
            3*d*(-S(tt)*S(tp)*S(ph)*C(pp) + S(tt)*C(tp)*C(ph) - C(tt)*S(tp)*S(pp)))

def E_3NC2d(pars,*args):
    tt, tp, ph, pp = pars
    h,t,d = args
    return (3*h*(S(tt)*S(tp)*C(ph-pp) + C(tt)*C(tp) - S(tt)*C(tp)*S(ph) + C(tt)*S(tp)*C(pp) - S(tt)*S(tp)*C(ph)*S(pp)) +
            3*t*(- S(tt)*C(tt)*C(ph) - S(tt)*C(tt)*S(ph) + S(tt)**2*S(ph)*C(ph) - S(tp)*C(tp)*C(pp) + S(tp)*C(tp)*S(pp) - S(tp)**2*S(pp)*C(pp)) + 
            3*d*(-S(tt)*S(tp)*S(ph)*C(pp) - S(tt)*C(tp)*C(ph) + C(tt)*S(tp)*S(pp)))

E4s = [E_3NC2a,E_3NC2b,E_3NC2c,E_3NC2d]


Eall = {
        '6': [E_FM, E_Neel, E_aC6,E_aK6,E_aC3,E_aK3,E_aC2,E_aK2, E_NC2,E_NK2,E_NC3,E_NK3],
        '3': [E_FM, E_Neel, E_aA3, E_bA3, E_3NC9a, E_3NC9b, E_3NC2a,E_3NC2b,E_3NC2c,E_3NC2d],
        'all': [E_FM, E_Neel, E_aA3, E_bA3, E_3NC9a, E_3NC9b, E_3NC2a,E_3NC2b,E_3NC2c,E_3NC2d, E_aC6,E_aK6,E_aC3,E_aK3,E_aC2,E_aK2, E_NC2,E_NK2,E_NC3,E_NK3],
        }

name_list = {
        '6': ['FM','Neel','aC6','aK6','aC3','aK3','aC2','aK2','NC2','NK2','NC3','NK3'],
        '3': ['FM','Neel','aA3','bA3','3NC9a','3NC9b','3NC2a','3NC2b','3NC2c','3NC2d'],
        'all': ['FM','Neel','aA3','bA3','3NC9a','3NC9b','3NC2a','3NC2b','3NC2c','3NC2d','aC6','aK6','aC3','aK3','aC2','aK2','NC2','NK2','NC3','NK3']
        }

