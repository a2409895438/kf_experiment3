from sympy import sec, tan, cos, sin, pi

e = 1 / 298.257
R_e = 6378137

def getRmRn(sinL, posH, e=e, R_e=R_e):
    Rm = R_e * (1 - 2 * e + 3 * e * (sinL**2))
    Rn = R_e * (1 + e * (sinL**2))
    RmCur = Rm + posH
    RnCur = Rn + posH
    return RmCur, RnCur


def getL(L):
    #弧度制的计算
    return sin(L), cos(L), tan(L), sec(L)


def getPosVel(record):
    fE = record[6]
    fN = record[7]
    fU = record[8]

    posL = record[0]
    posl = record[1]
    posH = record[2]

    velE = record[3]
    velN = record[4]
    velU = record[5]
    return fE, fN, fU, posL, posl, posH, velE, velN, velU