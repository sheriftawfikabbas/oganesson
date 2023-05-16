
s_groups = [[
    4, 12, 20, 38, 56
],
    [3, 11, 19, 37, 55]
]

rare_earths = [x for x in range(57, 72)] + [x for x in range(89, 95)]

d_groups = [
    [21, 39] + rare_earths
]

d_groups += [[x, x+18, x+50] for x in range(22, 31)]


d_groups_no_REAs = [
    [21, 39]
]

d_groups_no_REAs += [[x, x+18, x+50] for x in range(22, 31)]

p_groups = [[x, x+8, x+26, x+44, x+76] for x in range(5, 8)]
p_groups += [[x, x+8, x+26, x+44] for x in range(8, 11)]

all_groups = d_groups + p_groups + s_groups
all_groups_no_REAs = d_groups_no_REAs + p_groups + s_groups

labels =\
    ["e", "H", "He",
     "Li", "Be", "B", "C", "N", "O", "F", "Ne",
     "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
     "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
     "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe",
     "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn",
     "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"
     ]

toxic_elements = [33,48,24,82,80]

def env_friendly(r):
    s1 = set(r.numbers)
    s2 = set(rare_earths + toxic_elements)
    if len(s1.intersection(s2)) == 0:
        return True
    else:
        return False
  