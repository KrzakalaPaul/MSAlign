CHEM_ELEMS_MS = ['H', 'C',  'O', 'N', 'P', 'S', 'Cl', 'F', 'Br', 'I', 'B', 'As', 'Si', 'Se']
CHEM_ELEMS_MS_ABUNDANCE = [102.0, 59.0, 25.0, 13.0, 3.0, 6.0, 6.0, 17.0, 4.0, 4.0, 1.0, 1.0, 5.0, 2.0]

print(len(CHEM_ELEMS_MS))
print(len(CHEM_ELEMS_MS_ABUNDANCE))
assert len(CHEM_ELEMS_MS) == len(CHEM_ELEMS_MS_ABUNDANCE), "Element list and abundance list must be the same length"