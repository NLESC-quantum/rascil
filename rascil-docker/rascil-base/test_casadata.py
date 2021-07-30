import casacore.measures

dm = casacore.measures.measures()
vla = dm.observatory("VLA")
print(vla)
assert isinstance(vla, dict)
exit(0)
