from piezo_stroh.io import MaterialDB

db = MaterialDB()
aln_bulk = db.get("aln", "singlecrystal_sotnikov2010_our")
aln_film = db.get("aln", "thinfilm_tsubouchi1981_via_sotnikov2010")

print(aln_bulk.name, aln_bulk.C6[0,0])
print(aln_film.name, aln_film.C6[0,0])