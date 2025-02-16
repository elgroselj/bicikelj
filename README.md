# Napoved števila koles na Biciklj postaji

Na Bicikelj postajah se pogosto zgodi, da na
kakšni postaji koles sploh ni - ali pa obratno, da
je postaja polna in kolesa več ne gre zakleniti.
Upravitelj Bicikelja se proti temu bori tako, da
s kombiji kolesa razvaža med postajami. Moj cilj
je bil narediti model, ki bi upravitelju pomagal
oceniti, katera kolesa mora kam prestaviti, tako
da pravilno napovem stanje na postajah, čez eno
in dve uri. No voljo sem imela približno za dva
meseca podatkov (lanski avgust in september) -
število koles na vseh Bicikelj postajah za vsake 5
minut.

Za model sem izbrala gradientBoostingRegressor iz knjižnice
sklearn in zaokrožila njegove napovedi na najbližje
število. Za delo s podatki sem uporabila še numpy
in pa pandas. Podatke o vremenu sem pridobila s
knjižnico meteostat.

Značilke
- jePonedeljek[bool]
- jePetek[bool]
- jeVikend[bool]
- pocitnice[bool]
- minutaDneva[int]
- stKolesPredEnoUro[int]
- stKolesPredDvemaUrama[int]
