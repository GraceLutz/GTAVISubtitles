# Hibanapló és teendők

A játék közbeni felvételekből talált hibák, a javításuk állapota és a kért, még meg nem csinált munka.
Minden javításhoz automatikus teszt tartozik (`tests/`).

Állapot: 2026-11-22. Utolsó commit: `fb039dd`. A `main` ág több committal jár a GitHub előtt (nincs pusholva, nincs új Release).

## Kijavítva (commitolva)

| Commit | Hiba a játékban | Javítás |
|---|---|---|
| `8816f20` | Középen álló, biztos sort a tiszta kép nem takart a fordítás előtt | A szomszéd képkockákon stabilan látott középső sor is takarást kap |
| `268a4a9` | Levágott elejű/végű olvasat („ul: We gotta…”) fordítása maradt kint | Ha jön teljesebb olvasat, azt fordítja újra |
| `c575a5a` | Betűhibák félrefordítása: „l'm”, „Movei Move”, „On my od”, „Isay” | Ezek javítása fordítás előtt |
| `42c66c7` | A DeepL idézőjelbe tette az aposztróffal kezdődő sort | Az egész sort körülvevő idézőjel törlése |
| `39731bb` | Csonka olvasat, énekelt sor (♪), „IWant” fordult le | Ezek kiszűrése |
| `a17abc5` | Világos jelenetben az angol eltűnése után elmosott folt maradt | Ha az angol sor eltűnt, alatta már nincs takarás |
| `4290329` | Két beszélő sora 6-szor újra kiíródott (hol 1, hol 2 sort olvasott); „…eat? l.” | Csak egyszer írja ki; a kósza „l.” törlése |
| `27e0968` | Pár nélküli „ a fordítás elején; a hosszú sor csonka, szóköz nélküli olvasata újra lefordult | Mindkettő javítva |
| `c268331` | „ocal: Bruh.” csonka név kiírva; „homegiri” → rossz fordítás | Csonka nevű sor nem kerül ki; „giri” → „girl” |
| `1cb8770` | Dalsor egy félreolvasott hangjeggyel lefordult („J Ass fatJ”); Lucia sora ötször váltakozott a sorvég nélküli változattal | Dalsor kiszűrve; a sorvég nélküli olvasat ugyanaz a felirat |
| `3ccfde6` | A mondat kisbetűs második sora külön mondatként fordult le | Az előző mondathoz fűzi |
| `c711870` | Hibás beszélőnevek: „Raut”, „L'ucia”, „Toufists”, „GTourist”, „Locat”, „Jasor”, „Cierk”… | **Tanuló névfelismerés**: az ismert vagy megtanult névre javít; az új, sokszor látott nevet megtanulja. A tanult neveket és a javításokat („Raut” → „Raul”, hányszor) a program magától a `config.json` `speaker_names` kulcsába írja (a csomagba a commitolt config kerül, a tanultak nem) |
| `fb039dd` | Mintás, világos háttéren (téglafal, ruha) a kitörölt betűk helyén elkent, „kisatírozott” sáv maradt | **Háttér-emlékezet**: a felirat megjelenése előtti képekből megjegyzi, mi van a betűk alatt, és álló kameránál azt teszi a helyükre. Ha a kép mozog vagy vágás volt, marad az elmosás. Fele felbontáson, legfeljebb 10×/mp tanul (~8–10 ms) |

Az exe 06:3x-kor újraépült, ezekkel együtt.

## Ellenőrizve a felvételen

- A takarás ugyanabban a 0,1 mp-es képkockában jelenik meg, mint az angol sor.
- A magyar általában 0,3 mp-cel később jön; ha a DeepL lassú (0,8 mp), 1,2 mp-ig üres a hely.
- Az angol eltűnése után már nem marad elmosott folt (`a17abc5`).

## Próbapad és a teljes felvétel gépi átnézése

- `tools/replay_bench.py`: a felvett képkockákat a valódi feldolgozáson játssza vissza (fordítás csak a
  gyorsítótárból, API-hívás nélkül), és képkockánként méri, hol látszik angol szöveg, mikor jön a magyar.
- A felvétel átnézése 372 kilógást talált. Többségük (225) 1–2 képkockás felvillanás az új sor megjelenésekor;
  a próbapad szerint a takarás ugyanabban a képkockában elkészül, ez kirajzolási késés. A sorszám változásakor
  70, csonka név vagy sorvég miatt 89 eset. Maradó maszat nem volt. A magyar mediánkésése 0,33 mp.
- Hosszú kilógások: 06:22:40 (a szürke „Locals:” sor 12,7 mp-ig angolul: a sorkereső nem találta a világos
  háttéren — javítva: a megtalált sor közvetlen szürke szomszédját is felveszi, teljes szélességben), 06:20:07–20 (♪ dalsorok 3–5 mp), 06:15:43, 04:39:02, 04:37:32 (zárójeles hangok).

## Ismert, még nem javított hibák

1. **Angolul felvillanó alsó sor**: két sorból a felső eltűnésekor az alsó („Local: Bruh…”) egy pillanatra angolul látszik, majd visszaáll magyarra. Kivizsgálás alatt.
2. **Beúszó sor**: 1–2 képkockán az angol még látszik, vagy a takarás csak a sor közepét fedi (a név és a vége kilóg).
3. **Csonka sor felvillanása**: a „The fuck are you” olvasat lefordul a teljes „Local: The fuck are you doing?” előtt. Egy próbált javítást visszavontam, mert a változatlan sor fordítás nélkül maradhatott volna.
4. ~~Hibás nevek~~ – kijavítva (`c711870`).
5. **Betűhibák**: „Homegirt”, „tili” (till), „talkinf”.
6. **DeepL-fordítás minősége**: a „nigga” hol „nigger”, hol „fekete”, hol változatlan; „two juniors” → „két harmadikos”.

## Kért, még meg nem csinált munka

1. **Egy helyen álló felirat** (kérve 06:17): a magyar ne ugráljon, mindig ugyanott legyen, az angolt a szokásos módon törölje.
   Folyamatban.
2. **Tanuló rendszer** (kérve 06:15): a program maga ismerje fel a hibákat, és tanuljon belőlük.
   - ✅ beszélőnevek megtanulása és a hasonló, hibás nevek javítása (`c711870`);
   - szókincs a korábbi feliratokból, az egy betűben eltérő ismeretlen szó javítása (5. hiba). A naplóból
     kiderült, hogy a saját szókincs önmagában megbízhatatlan (pl. „ball”/„all”, „case”/„cause” mind valódi
     szavak), ehhez angol szótár kell;
   - hibanapló fájl a gyanús esetekről.
4. ✅ **A teljes felvétel gépi átnézése** (kérve 06:25): kész, eredménye fent.
3. **Kiadás**: push és GitHub Release csak jóváhagyással, a zip méretének és tartalmának bemutatása után.
