# GTA felirat – fordított feliratpanel capture cardhoz és PS Remote Playhez

🇬🇧 [English](README.md) · 🇭🇺 Magyar

A konzol (PS5 / Xbox) képét capture carddal a PC-re hozza, kiolvassa az angol
feliratot, lefordítja **bármelyik nyelvre** (DeepL, Claude, ChatGPT vagy Google),
és egy állítható feliratpanelen mutatja. **A konzolhoz és a játékhoz nem nyúl**,
ezért kitiltást nem kockáztat.

```
Konzol ──HDMI──▶ Capture card ──USB──▶ PC: OCR → fordítás → feliratpanel
                     └─HDMI passthrough──▶ TV (opcionális)

PS5 ──hálózat──▶ PS Remote Play ablak a PC-n ◀── átlátszó feliratréteg fölötte
```

Capture card nélkül is működik: **PS Remote Play módban** a program a Remote Play
ablak fölé tesz egy átlátszó réteget, lásd [PS Remote Play mód](#ps-remote-play-mód).

## Letöltés streamereknek

Python, telepítés és kicsomagolás nélkül, egy kattintással:

**[⬇ GTA6Felirat.exe letöltése](https://github.com/GraceLutz/GTAVISubtitles/releases/latest/download/GTA6Felirat.exe)**
(vagy a [Releases](https://github.com/GraceLutz/GTAVISubtitles/releases) oldalról)

1. Tedd az exe-t egy saját mappába (pl. `C:\GTA6Felirat`), mert első indításkor
   maga mellé teszi a `config.json`-t.
2. Indítsd el a PS Remote Playt, és csatlakozz a konzolhoz.
3. Dupla kattintás a `GTA6Felirat.exe`-re. Magától megkeresi a forrást: a futó
   [Veyra](https://github.com/Likely7/Veyra-NRVideo) ablakát, ha az nincs, a PS Remote
   Playt, ha az sincs, a capture cardot (ha több videóeszköz van, kiírja, melyiket
   add meg `--device`-szal). A felirat a Medal/OBS monitorfelvételen és a streamen is
   látszik. Az indulás
   10–20 másodperc, mert az exe minden induláskor kicsomagolja magát.
4. Ha a SmartScreen figyelmeztet: *További információ → Futtatás mindenképp*
   (az exe nincs digitálisan aláírva).

Gyorsabban induló, mappás változat:
[GTA6Felirat-win64.zip](https://github.com/GraceLutz/GTAVISubtitles/releases/latest/download/GTA6Felirat-win64.zip)
(kicsomagolás után a `GTA6Felirat\GTA6Felirat.exe`-t indítsd; mellette a `README.txt`
rövid angol leírás).

Az API-kulcsot a beállítóablakban kell megadni; a Windows Hitelesítőadat-kezelője
tárolja, nem kerül a mappába. A `config.json` az exe mellett van. Az összes
[kapcsoló](#parancssori-kapcsolók) működik, pl. `GTA6Felirat.exe --lang de` vagy
capture carddal `GTA6Felirat.exe --device 1`.

A csomag elkészítése (fejlesztőknek, a `.venv` már létezzen):

```powershell
.\build_exe.ps1
```

Eredmény: `dist\GTA6Felirat.exe` (egyfájlos), valamint `dist\GTA6Felirat\` és
`dist\GTA6Felirat-win64.zip` (mappás).

## Amire szükség van

| Mi | Megjegyzés |
|---|---|
| Windows 10/11 PC, Python 3.11+ | `py --version` |
| Capture card | pl. Elgato HD60 X, AVerMedia Live Gamer, olcsó USB 3 HDMI-kártya |
| Feliratok bekapcsolva a játékban | **Beállítások → Kijelző/Hang → Feliratok: Be** |
| Egy fordító API-kulcsa (ajánlott) | lásd [Fordítók](#fordítók) |

**A HDCP-t ki kell kapcsolni PS5-ön**, különben a capture card fekete képet kap:
*Beállítások → Rendszer → HDMI → HDCP engedélyezése: Ki.* (Xbox-on játék közben
nincs HDCP, ott nem kell semmit állítani.)

## Telepítés és indítás

1. Tedd a mappát **rövid elérési útra**, pl. `C:\gta_felirat`. Nagyon hosszú
   útvonalon a Windows nem tudja betölteni az OCR egyik DLL-jét.
2. PowerShellben, a mappában:

```powershell
.\inditas.ps1 --list-devices
```

Az első indítás telepít mindent (pár perc), utána kiírja a videóeszközöket,
pl. `1: 1920x1080`. Ez a capture card sorszáma.

```powershell
.\inditas.ps1 --device 1
```

Elindul a játékablak és mellette a **beállítóablak**. Ha a PowerShell nem engedi
futtatni a szkriptet:

```powershell
powershell -ExecutionPolicy Bypass -File .\inditas.ps1 --device 1
```

## PS Remote Play mód

Nem kell capture card és HDCP-állítás sem. Indítsd el a PS Remote Playt, csatlakozz
a PS5-höz, kapcsold be a feliratokat a játékban, majd:

```powershell
.\inditas.ps1 --remote-play
```

- A fordított felirat egy **átlátszó, mindig felül lévő rétegen** jelenik meg a Remote
  Play ablak fölött, és követi az ablakot (mozgatás, átméretezés, teljes képernyő).
- A **kontroller a Remote Playben marad**: a rétegre kattintás nem veszi el a fókuszt,
  az átlátszó részeken pedig a kattintás átmegy a Remote Playhez.
- A panelt itt is **húzhatod, görgővel méretezheted, jobb klikkel visszaáll**.
- A beállítóablakban a *„Mintaszöveg és a figyelt sáv mutatása”* jelölőnégyzet
  mutatja, hol keresi a program az angol feliratot.
- **Kilépés:** zárd be a beállítóablakot. Ha a Remote Playt zárod be, a program is kilép.
- A magyar felirat alapból **látszik a felvételeken és streamen** (lásd lent). A program
  közvetlenül a Remote Play ablak tartalmát olvassa (a fölötte lévő réteg nélkül), így
  mindig az eredeti angol feliratot látja, akkor is, ha a réteg letakarja.
- Az **angol felirat eltakarása** bekapcsolva (`h` gomb vagy a beállítóablak) az angol
  szöveg helyére a játékkép elmosott, kissé sötétített, lágy szélű változata kerül, rajta a
  fordítással. Ez a felvételen és a streamen is így látszik.

### Felvétel és stream (Medal, OBS → Twitch, Kick, YouTube)

A feliratréteg egy külön ablak a Remote Play fölött, ezért a rögzítőprogramban a
**monitort** kell felvenni, ne csak a Remote Play ablakot vagy játékot:

| Program | Beállítás |
|---|---|
| **Medal** | Capture mode: *Monitor / Full screen* azon a monitoron, ahol a Remote Play fut |
| **OBS** (Twitch, Kick, YouTube) | Forrás: *Képernyő-felvétel* (Display Capture) a kijelölt monitorra |
| **Xbox Game Bar, NVIDIA ShadowPlay** | teljes asztal / monitor rögzítése |

A panel „Látszódjon a felvételen és streamen” kapcsolójával ki is kapcsolhatod: ekkor a
felirat csak neked látszik, a felvételen nem (Windows 10 2004 vagy újabb kell hozzá).
Ha a Remote Play ablak tartalma valamiért nem rögzíthető (a program ilyenkor figyelmeztet
és a képernyőről olvas tovább), felvételre látszó módban az „angol felirat eltakarása” nem
érvényes: a fordítás az angol alá kerül, mert a takarást az OCR is látná.

**Streamereknek:** a program a GitHubról letölthető, a telepítés lépései fent vannak. Mindenki a
saját fordítókulcsát adja meg a panelen (DeepL ingyenes kulcs is elég).

#### Tiszta kép a nézőknek (nem kötelező)

Élőben az új angol sor első pillanata átcsúszhat, és amíg a fordítás megérkezik, csak
takarás látszik. Ha a nézők ebből semmit se lássanak, kapcsold be a panel „Tiszta kép a
nézőknek” kapcsolóját. Megnyílik egy külön ablak, benne a játék képe **~1 mp késéssel**:
mire egy képkocka kikerül, a feliratának fordítása elkészül, így csak a kész magyar szöveg
látszik rajta, angol soha.

- OBS-ben a monitor helyett a *„Tiszta kép – ezt az ablakot vedd fel”* ablakot add hozzá
  *Ablak-felvétel* (Window Capture) forrásként (Medalban: ezt az ablakot rögzítsd).
- A kép késik, ezért a játék hangját is késleltesd ugyanennyivel OBS-ben
  (*Speciális hangtulajdonságok → Szinkron eltolás*: `1000` ms).
- Te továbbra is az élő képen játszol; a tiszta ablak csak a nézőknek szól.
- Kb. 200 MB-tal több memóriát használ. Alapból ki van kapcsolva; a késés a
  `clean_output_delay_seconds` beállítással változtatható (0,5–5 mp).

Más ablak fölé is teheted (pl. Xbox alkalmazás, böngészős felhőjáték), az ablak
pontos címével:

```powershell
.\inditas.ps1 --window "Xbox"
```

Mért értékek egy 1920×1080-as Remote Play ablakon: képrögzítés ~36 ms, OCR felirat
nélkül ~8 ms, felirattal ~75–120 ms. A teljes késést a [fordító](#mekkora-a-késés)
határozza meg. A Remote Play saját képkésése (kb. 50–150 ms) ehhez nem adódik
hozzá, mert a felirat ugyanahhoz a képhez igazodik.

## A feliratpanel

**A játékablakban, egérrel:**

| Mozdulat | Hatás |
|---|---|
| Bal gombbal húzod | a panel oda kerül, ahová teszed (elmentődik) |
| Görgő a panel fölött | nagyobb vagy kisebb betű |
| Jobb klikk a panelen | vissza az angol felirat helyére |
| `r` gomb | mintaszöveg, ha épp nincs felirat (így be tudod állítani) |

Ha az angol felirat eltakarása be van kapcsolva, az eredeti angol szöveget a játékkép
elmosott, lágy szélű változata takarja, akkor is, ha a panelt máshová húzod
(a `h` gombbal kikapcsolható).

**A beállítóablakban** (a `p` gombbal elrejthető vagy előhozható):

- **🌐 A program nyelve:** az ablak tetején. Automatikus = a Windows nyelve, vagy
  English, Magyar, Deutsch, Español, Français, Polski, Português. Ez csak a program
  felületének nyelve; a felirat célnyelve külön állítható, és sokkal több nyelv közül.
- **Célnyelv:** 30 nyelv listából, vagy bármelyik nyelvkód beírva + Enter (pl. `lt`, `et`).
- **Fordító:** Automatikus, DeepL, Claude, ChatGPT vagy Google.
- **API-kulcs:** a **Mentés** gomb a Windows Hitelesítőadat-kezelőjébe menti (titkosítva, a
  felhasználói fiókodhoz kötve), így a következő indításkor nem kell újra beírni. A **Törlés**
  gomb eltávolítja. A kulcs soha nem kerül a `config.json`-ba vagy a repóba.
- **AI modell:** Claude-nál vagy GPT-nél átírható, Enterrel érvényes.
- **Betűméret, háttér fedése:** csúszkák, azonnal látszik a hatásuk.
- **Angol felirat eltakarása**, valamint **panel vissza az eredeti helyre**.
- **Vékony betű (mint a GTA-ban):** a beszélő neve türkiz, a szöveg fehér, halvány árnyékkal.
- **Csak a kimondott szöveg:** csak a „Név: szöveg” párbeszédsorokat fordítja; küldetésszöveg,
  tábla, menü, telefonhívás-kijelző nem megy fordításra (alapból bekapcsolva). A tévéműsor név nélküli
  felirata is lefordul, ha beszédre vall (kérdés, felkiáltás, első személyű mondat, vagy halvány,
  szürkésfehér betű); a felszólító küldetésszöveg („Go to the Vice City Port.”) nem.
- **Streamer mód:** a sértő szavakat (n-szó, homofób és más gyalázkodás, magyarul is) kicsillagozva
  mutatja, pl. `n****`. Külön kapcsolható, alapból kikapcsolva; a fordítást nem változtatja meg, csak
  a kirajzolt szöveget, így bármikor ki-be kapcsolható.
- Alul az állapot: melyik fordító fut, **a mért fordítási idő (ms)**, és ha hiba van, mi a gond.

Minden módosítás azonnal érvényes, és bekerül a `config.json`-ba (a kulcsok kivételével).

## Fordítók

| Fordító | Sebesség (becsült) | Minőség | Kulcs (környezeti változó) |
|---|---|---|---|
| **DeepL** | leggyorsabb, kb. 150–400 ms | nagyon jó | `DEEPL_API_KEY` – a *DeepL API Free* havi 500 000 karakterig ingyenes |
| **Claude** (Haiku) | kb. 400–900 ms | a legtermészetesebb, érti a szlenget és az előző sorokat | `ANTHROPIC_API_KEY` – console.anthropic.com |
| **ChatGPT** (GPT mini) | kb. 400–1000 ms | nagyon jó | `OPENAI_API_KEY` – platform.openai.com |
| **Google** | kb. 100–300 ms | közepes | nem kell kulcs, de néha letiltja a sok kérést |

**Fontos:** a Claude Pro és a ChatGPT Plus előfizetés **nem ad API-kulcsot**. Az
API-t külön kell feltölteni a fenti oldalakon, használat szerint fizetős. Egy
végigjátszás feliratai jellemzően pár dollárba kerülnek.

Kulcs tartós beállítása a panel **Mentés** gombja helyett környezeti változóval
(utána nyiss új PowerShell-ablakot):

```powershell
setx DEEPL_API_KEY "ide-a-kulcsod"
```

Az **Automatikus** mód ebben a sorrendben választ abból, amihez van kulcs:
DeepL → Claude → ChatGPT → Google. Az AI-fordítók az előző 3 sort is megkapják,
így a nemeket, a megszólítást és a hangnemet jobban eltalálják. Ha a fordítás nem
sikerül, az angol szöveg jelenik meg „(fordítás nem sikerült)” előtaggal.

### Mekkora a késés?

Mért idő a felirat megjelenésétől a lefordított szövegig (300 ms-os, illetve
700 ms-os fordítást szimulálva, ugyanezen a gépen):

| Fordítási idő | Átlagos teljes késés |
|---|---|
| ~300 ms (DeepL-szerű) | **kb. 0,5 mp** (0,44–0,6) |
| ~700 ms (Claude/GPT-szerű) | **kb. 0,9 mp** (0,82–1,09) |
| ismétlődő mondat | csak az OCR, kb. 0,1–0,25 mp |

Amitől ilyen gyors:

- A fordítás **már az első olvasásnál elindul**, a megerősítést nem várja meg. Ha
  a második olvasás ugyanazt adja, a kész fordítás azonnal megjelenik.
- A fordítóval **nyitva tartott kapcsolat** marad, így nem kell minden mondatnál
  új HTTPS-kézfogás (a fenti mérésben ez még nincs benne, élesben tovább gyorsít).
- 0,15 mp-enként olvas, és egy olvasás kb. 75 ms.

A beállítóablak alján élesben is látszik a mért fordítási idő.

## Webkamera (nem kötelező)

Csak akkor indul, ha megadod: `.\inditas.ps1 --device 1 --webcam 0`. A képet
egérrel húzhatod, görgővel méretezheted, a `c` gombbal elrejtheted.

## Hang

Az overlay ablak csak képet mutat. A hanghoz válassz egyet:

- **Windows hangbemenet:** *Gépház → Rendszer → Hang → a capture card bemenete →
  Tulajdonságok → „Az eszköz hallgatása”*. A legegyszerűbb, de 50-100 ms késést adhat.
- **Fejhallgató a kontrolleren** (PS5 DualSense / Xbox kontroller jack). Ez késés nélküli.
- **Passthrough a TV-re** némítva, a hang a TV-ből, a kép a PC-ről.

## Billentyűk a játékablakban

| Gomb | Funkció |
|---|---|
| `q` / `Esc` | kilépés |
| `p` | beállítóablak elrejtése / előhozása |
| `f` | teljes képernyő be/ki |
| `h` | eredeti angol felirat takarása be/ki |
| `r` | figyelt sáv keretezése + mintaszöveg a panel helyén |
| `o` | feliratpanel ki/be |
| `c` | webkamerás kép ki/be |

## Parancssori kapcsolók

| Kapcsoló | Jelentés |
|---|---|
| `--auto` | a forrás automatikus keresése: Veyra, PS Remote Play, capture card (az exe alapmódja) |
| `--device N` | capture card sorszáma |
| `--capture-api auto\|dshow\|msmf` | a capture card megnyitásának módja; ha a kártya DirectShow-n nem nyílik meg, próbáld az `msmf`-et (config.json: `capture_api`) |
| `--remote-play` | a PS Remote Play ablak fölé teszi a feliratot |
| `--window "CÍM"` | tetszőleges ablak fölé, a pontos ablakcímmel vagy a címe elejével (pl. `--window Veyra`) |
| `--lang KÓD` | célnyelv, pl. `hu`, `de`, `es` |
| `--translator NÉV` | `auto`, `deepl`, `claude`, `openai`, `google` |
| `--windowed` | ablakban indul |
| `--no-panel` | beállítóablak nélkül indul |
| `--image kép.png` | teszt egy képernyőképpel, capture card nélkül |
| `--list-devices` | videóeszközök listája |

## Beállítások (`config.json`)

| Kulcs | Alapérték | Jelentés |
|---|---|---|
| `device_index` | `0` | capture card sorszáma |
| `capture_width` / `capture_height` / `capture_fps` | 1920 / 1080 / 60 | kért képméret |
| `subtitle_region` | 0.06 / 0.55 / 0.94 / 0.98 | ahol a program az angol feliratot keresi |
| `ocr_interval_seconds` | `0.1` | milyen gyakran olvas (az OCR ~0,1 mp, ennél sűrűbben nem tud) |
| `min_text_score` | `0.6` | ennél bizonytalanabb OCR-találatot eldob |
| `stable_reads` | `2` | ennyi egyező olvasás után fordít |
| `similarity_threshold` | `0.85` | ennyire hasonló szöveget tekint ugyanannak |
| `hold_seconds` | `0.8` | ennyi ideig marad kint a fordítás az angol eltűnése után |
| `translator` | `"auto"` | `auto`, `deepl`, `claude`, `openai`, `google` |
| `target_lang` | `"hu"` | célnyelv kódja |
| `ui_language` | `"auto"` | a program felületének nyelve: `"auto"` (a Windows nyelve) vagy pl. `"en"`, `"hu"` |
| `claude_model` | `"claude-haiku-5-5"` | Claude-modell |
| `openai_model` | `"gpt-4.1-mini"` | GPT-modell |
| `hide_original` | `true` | az angol szöveg takarása |
| `visible_in_recordings` | `true` | a felirat látszik a Medal/OBS monitorfelvételen és streamen |
| `only_spoken_lines` | `true` | csak a „Név: szöveg” párbeszédsorok fordítása |
| `streamer_mode` | `false` | streamer mód: a sértő szavak kicsillagozva |
| `clean_output` | `false` | Remote Play mód: késleltetett tiszta kép a nézőknek külön ablakban (csak a kész fordítás) |
| `clean_output_delay_seconds` | `1.0` | a tiszta kép késése (0,5–5 mp); a stream hangját is ennyivel késleltesd |
| `fast_capture` | `true` | Remote Play mód: a feliratsáv folyamatos rögzítése külön szálon (gyorsabb takarás és olvasás); `false` értékkel visszaáll az olvasásonkénti rögzítés |
| `row_reading` | `true` | Remote Play mód: soronkénti olvasás – csak az újonnan megjelent feliratsort olvassa be (gyorsabb fordítás, kisebb terhelés); a `fast_capture` kell hozzá és legalább 1600 képpont széles játékkép, különben mindig a teljes sávot olvassa. `false` értékkel mindig a teljes sávot olvassa |
| `panel_opacity` | `245` | a panel háttere, 0 (átlátszó) – 255 (teljesen fedő) |
| `panel_width_ratio` | `0.9` | a fordítás legnagyobb szélessége a képszélesség arányában (0.3–1.0); Minta módban a panel oldalát húzva |
| `panel_height_ratio` | `null` | a panel legkisebb magassága a képmagasság arányában (0.04–0.5); `null` = a szöveghez igazodik; Minta módban a panel tetejét vagy alját húzva |
| `subtitle_x` / `subtitle_y` | `null` | a panel közepe a kép arányában; `null` = az angol felirat helyén |
| `font_path` | Segoe UI (vékony, GTA-stílus) | bármilyen `.ttf`; a panel „Vékony betű” kapcsolója Segoe UI és Segoe UI Bold között vált |
| `speaker_color` | `"#7ECFD6"` | a beszélő nevének színe (`#RRGGBB`), mint a GTA-feliratban |
| `font_size_ratio` | `0.034` | betűméret a képmagasság arányában (0.015–0.08) |
| `fullscreen` | `true` | teljes képernyőn indul |

**Kínai, japán, koreai, arab** célnyelvnél a Segoe UI nem tud minden betűt. Állítsd
a `font_path`-t pl. `C:/Windows/Fonts/msyh.ttc`-re (kínai), `YuGothB.ttc`-re
(japán), `malgunbd.ttf`-re (koreai).

## Tesztelés capture card nélkül

```powershell
.\inditas.ps1 --image screenshot.png --windowed
```

Fejlesztői tesztek:

```powershell
.\.venv\Scripts\python -m pip install pytest==9.2.0
.\.venv\Scripts\python -m pytest tests
```

## Korlátok

- Gépi fordítás: szójátékok és utalások néha furán jönnek át (AI-fordítóval ritkábban).
- Ha a játék más helyre vagy más színnel írja a feliratot, a `subtitle_region`-t
  kell igazítani. Fehér szöveghez van hangolva.
- A beszélő nevét („Jason: …”) nem fordítja, csak a mondatot.
- Arab nyelvnél a jobbról balra írást a panel nem rendezi át.
- Remote Play módban a felirat a Remote Play ablakon jelenik meg, a TV-n nem.
  Ha a PS5 egy jelenetnél letiltja a Remote Play képet, ott a felirat is kimarad.
