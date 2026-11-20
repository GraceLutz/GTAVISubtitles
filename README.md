# GTA felirat – fordított feliratpanel capture cardhoz és PS Remote Playhez

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
- A réteg kimarad a képrögzítésből (Windows 10 2004 vagy újabb), így a program
  mindig az eredeti angol feliratot olvassa, nem a saját fordítását.

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

Ha máshová húzod a panelt, az eredeti angol szöveget egy sötét doboz takarja
(a `h` gombbal kikapcsolható).

**A beállítóablakban** (a `p` gombbal elrejthető vagy előhozható):

- **Célnyelv:** 30 nyelv listából, vagy bármelyik nyelvkód beírva + Enter (pl. `lt`, `et`).
- **Fordító:** Automatikus, DeepL, Claude, ChatGPT vagy Google.
- **API-kulcs:** a **Mentés** gomb a Windows Hitelesítőadat-kezelőjébe menti (titkosítva, a
  felhasználói fiókodhoz kötve), így a következő indításkor nem kell újra beírni. A **Törlés**
  gomb eltávolítja. A kulcs soha nem kerül a `config.json`-ba vagy a repóba.
- **AI modell:** Claude-nál vagy GPT-nél átírható, Enterrel érvényes.
- **Betűméret, háttér fedése:** csúszkák, azonnal látszik a hatásuk.
- **Angol felirat eltakarása**, valamint **panel vissza az eredeti helyre**.
- **Csak a kimondott szöveg:** csak a „Név: szöveg” párbeszédsorokat fordítja; küldetésszöveg,
  tábla, menü, telefonhívás-kijelző nem megy fordításra (alapból bekapcsolva).
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
| `--device N` | capture card sorszáma |
| `--remote-play` | a PS Remote Play ablak fölé teszi a feliratot |
| `--window "CÍM"` | tetszőleges ablak fölé, pontos ablakcímmel |
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
| `subtitle_region` | 0.12 / 0.78 / 0.88 / 0.98 | ahol a program az angol feliratot keresi |
| `ocr_interval_seconds` | `0.1` | milyen gyakran olvas (az OCR ~0,1 mp, ennél sűrűbben nem tud) |
| `min_text_score` | `0.6` | ennél bizonytalanabb OCR-találatot eldob |
| `stable_reads` | `2` | ennyi egyező olvasás után fordít |
| `similarity_threshold` | `0.85` | ennyire hasonló szöveget tekint ugyanannak |
| `hold_seconds` | `0.8` | ennyi ideig marad kint a fordítás az angol eltűnése után |
| `translator` | `"auto"` | `auto`, `deepl`, `claude`, `openai`, `google` |
| `target_lang` | `"hu"` | célnyelv kódja |
| `claude_model` | `"claude-haiku-5-5"` | Claude-modell |
| `openai_model` | `"gpt-4.1-mini"` | GPT-modell |
| `hide_original` | `true` | az angol szöveg takarása |
| `only_spoken_lines` | `true` | csak a „Név: szöveg” párbeszédsorok fordítása |
| `panel_opacity` | `245` | a panel háttere, 0 (átlátszó) – 255 (teljesen fedő) |
| `subtitle_x` / `subtitle_y` | `null` | a panel közepe a kép arányában; `null` = az angol felirat helyén |
| `font_path` | Segoe UI Bold | bármilyen `.ttf`, ami tudja a célnyelv betűit |
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
