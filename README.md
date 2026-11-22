# GTA Subtitles – translated subtitle panel for capture cards and PS Remote Play

🇬🇧 English · 🇭🇺 [Magyar](README.hu.md)

Brings your console's (PS5 / Xbox) picture to the PC through a capture card, reads the
English subtitles, translates them into **any language** (DeepL, Claude, ChatGPT or
Google) and shows them on an adjustable subtitle panel. **It never touches the console
or the game**, so there is no risk of a ban.

```
Console ──HDMI──▶ Capture card ──USB──▶ PC: OCR → translation → subtitle panel
                     └─HDMI passthrough──▶ TV (optional)

PS5 ──network──▶ PS Remote Play window on the PC ◀── transparent subtitle layer on top
```

It also works without a capture card: in **PS Remote Play mode** the program puts a
transparent layer over the Remote Play window, see [PS Remote Play mode](#ps-remote-play-mode).

## Download for streamers

No Python, no installation, no unzipping, just one click:

**[⬇ Download GTA6Felirat.exe](https://github.com/GraceLutz/GTAVISubtitles/releases/latest/download/GTA6Felirat.exe)**
(or from the [Releases](https://github.com/GraceLutz/GTAVISubtitles/releases) page)

1. Put the exe in its own folder (e.g. `C:\GTA6Felirat`), because on first start it
   creates its `config.json` next to itself.
2. Start PS Remote Play and connect to your console.
3. Double-click `GTA6Felirat.exe`. It finds the source by itself: a running
   [Veyra](https://github.com/Likely7/Veyra-NRVideo) window, otherwise PS Remote
   Play, otherwise a capture card (with several video devices it prints which one to pass
   with `--device`). The subtitles also show up in Medal/OBS monitor recordings and on
   stream. Startup takes
   10–20 seconds, because the exe unpacks itself every time it starts.
4. If SmartScreen warns you: *More info → Run anyway*
   (the exe is not digitally signed).

Faster-starting folder version:
[GTA6Felirat-win64.zip](https://github.com/GraceLutz/GTAVISubtitles/releases/latest/download/GTA6Felirat-win64.zip)
(after unzipping, run `GTA6Felirat\GTA6Felirat.exe`; the `README.txt` next to it is a
short guide).

Enter the API key in the settings window; it is stored in Windows Credential Manager,
not in the folder. The `config.json` sits next to the exe. All
[command-line options](#command-line-options) work, e.g. `GTA6Felirat.exe --lang de`, or
with a capture card `GTA6Felirat.exe --device 1`.

**Program language:** choose it at the top of the settings window (🌐 *Program
language*). *Automatic* follows the Windows display language; available: English,
Magyar, Deutsch, Español, Français, Polski, Português. The target language of the
subtitles is set separately, and there are many more of those.

Building the packages (for developers, the `.venv` must already exist):

```powershell
.\build_exe.ps1
```

Output: `dist\GTA6Felirat.exe` (single file), plus `dist\GTA6Felirat\` and
`dist\GTA6Felirat-win64.zip` (folder version).

## Requirements

| What | Notes |
|---|---|
| Windows 10/11 PC, Python 3.11+ | `py --version` |
| Capture card | e.g. Elgato HD60 X, AVerMedia Live Gamer, a cheap USB 3 HDMI card |
| Subtitles turned on in the game | **Settings → Display/Audio → Subtitles: On** |
| An API key for a translator (recommended) | see [Translators](#translators) |

**HDCP must be turned off on PS5**, otherwise the capture card only gets a black
picture: *Settings → System → HDMI → Enable HDCP: Off.* (Xbox has no HDCP during
gameplay, so nothing to change there.)

## Installation and launch

1. Put the folder on a **short path**, e.g. `C:\gta_felirat`. On a very long path
   Windows cannot load one of the OCR DLLs.
2. In PowerShell, inside the folder:

```powershell
.\inditas.ps1 --list-devices
```

The first run installs everything (a few minutes), then lists the video devices,
e.g. `1: 1920x1080`. That number is your capture card.

```powershell
.\inditas.ps1 --device 1
```

The game window opens, with the **settings window** next to it. If PowerShell refuses
to run the script:

```powershell
powershell -ExecutionPolicy Bypass -File .\inditas.ps1 --device 1
```

## PS Remote Play mode

No capture card and no HDCP change needed. Start PS Remote Play, connect to your PS5,
turn on subtitles in the game, then:

```powershell
.\inditas.ps1 --remote-play
```

- The translated subtitle appears on a **transparent, always-on-top layer** over the
  Remote Play window, and follows the window (moving, resizing, full screen).
- **The controller stays in Remote Play**: clicking the layer does not steal focus, and
  clicks on the transparent parts go through to Remote Play.
- Here too you can **drag the panel, resize it with the wheel, and right-click to reset it**.
- In the settings window, the *“Show sample text and the watched area (for setup)”*
  checkbox shows where the program looks for the English subtitle.
- **To quit:** close the settings window. If you close Remote Play, the program quits too.
- By default the translated subtitle **shows up in recordings and on stream** (see below).
  The program reads the Remote Play window's content directly (without the layer on top),
  so it always sees the original English subtitle, even when the layer covers it.
- With **covering the English subtitle** turned on (`h` key or the settings window), the
  English text is replaced by a blurred, slightly darkened, soft-edged version of the game
  picture, with the translation on top. Recordings and streams show it the same way.

### Recording and streaming (Medal, OBS → Twitch, Kick, YouTube)

The subtitle layer is a separate window over Remote Play, so your recording software has
to capture the **monitor**, not just the Remote Play window or the game:

| Program | Setting |
|---|---|
| **Medal** | Capture mode: *Monitor / Full screen* on the monitor where Remote Play runs |
| **OBS** (Twitch, Kick, YouTube) | Source: *Display Capture* on that monitor |
| **Xbox Game Bar, NVIDIA ShadowPlay** | record the full desktop / monitor |

You can turn this off with the panel's “Show in recordings and streams (Medal, OBS,
Twitch, Kick, YouTube)” switch: the subtitle is then visible only to you, not in the
recording (requires Windows 10 2004 or later). If the Remote Play window's content cannot
be captured for some reason (the program warns you and keeps reading from the screen),
“Cover the English subtitle” does not apply while the subtitle is visible in recordings:
the translation goes below the English text instead, because the OCR would see the cover.

**For streamers:** the program can be downloaded from GitHub; the setup steps are above.
Everyone enters their own translator key in the panel (a free DeepL key is enough).

#### Clean picture for viewers (optional)

Live, the first moment of a new English line can slip through, and until the translation
arrives only a cover shows. If viewers should see none of that, turn on the panel's
“Clean picture for viewers” switch. A separate window opens with the game picture
**delayed by ~1 second**: by the time a frame is shown, the translation of its subtitle is
ready, so it shows only the finished Hungarian text, never the English one.

- In OBS add the window *“Clean picture – capture this window”* as a *Window Capture*
  source (in Medal: capture that window), instead of the monitor.
- The picture is delayed, so delay the game audio by the same amount in OBS
  (*Advanced Audio Properties → Sync Offset*: `1000` ms).
- You keep playing on the live picture; the clean window is for viewers only.
- It uses about 200 MB more memory. Off by default; the delay can be set with
  `clean_output_delay_seconds` (0.5–5 s).

#### The subtitle as text in OBS (optional)

The program also writes the subtitle on screen to **`obs_felirat.txt`** next to the exe (one line
per speaker; empty when there is no subtitle). OBS can then show it as its own text, with the font,
colour, outline and position set in OBS.

- In OBS: *Sources → + → Text (GDI+)*, tick *Read from file* and pick `obs_felirat.txt`.
- With the clean picture for viewers on, the file is delayed by the same amount as the picture.
- In streamer mode the slurs are masked in the file too.
- This is only the translated text: the English subtitle in the game picture is removed by the cover
  or the clean picture.
- To turn it off: `"obs_text_file": false` in `config.json`.

You can also put it over another window (e.g. the Xbox app, a cloud game in the browser),
using the window's exact title:

```powershell
.\inditas.ps1 --window "Xbox"
```

Measured on a 1920×1080 Remote Play window: frame capture ~36 ms, OCR without subtitles
~8 ms, with subtitles ~75–120 ms. The total delay is mostly down to the
[translator](#how-long-is-the-delay). Remote Play's own video latency (about 50–150 ms)
does not add to this, because the subtitle is matched to the same frame.

## The subtitle panel

**In the game window, with the mouse:**

| Action | Effect |
|---|---|
| Drag with the left button | the panel goes where you drop it (saved) |
| Wheel over the panel | bigger or smaller text |
| Right-click the panel | back to the English subtitle's position |
| `r` key | sample text when there is no subtitle (so you can set it up) |

When covering the English subtitle is on, the original English text is hidden by a
blurred, soft-edged version of the game picture, even if you drag the panel elsewhere
(the `h` key turns it off).

**In the settings window** (hide or show it with the `p` key):

- **🌐 Program language:** at the top of the window. *Automatic* follows the Windows
  display language; or pick English, Magyar, Deutsch, Español, Français, Polski,
  Português. This is only the language of the program itself; the subtitle target
  language is separate and has many more options.
- **Target language:** pick one of 30 languages from the list, or type any language code
  + Enter (e.g. `lt`, `et`).
- **Translator:** Automatic, DeepL, Claude, ChatGPT or Google (free).
- **API key:** the **Save** button stores it in Windows Credential Manager (encrypted, tied
  to your user account), so you don't have to type it again next time. The **Delete**
  button removes it. The key never goes into `config.json` or the repository.
- **AI model:** can be changed for Claude or GPT, applied with Enter.
- **Font size, Background opacity %:** sliders, the effect is visible immediately.
- **Cover the English subtitle**, and **Move the panel back to the English subtitle**.
- **Thin font (like in GTA):** the speaker's name in turquoise, the text in white, with a
  faint shadow.
- **Spoken lines only (“Name: text” lines):** only translates “Name: text” dialogue
  lines; mission text, signs, menus and the phone-call display are not translated (on by
  default). A nameless TV-show subtitle is translated too when it
  sounds spoken (a question, an exclamation, a first-person sentence, or dim greyish-white
  letters); imperative mission text ("Go to the Vice City Port.") is not.
- **Streamer mode:** slurs (the n-word, homophobic and other slurs, in Hungarian too) are
  shown starred out, e.g. `n****`. A separate switch, off by default; it only changes the
  drawn text, not the translation, so it can be turned on and off at any time.
- At the bottom, the status: which translator is running, **the measured translation time
  (ms)**, and if something is wrong, what the problem is.

Every change takes effect immediately and is saved to `config.json` (except the keys).

## Translators

| Translator | Speed (estimated) | Quality | Key (environment variable) |
|---|---|---|---|
| **DeepL** | fastest, about 150–400 ms | very good | `DEEPL_API_KEY` – *DeepL API Free* is free up to 500,000 characters a month |
| **Claude** (Haiku) | about 400–900 ms | the most natural, understands slang and the previous lines | `ANTHROPIC_API_KEY` – console.anthropic.com |
| **ChatGPT** (GPT mini) | about 400–1000 ms | very good | `OPENAI_API_KEY` – platform.openai.com |
| **Google** | about 100–300 ms | average | no key needed, but it sometimes blocks too many requests |

**Important:** a Claude Pro or ChatGPT Plus subscription **does not give you an API
key**. The API has to be topped up separately on the pages above and is paid per use.
The subtitles of a full playthrough typically cost a few dollars.

To set a key permanently with an environment variable instead of the panel's **Save**
button (open a new PowerShell window afterwards):

```powershell
setx DEEPL_API_KEY "your-key-here"
```

**Automatic** mode picks, in this order, the first one you have a key for:
DeepL → Claude → ChatGPT → Google. The AI translators also get the previous 3 lines,
so they get gender, forms of address and tone right more often. If a translation fails,
the English text is shown with a prefix saying the translation failed.

### How long is the delay?

Measured time from the subtitle appearing to the translated text (simulating a 300 ms
and a 700 ms translation, on the same machine):

| Translation time | Average total delay |
|---|---|
| ~300 ms (DeepL-like) | **about 0.5 s** (0.44–0.6) |
| ~700 ms (Claude/GPT-like) | **about 0.9 s** (0.82–1.09) |
| repeated line | OCR only, about 0.1–0.25 s |

What makes it this fast:

- Translation **starts on the first read**, without waiting for confirmation. If the
  second read gives the same text, the finished translation appears immediately.
- The connection to the translator is **kept open**, so there is no new HTTPS handshake
  for every line (the measurement above does not include this yet; it is even faster in
  real use).
- It reads every 0.15 s, and one read takes about 75 ms.

The measured translation time is also shown live at the bottom of the settings window.

## Webcam (optional)

Only starts if you ask for it: `.\inditas.ps1 --device 1 --webcam 0`. You can drag the
picture with the mouse, resize it with the wheel, and hide it with the `c` key.

## Audio

The overlay window only shows the picture. For sound, pick one:

- **Windows audio input:** *Settings → System → Sound → the capture card's input →
  Properties → “Listen to this device”*. The simplest, but it can add 50–100 ms of delay.
- **Headphones on the controller** (PS5 DualSense / Xbox controller jack). No delay.
- **Passthrough to the TV**, muted: sound from the TV, picture from the PC.

## Keys in the game window

| Key | Function |
|---|---|
| `q` / `Esc` | quit |
| `p` | hide / show the settings window |
| `f` | full screen on/off |
| `h` | cover the original English subtitle on/off |
| `r` | outline the watched area + sample text in the panel's place |
| `o` | subtitle panel on/off |
| `c` | webcam picture on/off |

## Command-line options

| Option | Meaning |
|---|---|
| `--auto` | find the source automatically: Veyra, PS Remote Play, capture card (the exe's default) |
| `--device N` | capture card number |
| `--capture-api auto\|dshow\|msmf` | how to open the capture card; if it won't open through DirectShow, try `msmf` (config.json: `capture_api`) |
| `--remote-play` | puts the subtitles over the PS Remote Play window |
| `--window "TITLE"` | over any window, by its exact title or the start of it (e.g. `--window Veyra`) |
| `--lang CODE` | target language, e.g. `hu`, `de`, `es` |
| `--translator NAME` | `auto`, `deepl`, `claude`, `openai`, `google` |
| `--windowed` | starts in a window |
| `--no-panel` | starts without the settings window |
| `--image picture.png` | test with a screenshot, without a capture card |
| `--list-devices` | list video devices |

## Settings (`config.json`)

| Key | Default | Meaning |
|---|---|---|
| `device_index` | `0` | capture card number |
| `capture_width` / `capture_height` / `capture_fps` | 1920 / 1080 / 60 | requested picture size |
| `subtitle_region` | 0.06 / 0.55 / 0.94 / 0.98 | where the program looks for the English subtitle |
| `ocr_interval_seconds` | `0.1` | how often it reads (OCR takes ~0.1 s, it can't go faster) |
| `min_text_score` | `0.6` | OCR results less certain than this are dropped |
| `stable_reads` | `2` | translates after this many matching reads |
| `similarity_threshold` | `0.85` | text at least this similar counts as the same |
| `hold_seconds` | `0.8` | how long the translation stays after the English text disappears |
| `translator` | `"auto"` | `auto`, `deepl`, `claude`, `openai`, `google` |
| `target_lang` | `"hu"` | target language code |
| `ui_language` | `"auto"` | language of the program itself: `"auto"` (Windows display language) or e.g. `"en"`, `"hu"` |
| `claude_model` | `"claude-haiku-5-5"` | Claude model |
| `openai_model` | `"gpt-4.1-mini"` | GPT model |
| `hide_original` | `true` | cover the English text |
| `visible_in_recordings` | `true` | the subtitle shows up in Medal/OBS monitor recordings and on stream |
| `only_spoken_lines` | `true` | translate only “Name: text” dialogue lines |
| `streamer_mode` | `false` | streamer mode: slurs starred out |
| `clean_output` | `false` | Remote Play mode: delayed clean picture for viewers in a separate window (only the finished translation) |
| `clean_output_delay_seconds` | `1.0` | delay of the clean picture (0.5–5 s); delay the stream audio by the same amount |
| `obs_text_file` | `true` | also writes the shown subtitle to `obs_felirat.txt` next to the exe (for an OBS Text source) |
| `fast_capture` | `true` | Remote Play mode: capture the subtitle band continuously on its own thread (faster cover and reading); set `false` to go back to one capture per read |
| `row_reading` | `true` | Remote Play mode: read row by row – only a newly appeared subtitle row is read (faster translation, less CPU); needs `fast_capture` and a game picture at least 1600 px wide, otherwise the whole bar is read every time. Set `false` to always read the whole bar |
| `panel_opacity` | `245` | panel background, 0 (transparent) – 255 (fully opaque) |
| `panel_width_ratio` | `0.9` | widest translation as a fraction of the picture width (0.3–1.0); in preview, drag the panel's side |
| `panel_height_ratio` | `null` | lowest panel height as a fraction of the picture height (0.04–0.5); `null` = fits the text; in preview, drag the panel's top or bottom |
| `subtitle_x` / `subtitle_y` | `null` | panel center as a fraction of the picture; `null` = at the English subtitle |
| `font_path` | Segoe UI (thin, GTA style) | any `.ttf`; the panel's “Thin font” switch toggles between Segoe UI and Segoe UI Bold |
| `speaker_color` | `"#7ECFD6"` | color of the speaker's name (`#RRGGBB`), like in GTA's subtitles |
| `font_size_ratio` | `0.034` | font size as a fraction of the picture height (0.015–0.08) |
| `fullscreen` | `true` | starts in full screen |

For **Chinese, Japanese, Korean or Arabic** target languages, Segoe UI cannot display
every character. Set `font_path` to e.g. `C:/Windows/Fonts/msyh.ttc` (Chinese),
`YuGothB.ttc` (Japanese) or `malgunbd.ttf` (Korean).

## Testing without a capture card

```powershell
.\inditas.ps1 --image screenshot.png --windowed
```

Developer tests:

```powershell
.\.venv\Scripts\python -m pip install pytest==9.2.0
.\.venv\Scripts\python -m pytest tests
```

## Limitations

- Machine translation: puns and references sometimes come out oddly (less often with an
  AI translator).
- If the game draws subtitles in a different place or color, `subtitle_region` has to be
  adjusted. It is tuned for white text.
- The speaker's name (“Jason: …”) is not translated, only the sentence.
- For Arabic, the panel does not reorder text for right-to-left writing.
- In Remote Play mode the subtitle appears on the Remote Play window, not on the TV.
  If the PS5 blocks the Remote Play picture in a scene, the subtitle is missing there too.
