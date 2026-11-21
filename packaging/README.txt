GTA 6 subtitle translator - no-install version
==============================================

1. Unzip the zip into any folder, e.g. C:\GTA6Felirat.
2. Start PS Remote Play and connect to your console.
3. Start GTA6Felirat.exe. The subtitles appear over the Remote Play window,
   and they also show up in Medal/OBS monitor recordings.

On first start Windows SmartScreen may warn you ("Windows protected your PC"):
More info -> Run anyway.

Program language: choose it at the top of the settings window ("Program language").
Automatic follows the Windows display language.

API key (DeepL, Claude, OpenAI): enter it in the settings window. The program stores
it in Windows Credential Manager, not in this folder.

Settings: config.json is next to the exe; the settings window saves your changes there.

Command-line options (from a command prompt or a shortcut), e.g.:
  GTA6Felirat.exe --lang de --translator claude
  GTA6Felirat.exe --window Xbox
  GTA6Felirat.exe --device 1          (capture card mode)
  GTA6Felirat.exe --help

Full guide: https://github.com/GraceLutz/GTAVISubtitles

Magyar leírás: https://github.com/GraceLutz/GTAVISubtitles/blob/main/README.hu.md
