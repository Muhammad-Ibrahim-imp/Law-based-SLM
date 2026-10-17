# Verification — 17 October 2026

Environment: Windows 11, Flutter stable 3.47.7, Dart 3.13.5.

| Check | Result |
| --- | --- |
| `dart format --output=none --set-exit-if-changed lib test` | Passed, no formatting changes |
| `flutter analyze --no-pub` | Passed, no issues |
| `flutter test --no-pub --reporter expanded` | Passed, 16 tests |
| Desktop dark/light and mobile dark render captures | Generated and visually inspected with the supplied fonts and Material icons |
| Android ARM64 debug APK | Attempted; stopped during Gradle distribution setup after its Java downloader made no progress. No APK was produced; native Android compilation is unverified. |
| Physical Android phone | Not tested: no phone was connected |
| Windows native build | Not tested: Visual Studio C++ toolchain is absent; Flutter also reports missing symlink permission/Developer Mode |
| Linux native build | Not tested: this host is Windows |
| macOS/iOS native builds | Not tested: these require a Mac and Apple toolchains/signing |
| Real model generation | Not tested: no model supplied; no simulated provider is active |
| Accounts, sync, share links, attachments | Deferred by agreement; current controls explain their unavailable state |

The tests cover delayed-load write protection; concurrent-send protection; chat/project operations; role-aware history; generation failure, cancellation, and regeneration; legacy attachment metadata; ciphertext tampering and missing keys; file-save recovery; and mobile/desktop/short-landscape layout.

Tests use memory stores and fake model streams only inside `test/`. Widget captures are not installed-app screenshots. OS secure-storage integration and hardware inference still require platform/device testing.

Initial dependency resolution downloaded the packages and wrote `pubspec.lock`, but its final Windows/Linux plugin symlink step reported Windows Developer Mode is disabled. Checks ran with `--no-pub` against those resolved dependencies. See `SETUP.md` before running `flutter pub get` on this machine.

Android API 36, build-tools 36.0.0, platform-tools, NDK 28.2.13676358, and Java were installed. The native build reached Gradle's `assembleDebug` setup, but its distribution download remained at zero bytes. A direct-download diagnostic was also stopped incomplete. This was a tooling/setup attempt, not a successful native compilation. No model files were downloaded.

All existing tracked repository files were left unchanged. The pre-existing untracked `prompt for UI and backend.txt` and `scripts/data/main` were not added to this work.
