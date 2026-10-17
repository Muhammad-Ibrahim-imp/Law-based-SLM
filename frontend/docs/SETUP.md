# Platform setup

This folder is independent of the Python training project. Run Flutter commands from `frontend`, not the repository root. The app itself does not need internet; installing SDKs and resolving packages does.

## Tools installed on this PC

- Flutter: `C:\development\flutter` (stable 3.47.7, Dart 3.13.5).
- Java: `C:\development\java\jdk-17.0.20.1+1`.
- Android SDK: `C:\development\android-sdk`.
- Flutter's user PATH and Android SDK/JDK settings were configured. Restart terminals to pick up PATH changes.

If a terminal has not picked up PATH, use `& 'C:\development\flutter\bin\flutter.bat'` in place of `flutter` in PowerShell. `--no-version-check --suppress-analytics` can precede a Flutter command to avoid an optional SDK update check during setup.

## Android physical device first

1. Enable developer options and USB debugging on the phone, connect by USB, and approve its debugging prompt.
2. Run `flutter devices` and copy the phone's device ID.
3. Run `flutter pub get`, then `flutter run -d <device-id>`.
4. To create an installable development APK, run `flutter build apk --debug --target-platform android-arm64`. Use another target architecture if the actual device requires it.

The minimum Android version is API 23. The generated project compiles against API 36 and uses Flutter's NDK version. No emulator or model is required to use the frontend. A model is required for answers.

Android backup is disabled in `android/app/src/main/AndroidManifest.xml`, preventing encrypted history from being restored without its device key. Uninstalling the app can remove its local history and key.

The Java version on this PC encountered a Windows Unix-domain socket error during SDK downloads. The setup workaround, scoped to a PowerShell session, was:

```powershell
$env:JAVA_TOOL_OPTIONS = '-Djdk.net.unixdomain.tmpdir=C:\development\no-unix-sockets'
```

That path must remain nonexistent for Java to use its TCP fallback. This is a local toolchain workaround, not an application setting.

## Windows

Enable Windows Developer Mode in Settings to permit Flutter plugin symbolic links. Install Visual Studio with **Desktop development with C++**, the Windows SDK, CMake tools, and **C++ ATL** (required by secure storage). Then run `flutter pub get` and `flutter run -d windows`.

## Linux

Build on Linux using Flutter's Linux desktop prerequisites: Clang, CMake, Ninja, pkg-config, GTK development libraries, and libsecret development/runtime libraries. A running desktop Secret Service/keyring is required to store the encryption key. On Ubuntu, the secure-storage prerequisites include `libsecret-1-dev` and `libsecret-1-0`. Run `flutter pub get`, then `flutter run -d linux`.

## macOS and iOS

Build on a Mac with Xcode and its command-line tools. Use `flutter doctor` to check the local Apple toolchain. For iOS, select a signing team in Xcode, connect and trust the phone, enable its Developer Mode, and run `flutter run -d <device-id>` using the ID from `flutter devices`. For macOS, run `flutter run -d macos`.

`ios/Runner/Runner.entitlements` is linked to Debug, Profile, and Release configurations for Keychain access. macOS uses the legacy local Keychain through `MacOsOptions(usesDataProtectionKeychain: false)`, avoiding a shared-keychain provisioning requirement. Neither platform shares local encryption keys between devices.

The generated `com.example` application identifiers and development signing configuration must be replaced before store distribution. Android release signing currently uses Flutter's generated debug-key configuration; do not publish that build as a production release.

## Checks

```sh
dart format --output=none --set-exit-if-changed lib test
flutter analyze --no-pub
flutter test --no-pub
```

Dependencies and generated platform runners cover all five targets, but successful Dart/widget tests do not prove native packaging or OS key-store behavior on every platform. See `VERIFICATION.md` for the checks actually completed on this PC.
