import 'dart:io';
import 'package:path_provider/path_provider.dart';
import 'blob_storage.dart';

class BlobStorage implements BlobStore {
  BlobStorage({Future<Directory> Function()? directory})
    : _directory = directory ?? getApplicationSupportDirectory;
  final Future<Directory> Function() _directory;
  Future<File> _file() async {
    final dir = await _directory();
    await dir.create(recursive: true);
    return File('${dir.path}${Platform.pathSeparator}workspace.enc');
  }

  @override
  Future<String?> read() async {
    final file = await _file();
    if (await file.exists()) return file.readAsString();
    final backup = File('${file.path}.bak');
    if (!await backup.exists()) return null;
    await backup.rename(file.path);
    return file.readAsString();
  }

  @override
  Future<void> writeAtomic(String value) async {
    final file = await _file();
    final temp = File('${file.path}.tmp');
    final backup = File('${file.path}.bak');
    await temp.writeAsString(value, flush: true);
    final hadOriginal = await file.exists();
    if (hadOriginal) {
      if (await backup.exists()) await backup.delete();
      await file.rename(backup.path);
    }
    try {
      await temp.rename(file.path);
      if (await backup.exists()) await backup.delete();
    } catch (_) {
      if (!await file.exists() && await backup.exists()) {
        await backup.rename(file.path);
      }
      rethrow;
    }
  }
}
