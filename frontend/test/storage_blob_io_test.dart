import 'dart:io';
import 'package:flutter_test/flutter_test.dart';
import 'package:legal_chat/storage_blob_io.dart';

void main() {
  test(
    'atomic replacement recovers the last committed copy after an interrupted rename',
    () async {
      final dir = await Directory.systemTemp.createTemp('legal-chat-atomic');
      addTearDown(() => dir.delete(recursive: true));
      final storage = BlobStorage(directory: () async => dir);
      await storage.writeAtomic('previous encrypted blob');
      final file = File('${dir.path}${Platform.pathSeparator}workspace.enc');
      await file.rename('${file.path}.bak');
      expect(await storage.read(), 'previous encrypted blob');
      await storage.writeAtomic('next encrypted blob');
      expect(await storage.read(), 'next encrypted blob');
    },
  );
}
