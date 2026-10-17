import 'dart:convert';
import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'package:legal_chat/blob_storage.dart';
import 'package:legal_chat/models.dart';
import 'package:legal_chat/secure_store.dart';

class TestSecureStorage extends Fake implements FlutterSecureStorage {
  String? value;
  @override
  Future<String?> read({
    required String key,
    AppleOptions? iOptions,
    AndroidOptions? aOptions,
    LinuxOptions? lOptions,
    WindowsOptions? wOptions,
    AppleOptions? mOptions,
    WebOptions? webOptions,
  }) async => value;
  @override
  Future<void> write({
    required String key,
    required String? value,
    AppleOptions? iOptions,
    AndroidOptions? aOptions,
    LinuxOptions? lOptions,
    WindowsOptions? wOptions,
    AppleOptions? mOptions,
    WebOptions? webOptions,
  }) async {
    this.value = value;
  }
}

class MemoryBlob implements BlobStore {
  String? value;
  @override
  Future<String?> read() async => value;
  @override
  Future<void> writeAtomic(String value) async {
    this.value = value;
  }
}

void main() {
  test(
    'encrypted workspace hides content and detects ciphertext corruption',
    () async {
      final secure = TestSecureStorage();
      final blob = MemoryBlob();
      final store = EncryptedWorkspaceStore(
        secureStorage: secure,
        blobStorage: blob,
      );
      final workspace = Workspace(
        chats: [
          Chat(
            id: 'x',
            title: 'Sensitive title',
            updatedAt: DateTime.utc(2026),
          ),
        ],
      );
      await store.save(workspace);
      expect(blob.value!.contains('Sensitive title'), isFalse);
      expect((await store.load()).chats.single.title, 'Sensitive title');
      final envelope = jsonDecode(blob.value!) as Map<String, dynamic>;
      envelope['ciphertext'] = base64Encode([0]);
      blob.value = jsonEncode(envelope);
      await expectLater(store.load(), throwsA(anything));
    },
  );
  test(
    'missing OS key never creates a replacement key for existing ciphertext',
    () async {
      final secure = TestSecureStorage();
      final blob = MemoryBlob();
      final store = EncryptedWorkspaceStore(
        secureStorage: secure,
        blobStorage: blob,
      );
      await store.save(Workspace());
      secure.value = null;
      await expectLater(store.load(), throwsA(isA<FormatException>()));
      expect(blob.value, isNotNull);
    },
  );
}
