import 'dart:convert';
import 'dart:math';
import 'storage_blob_stub.dart' if (dart.library.io) 'storage_blob_io.dart';
import 'package:cryptography/cryptography.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'models.dart';
import 'blob_storage.dart';

abstract interface class WorkspaceStore {
  Future<Workspace> load();
  Future<void> save(Workspace workspace);
}

class EncryptedWorkspaceStore implements WorkspaceStore {
  EncryptedWorkspaceStore({
    FlutterSecureStorage? secureStorage,
    BlobStore? blobStorage,
  }) : _secure =
           secureStorage ??
           const FlutterSecureStorage(
             mOptions: MacOsOptions(usesDataProtectionKeychain: false),
           ),
       _blob = blobStorage ?? BlobStorage();
  static const _keyName = 'legal_chat_workspace_key_v1';
  final FlutterSecureStorage _secure;
  final BlobStore _blob;
  final _aes = AesGcm.with256bits();
  Future<SecretKey> _key({required bool create}) async {
    var raw = await _secure.read(key: _keyName);
    if (raw == null && create) {
      final random = Random.secure();
      final bytes = List<int>.generate(32, (_) => random.nextInt(256));
      raw = base64Encode(bytes);
      await _secure.write(key: _keyName, value: raw);
    }
    if (raw == null) {
      throw const FormatException(
        'Workspace key is missing; encrypted data was preserved',
      );
    }
    final bytes = base64Decode(raw);
    if (bytes.length != 32) {
      throw const FormatException('Invalid workspace encryption key');
    }
    return SecretKey(bytes);
  }

  @override
  Future<Workspace> load() async {
    final contents = await _blob.read();
    if (contents == null) return Workspace();
    final envelope = jsonDecode(contents) as Map<String, dynamic>;
    final box = SecretBox(
      base64Decode(envelope['ciphertext']),
      nonce: base64Decode(envelope['nonce']),
      mac: Mac(base64Decode(envelope['mac'])),
    );
    final clear = await _aes.decrypt(
      box,
      secretKey: await _key(create: false),
      aad: utf8.encode('legal-chat-workspace-v1'),
    );
    return Workspace.decode(utf8.decode(clear));
  }

  @override
  Future<void> save(Workspace workspace) async {
    final box = await _aes.encrypt(
      utf8.encode(workspace.encode()),
      secretKey: await _key(create: true),
      aad: utf8.encode('legal-chat-workspace-v1'),
    );
    await _blob.writeAtomic(
      jsonEncode({
        'nonce': base64Encode(box.nonce),
        'ciphertext': base64Encode(box.cipherText),
        'mac': base64Encode(box.mac.bytes),
      }),
    );
  }
}
